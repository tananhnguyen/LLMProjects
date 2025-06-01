"""
PubMed data retrieval and processing module.
Provides utilities for accessing and processing PubMed articles.
"""

import os
import json
import logging
import time
import requests
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
from Bio import Entrez


@dataclass
class PubMedConfig:
    """Configuration for PubMed data retrieval."""
    
    # API keys and credentials
    email: str = "user@example.com"  # Required for NCBI E-utilities
    api_key: Optional[str] = None  # Optional NCBI API key for higher rate limits
    
    # Rate limiting parameters
    requests_per_second: float = 3.0  # Max requests per second without API key
    
    # Search parameters
    max_results: int = 100  # Maximum number of results to retrieve per query
    
    # Data storage
    cache_dir: str = "data/pubmed/cache"
    
    # Fields to retrieve
    fields: List[str] = field(default_factory=lambda: [
        "title", "abstract", "authors", "journal", "pubdate", "doi", "pmid"
    ])


class PubMedClient:
    """
    Client for retrieving and processing PubMed data.
    """
    
    def __init__(
        self,
        config: Optional[PubMedConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the PubMed client.
        
        Args:
            config: PubMed configuration
            logger: Logger instance
        """
        self.config = config or PubMedConfig()
        
        # Set up logging
        self.logger = logger or logging.getLogger(__name__)
        
        # Set up Entrez
        Entrez.email = self.config.email
        if self.config.api_key:
            Entrez.api_key = self.config.api_key
            
        # Create cache directory
        os.makedirs(self.config.cache_dir, exist_ok=True)
        
        self.logger.info(f"Initialized PubMed client with email: {self.config.email}")
        
    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        sort: str = "relevance",
        min_date: Optional[str] = None,
        max_date: Optional[str] = None
    ) -> List[str]:
        """
        Search PubMed for articles matching the query.
        
        Args:
            query: PubMed search query
            max_results: Maximum number of results to retrieve
            sort: Sort order ("relevance", "pub_date", "first_author")
            min_date: Minimum publication date (YYYY/MM/DD)
            max_date: Maximum publication date (YYYY/MM/DD)
            
        Returns:
            List of PubMed IDs (PMIDs)
        """
        self.logger.info(f"Searching PubMed for: {query}")
        
        # Use default max_results if not specified
        max_results = max_results or self.config.max_results
        
        # Build date range filter if provided
        date_filter = ""
        if min_date or max_date:
            min_date_str = min_date or "1900/01/01"
            max_date_str = max_date or datetime.now().strftime("%Y/%m/%d")
            date_filter = f" AND {min_date_str}:{max_date_str}[Date - Publication]"
            
        # Perform search
        try:
            # First get the count of results
            search_handle = Entrez.esearch(
                db="pubmed",
                term=query + date_filter,
                retmax=0,
                sort=sort
            )
            search_results = Entrez.read(search_handle)
            search_handle.close()
            
            total_count = int(search_results["Count"])
            self.logger.info(f"Found {total_count} results for query: {query}")
            
            # Limit to max_results
            retmax = min(total_count, max_results)
            
            # Retrieve the PMIDs
            search_handle = Entrez.esearch(
                db="pubmed",
                term=query + date_filter,
                retmax=retmax,
                sort=sort
            )
            search_results = Entrez.read(search_handle)
            search_handle.close()
            
            pmids = search_results["IdList"]
            self.logger.info(f"Retrieved {len(pmids)} PMIDs")
            
            return pmids
            
        except Exception as e:
            self.logger.error(f"Error searching PubMed: {e}")
            return []
            
    def fetch_articles(
        self,
        pmids: List[str],
        batch_size: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Fetch article details for a list of PMIDs.
        
        Args:
            pmids: List of PubMed IDs
            batch_size: Number of articles to fetch in each batch
            
        Returns:
            List of article details
        """
        self.logger.info(f"Fetching {len(pmids)} articles from PubMed")
        
        articles = []
        
        # Process in batches to avoid overloading the API
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i+batch_size]
            
            try:
                # Fetch articles
                fetch_handle = Entrez.efetch(
                    db="pubmed",
                    id=",".join(batch_pmids),
                    retmode="xml"
                )
                
                # Parse XML
                records = Entrez.read(fetch_handle)
                fetch_handle.close()
                
                # Process each article
                for record in records["PubmedArticle"]:
                    article = self._parse_pubmed_article(record)
                    articles.append(article)
                    
                self.logger.info(f"Fetched batch of {len(batch_pmids)} articles")
                
                # Rate limiting
                time.sleep(1.0 / self.config.requests_per_second)
                
            except Exception as e:
                self.logger.error(f"Error fetching articles: {e}")
                
        self.logger.info(f"Fetched {len(articles)} articles in total")
        
        return articles
        
    def _parse_pubmed_article(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse a PubMed article record.
        
        Args:
            record: PubMed article record
            
        Returns:
            Parsed article details
        """
        article = {}
        
        try:
            # Extract PMID
            article["pmid"] = record["MedlineCitation"]["PMID"]
            
            # Extract article data
            article_data = record["MedlineCitation"]["Article"]
            
            # Extract title
            if "ArticleTitle" in article_data:
                article["title"] = article_data["ArticleTitle"]
                
            # Extract abstract
            if "Abstract" in article_data:
                abstract_texts = article_data["Abstract"].get("AbstractText", [])
                if isinstance(abstract_texts, list):
                    # Handle structured abstracts
                    abstract_parts = []
                    for abstract_part in abstract_texts:
                        if hasattr(abstract_part, "attributes") and "Label" in abstract_part.attributes:
                            label = abstract_part.attributes["Label"]
                            abstract_parts.append(f"{label}: {abstract_part}")
                        else:
                            abstract_parts.append(str(abstract_part))
                    article["abstract"] = " ".join(abstract_parts)
                else:
                    article["abstract"] = str(abstract_texts)
            else:
                article["abstract"] = ""
                
            # Extract journal
            if "Journal" in article_data:
                journal_data = article_data["Journal"]
                journal_name = journal_data.get("Title", "")
                
                # Extract publication date
                if "JournalIssue" in journal_data and "PubDate" in journal_data["JournalIssue"]:
                    pub_date = journal_data["JournalIssue"]["PubDate"]
                    year = pub_date.get("Year", "")
                    month = pub_date.get("Month", "")
                    day = pub_date.get("Day", "")
                    
                    if year:
                        if month and day:
                            article["pubdate"] = f"{year}/{month}/{day}"
                        elif month:
                            article["pubdate"] = f"{year}/{month}"
                        else:
                            article["pubdate"] = year
                            
                article["journal"] = journal_name
                
            # Extract authors
            if "AuthorList" in article_data:
                authors = []
                for author in article_data["AuthorList"]:
                    if "LastName" in author and "ForeName" in author:
                        authors.append(f"{author['LastName']} {author['ForeName']}")
                    elif "LastName" in author:
                        authors.append(author["LastName"])
                    elif "CollectiveName" in author:
                        authors.append(author["CollectiveName"])
                        
                article["authors"] = ", ".join(authors)
                
            # Extract DOI
            if "ELocationID" in article_data:
                for location in article_data["ELocationID"]:
                    if location.attributes.get("EIdType") == "doi":
                        article["doi"] = str(location)
                        break
                        
        except Exception as e:
            self.logger.error(f"Error parsing article: {e}")
            
        return article
        
    def search_and_fetch(
        self,
        query: str,
        max_results: Optional[int] = None,
        sort: str = "relevance",
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search PubMed and fetch article details in one step.
        
        Args:
            query: PubMed search query
            max_results: Maximum number of results to retrieve
            sort: Sort order ("relevance", "pub_date", "first_author")
            min_date: Minimum publication date (YYYY/MM/DD)
            max_date: Maximum publication date (YYYY/MM/DD)
            use_cache: Whether to use cached results
            
        Returns:
            List of article details
        """
        # Check cache first
        if use_cache:
            cache_file = self._get_cache_file_path(query, max_results, sort, min_date, max_date)
            if os.path.exists(cache_file):
                self.logger.info(f"Loading cached results from {cache_file}")
                with open(cache_file, "r") as f:
                    return json.load(f)
                    
        # Search for PMIDs
        pmids = self.search(query, max_results, sort, min_date, max_date)
        
        # Fetch article details
        articles = self.fetch_articles(pmids)
        
        # Cache results
        if use_cache:
            cache_file = self._get_cache_file_path(query, max_results, sort, min_date, max_date)
            self.logger.info(f"Caching results to {cache_file}")
            with open(cache_file, "w") as f:
                json.dump(articles, f)
                
        return articles
        
    def _get_cache_file_path(
        self,
        query: str,
        max_results: Optional[int],
        sort: str,
        min_date: Optional[str],
        max_date: Optional[str]
    ) -> str:
        """
        Generate a cache file path for a query.
        
        Args:
            query: PubMed search query
            max_results: Maximum number of results
            sort: Sort order
            min_date: Minimum publication date
            max_date: Maximum publication date
            
        Returns:
            Cache file path
        """
        # Create a hash of the query parameters
        import hashlib
        params = f"{query}_{max_results}_{sort}_{min_date}_{max_date}"
        query_hash = hashlib.md5(params.encode()).hexdigest()
        
        return os.path.join(self.config.cache_dir, f"pubmed_{query_hash}.json")
        
    def save_articles_to_csv(
        self,
        articles: List[Dict[str, Any]],
        output_file: str
    ):
        """
        Save articles to a CSV file.
        
        Args:
            articles: List of article details
            output_file: Output file path
        """
        self.logger.info(f"Saving {len(articles)} articles to {output_file}")
        
        # Convert to DataFrame
        df = pd.DataFrame(articles)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        
    def save_articles_to_json(
        self,
        articles: List[Dict[str, Any]],
        output_file: str
    ):
        """
        Save articles to a JSON file.
        
        Args:
            articles: List of article details
            output_file: Output file path
        """
        self.logger.info(f"Saving {len(articles)} articles to {output_file}")
        
        # Save to JSON
        with open(output_file, "w") as f:
            json.dump(articles, f, indent=2)


class PubMedLinkCollector:
    """
    Collector for PubMed article links and citations.
    """
    
    def __init__(
        self,
        config: Optional[PubMedConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the PubMed link collector.
        
        Args:
            config: PubMed configuration
            logger: Logger instance
        """
        self.config = config or PubMedConfig()
        
        # Set up logging
        self.logger = logger or logging.getLogger(__name__)
        
        # Set up Entrez
        Entrez.email = self.config.email
        if self.config.api_key:
            Entrez.api_key = self.config.api_key
            
        self.logger.info(f"Initialized PubMed link collector with email: {self.config.email}")
        
    def get_citations(
        self,
        pmid: str,
        max_results: int = 100
    ) -> List[str]:
        """
        Get PMIDs of articles that cite the given article.
        
        Args:
            pmid: PubMed ID
            max_results: Maximum number of results to retrieve
            
        Returns:
            List of citing PMIDs
        """
        self.logger.info(f"Getting citations for PMID: {pmid}")
        
        try:
            # Use elink to get citing articles
            link_handle = Entrez.elink(
                dbfrom="pubmed",
                db="pubmed",
                id=pmid,
                linkname="pubmed_pubmed_citedin"
            )
            link_results = Entrez.read(link_handle)
            link_handle.close()
            
            # Extract PMIDs
            citing_pmids = []
            
            if link_results and len(link_results) > 0:
                link_set = link_results[0]
                if "LinkSetDb" in link_set and len(link_set["LinkSetDb"]) > 0:
                    citations = link_set["LinkSetDb"][0].get("Link", [])
                    citing_pmids = [link["Id"] for link in citations]
                    
            # Limit to max_results
            citing_pmids = citing_pmids[:max_results]
            
            self.logger.info(f"Found {len(citing_pmids)} citing articles")
            
            return citing_pmids
            
        except Exception as e:
            self.logger.error(f"Error getting citations: {e}")
            return []
            
    def get_related_articles(
        self,
        pmid: str,
        max_results: int = 100
    ) -> List[str]:
      
(Content truncated due to size limit. Use line ranges to read in chunks)