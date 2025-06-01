"""
Evaluation metrics and utilities for LLM benchmarking.
Provides common metrics for NLP tasks and LLM evaluation.
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
from collections import Counter

# For SQuAD evaluation
from transformers import squad_metrics


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    
    # Metrics to compute
    metrics: List[str] = field(default_factory=lambda: [
        "exact_match", "f1", "rouge", "bleu", "meteor"
    ])
    
    # Language settings
    languages: List[str] = field(default_factory=lambda: ["en", "ko"])
    
    # Output directory for results
    output_dir: str = "results/evaluation"
    
    # Whether to save detailed results
    save_details: bool = True


class EvaluationMetrics:
    """
    Evaluation metrics for LLM tasks.
    """
    
    def __init__(
        self,
        config: Optional[EvaluationConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the evaluation metrics.
        
        Args:
            config: Evaluation configuration
            logger: Logger instance
        """
        self.config = config or EvaluationConfig()
        
        # Set up logging
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize metric functions
        self.metric_functions = {
            "exact_match": self.compute_exact_match,
            "f1": self.compute_f1,
            "rouge": self.compute_rouge,
            "bleu": self.compute_bleu,
            "meteor": self.compute_meteor
        }
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        self.logger.info(f"Initialized evaluation metrics with config: {self.config}")
        
    def normalize_text(self, text: str, language: str = "en") -> str:
        """
        Normalize text for evaluation.
        
        Args:
            text: Text to normalize
            language: Language of the text
            
        Returns:
            Normalized text
        """
        if language == "en":
            # English normalization
            text = text.lower()
            # Remove punctuation
            import string
            for punct in string.punctuation:
                text = text.replace(punct, " ")
            # Remove extra whitespace
            text = " ".join(text.split())
            
        elif language == "ko":
            # Korean normalization
            text = text.lower()
            # Remove punctuation (including Korean punctuation)
            import string
            korean_punct = "，。、；：''""《》「」『』【】〔〕（）［］｛｝〈〉《》"
            for punct in string.punctuation + korean_punct:
                text = text.replace(punct, " ")
            # Remove extra whitespace
            text = " ".join(text.split())
            
        return text
        
    def compute_exact_match(
        self,
        predictions: List[str],
        references: List[str],
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Compute exact match score.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            language: Language of the texts
            
        Returns:
            Dictionary with exact match score
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
            
        exact_matches = []
        
        for pred, ref in zip(predictions, references):
            # Normalize texts
            norm_pred = self.normalize_text(pred, language)
            norm_ref = self.normalize_text(ref, language)
            
            # Check exact match
            exact_match = int(norm_pred == norm_ref)
            exact_matches.append(exact_match)
            
        # Compute average
        exact_match_score = np.mean(exact_matches)
        
        return {
            "exact_match": exact_match_score,
            "details": {
                "per_example": exact_matches
            }
        }
        
    def compute_f1(
        self,
        predictions: List[str],
        references: List[str],
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Compute F1 score based on token overlap.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            language: Language of the texts
            
        Returns:
            Dictionary with F1 score
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
            
        f1_scores = []
        
        for pred, ref in zip(predictions, references):
            # Normalize texts
            norm_pred = self.normalize_text(pred, language)
            norm_ref = self.normalize_text(ref, language)
            
            # Tokenize
            pred_tokens = norm_pred.split()
            ref_tokens = norm_ref.split()
            
            # Count token overlap
            common_tokens = Counter(pred_tokens) & Counter(ref_tokens)
            num_common = sum(common_tokens.values())
            
            # Compute precision, recall, F1
            if len(pred_tokens) == 0:
                precision = 0.0
            else:
                precision = num_common / len(pred_tokens)
                
            if len(ref_tokens) == 0:
                recall = 0.0
            else:
                recall = num_common / len(ref_tokens)
                
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
                
            f1_scores.append(f1)
            
        # Compute average
        f1_score = np.mean(f1_scores)
        
        return {
            "f1": f1_score,
            "details": {
                "per_example": f1_scores
            }
        }
        
    def compute_rouge(
        self,
        predictions: List[str],
        references: List[str],
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Compute ROUGE scores.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            language: Language of the texts
            
        Returns:
            Dictionary with ROUGE scores
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
            
        try:
            from rouge import Rouge
            rouge = Rouge()
            
            # Ensure non-empty strings
            valid_pairs = []
            for pred, ref in zip(predictions, references):
                if len(pred.strip()) > 0 and len(ref.strip()) > 0:
                    valid_pairs.append((pred, ref))
                    
            if not valid_pairs:
                return {
                    "rouge-1": {"f": 0.0, "p": 0.0, "r": 0.0},
                    "rouge-2": {"f": 0.0, "p": 0.0, "r": 0.0},
                    "rouge-l": {"f": 0.0, "p": 0.0, "r": 0.0}
                }
                
            valid_preds, valid_refs = zip(*valid_pairs)
            
            # Compute ROUGE scores
            scores = rouge.get_scores(valid_preds, valid_refs, avg=True)
            
            return scores
            
        except ImportError:
            self.logger.warning("Rouge package not found. Install with: pip install rouge")
            return {
                "rouge-1": {"f": 0.0, "p": 0.0, "r": 0.0},
                "rouge-2": {"f": 0.0, "p": 0.0, "r": 0.0},
                "rouge-l": {"f": 0.0, "p": 0.0, "r": 0.0}
            }
            
    def compute_bleu(
        self,
        predictions: List[str],
        references: List[str],
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Compute BLEU score.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            language: Language of the texts
            
        Returns:
            Dictionary with BLEU score
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
            
        try:
            from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
            import nltk
            
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
                
            # Tokenize
            if language == "en":
                from nltk.tokenize import word_tokenize
                tokenize_func = word_tokenize
            elif language == "ko":
                # For Korean, use simple whitespace tokenization if no specialized tokenizer
                tokenize_func = lambda x: x.split()
                
            # Prepare data for corpus_bleu
            tokenized_refs = [[tokenize_func(ref)] for ref in references]
            tokenized_preds = [tokenize_func(pred) for pred in predictions]
            
            # Compute BLEU score
            smoothing = SmoothingFunction().method1
            bleu_score = corpus_bleu(tokenized_refs, tokenized_preds, smoothing_function=smoothing)
            
            return {
                "bleu": bleu_score
            }
            
        except ImportError:
            self.logger.warning("NLTK package not found. Install with: pip install nltk")
            return {
                "bleu": 0.0
            }
            
    def compute_meteor(
        self,
        predictions: List[str],
        references: List[str],
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Compute METEOR score.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            language: Language of the texts
            
        Returns:
            Dictionary with METEOR score
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
            
        # METEOR is primarily for English, so return 0 for other languages
        if language != "en":
            return {
                "meteor": 0.0
            }
            
        try:
            from nltk.translate.meteor_score import meteor_score
            import nltk
            
            try:
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('wordnet')
            except LookupError:
                nltk.download('punkt')
                nltk.download('wordnet')
                
            # Tokenize
            from nltk.tokenize import word_tokenize
            
            meteor_scores = []
            
            for pred, ref in zip(predictions, references):
                # Tokenize
                tokenized_pred = word_tokenize(pred)
                tokenized_ref = word_tokenize(ref)
                
                # Compute METEOR score
                score = meteor_score([tokenized_ref], tokenized_pred)
                meteor_scores.append(score)
                
            # Compute average
            meteor_score_avg = np.mean(meteor_scores)
            
            return {
                "meteor": meteor_score_avg,
                "details": {
                    "per_example": meteor_scores
                }
            }
            
        except ImportError:
            self.logger.warning("NLTK package not found. Install with: pip install nltk")
            return {
                "meteor": 0.0
            }
            
    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
        metrics: Optional[List[str]] = None,
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Evaluate predictions against references using multiple metrics.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            metrics: List of metrics to compute
            language: Language of the texts
            
        Returns:
            Dictionary with evaluation results
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
            
        # Use default metrics if not specified
        metrics = metrics or self.config.metrics
        
        # Compute metrics
        results = {}
        
        for metric in metrics:
            if metric in self.metric_functions:
                metric_func = self.metric_functions[metric]
                metric_results = metric_func(predictions, references, language)
                results.update(metric_results)
            else:
                self.logger.warning(f"Unknown metric: {metric}")
                
        return results
        
    def save_results(
        self,
        results: Dict[str, Any],
        output_file: Optional[str] = None
    ):
        """
        Save evaluation results to a file.
        
        Args:
            results: Evaluation results
            output_file: Output file path
        """
        if output_file is None:
            # Generate default output file name
            import time
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_file = os.path.join(self.config.output_dir, f"eval_results_{timestamp}.json")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save results
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
            
        self.logger.info(f"Saved evaluation results to {output_file}")


class SQuADEvaluator:
    """
    Evaluator for SQuAD-style question answering tasks.
    """
    
    def __init__(
        self,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the SQuAD evaluator.
        
        Args:
            logger: Logger instance
        """
        # Set up logging
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info("Initialized SQuAD evaluator")
        
    def evaluate(
        self,
        predictions: Dict[str, str],
        dataset: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate predictions on SQuAD dataset.
        
        Args:
            predictions: Dictionary mapping example IDs to predicted answers
            dataset: SQuAD dataset
            
        Returns:
            Dictionary with evaluation results
        """
        self.logger.info("Evaluating predictions on SQuAD dataset")
        
        # Prepare references
        references = {}
        for article in dataset["data"]:
            for paragraph in article["paragraphs"]:
                for qa in paragraph["qas"]:
                    qa_id = qa["id"]
                    answers = [answer["text"] for answer in qa["answers"]]
                    references[qa_id] = answers
                    
        # Compute metrics
        exact_match, f1 = self._compute_squad_metrics(predictions, references)
        
        results = {
            "exact_match": exact_match,
            "f1": f1
        }
        
        self.logger.info(f"SQuAD evaluation results: {results}")
        
        return results
        
    def _compute_squad_metrics(
        self,
        predictions: Dict[str, str],
        references: Dict[str, List[str]]
    ) -> Tuple[float, float]:

(Content truncated due to size limit. Use line ranges to read in chunks)