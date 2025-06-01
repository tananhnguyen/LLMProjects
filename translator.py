"""
Translation system for English-Korean language pair.
Provides utilities for translation using pre-trained models.
"""

import os
import torch
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    MarianMTModel,
    MarianTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    pipeline
)
from dataclasses import dataclass


@dataclass
class TranslationConfig:
    """Configuration for translation models."""
    
    # Model type
    model_type: str = "marian"  # "marian", "t5", "custom"
    
    # Model names for different directions
    en_to_ko_model: str = "Helsinki-NLP/opus-mt-en-ko"
    ko_to_en_model: str = "Helsinki-NLP/opus-mt-ko-en"
    
    # Maximum sequence length
    max_length: int = 512
    
    # Batch size for translation
    batch_size: int = 8
    
    # Whether to use cached models
    use_cache: bool = True
    
    # Cache directory
    cache_dir: Optional[str] = None


class Translator:
    """
    Translator for English-Korean language pair.
    """
    
    def __init__(
        self,
        config: Optional[TranslationConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the translator.
        
        Args:
            config: Translation configuration
            device: Device to run the models on
            logger: Logger instance
        """
        self.config = config or TranslationConfig()
        self.device = device
        
        # Set up logging
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize models
        self.en_to_ko_model = None
        self.en_to_ko_tokenizer = None
        self.ko_to_en_model = None
        self.ko_to_en_tokenizer = None
        
        self.logger.info(f"Initialized translator with config: {self.config}")
        
    def load_models(self):
        """Load translation models."""
        self.logger.info("Loading translation models")
        
        # Load English to Korean model
        self.en_to_ko_model, self.en_to_ko_tokenizer = self._load_model(
            self.config.en_to_ko_model,
            self.config.model_type
        )
        
        # Load Korean to English model
        self.ko_to_en_model, self.ko_to_en_tokenizer = self._load_model(
            self.config.ko_to_en_model,
            self.config.model_type
        )
        
        self.logger.info("Translation models loaded")
        
    def _load_model(
        self,
        model_name: str,
        model_type: str
    ) -> Tuple[Any, Any]:
        """
        Load a translation model.
        
        Args:
            model_name: Name of the model to load
            model_type: Type of the model
            
        Returns:
            Tuple of (model, tokenizer)
        """
        self.logger.info(f"Loading {model_type} model: {model_name}")
        
        if model_type == "marian":
            # Load MarianMT model
            tokenizer = MarianTokenizer.from_pretrained(
                model_name,
                cache_dir=self.config.cache_dir if self.config.use_cache else None
            )
            
            model = MarianMTModel.from_pretrained(
                model_name,
                cache_dir=self.config.cache_dir if self.config.use_cache else None
            ).to(self.device)
            
        elif model_type == "t5":
            # Load T5 model
            tokenizer = T5Tokenizer.from_pretrained(
                model_name,
                cache_dir=self.config.cache_dir if self.config.use_cache else None
            )
            
            model = T5ForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=self.config.cache_dir if self.config.use_cache else None
            ).to(self.device)
            
        elif model_type == "custom":
            # Load custom model
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.config.cache_dir if self.config.use_cache else None
            )
            
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                cache_dir=self.config.cache_dir if self.config.use_cache else None
            ).to(self.device)
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        return model, tokenizer
        
    def translate_en_to_ko(
        self,
        text: Union[str, List[str]],
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None
    ) -> Union[str, List[str]]:
        """
        Translate English text to Korean.
        
        Args:
            text: English text or list of texts
            batch_size: Batch size for translation
            max_length: Maximum sequence length
            
        Returns:
            Korean translation(s)
        """
        # Load models if not loaded
        if self.en_to_ko_model is None:
            self.load_models()
            
        # Use default values if not specified
        batch_size = batch_size or self.config.batch_size
        max_length = max_length or self.config.max_length
        
        # Check if input is a single string
        is_single_text = isinstance(text, str)
        if is_single_text:
            text = [text]
            
        # Translate in batches
        translations = []
        
        for i in range(0, len(text), batch_size):
            batch_texts = text[i:i+batch_size]
            
            # Tokenize
            inputs = self.en_to_ko_tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)
            
            # Generate translations
            with torch.no_grad():
                outputs = self.en_to_ko_model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True
                )
                
            # Decode translations
            batch_translations = self.en_to_ko_tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True
            )
            
            translations.extend(batch_translations)
            
        # Return single string if input was a single string
        if is_single_text:
            return translations[0]
            
        return translations
        
    def translate_ko_to_en(
        self,
        text: Union[str, List[str]],
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None
    ) -> Union[str, List[str]]:
        """
        Translate Korean text to English.
        
        Args:
            text: Korean text or list of texts
            batch_size: Batch size for translation
            max_length: Maximum sequence length
            
        Returns:
            English translation(s)
        """
        # Load models if not loaded
        if self.ko_to_en_model is None:
            self.load_models()
            
        # Use default values if not specified
        batch_size = batch_size or self.config.batch_size
        max_length = max_length or self.config.max_length
        
        # Check if input is a single string
        is_single_text = isinstance(text, str)
        if is_single_text:
            text = [text]
            
        # Translate in batches
        translations = []
        
        for i in range(0, len(text), batch_size):
            batch_texts = text[i:i+batch_size]
            
            # Tokenize
            inputs = self.ko_to_en_tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)
            
            # Generate translations
            with torch.no_grad():
                outputs = self.ko_to_en_model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True
                )
                
            # Decode translations
            batch_translations = self.ko_to_en_tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True
            )
            
            translations.extend(batch_translations)
            
        # Return single string if input was a single string
        if is_single_text:
            return translations[0]
            
        return translations
        
    def translate(
        self,
        text: Union[str, List[str]],
        source_lang: str,
        target_lang: str,
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None
    ) -> Union[str, List[str]]:
        """
        Translate text between languages.
        
        Args:
            text: Text or list of texts to translate
            source_lang: Source language code ('en' or 'ko')
            target_lang: Target language code ('en' or 'ko')
            batch_size: Batch size for translation
            max_length: Maximum sequence length
            
        Returns:
            Translation(s)
        """
        # Validate language codes
        if source_lang not in ["en", "ko"] or target_lang not in ["en", "ko"]:
            raise ValueError("Only 'en' and 'ko' language codes are supported")
            
        # English to Korean
        if source_lang == "en" and target_lang == "ko":
            return self.translate_en_to_ko(text, batch_size, max_length)
            
        # Korean to English
        elif source_lang == "ko" and target_lang == "en":
            return self.translate_ko_to_en(text, batch_size, max_length)
            
        # Same language (no translation needed)
        elif source_lang == target_lang:
            return text
            
        else:
            raise ValueError(f"Unsupported language pair: {source_lang}-{target_lang}")
            
    def detect_language(self, text: str) -> str:
        """
        Detect the language of a text.
        
        Args:
            text: Text to detect language for
            
        Returns:
            Language code ('en' or 'ko')
        """
        # Simple heuristic: check for Korean characters
        korean_chars = 0
        english_chars = 0
        
        for char in text:
            # Korean Unicode range (Hangul syllables)
            if '\uAC00' <= char <= '\uD7A3':
                korean_chars += 1
            # English characters
            elif 'a' <= char.lower() <= 'z':
                english_chars += 1
                
        # Determine language based on character counts
        if korean_chars > english_chars:
            return "ko"
        else:
            return "en"
            
    def translate_auto(
        self,
        text: Union[str, List[str]],
        target_lang: str,
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None
    ) -> Union[str, List[str]]:
        """
        Automatically detect source language and translate to target language.
        
        Args:
            text: Text or list of texts to translate
            target_lang: Target language code ('en' or 'ko')
            batch_size: Batch size for translation
            max_length: Maximum sequence length
            
        Returns:
            Translation(s)
        """
        # Check if input is a single string
        is_single_text = isinstance(text, str)
        if is_single_text:
            # Detect language
            source_lang = self.detect_language(text)
            
            # Translate
            return self.translate(text, source_lang, target_lang, batch_size, max_length)
            
        else:
            # Process each text individually
            translations = []
            
            for t in text:
                # Detect language
                source_lang = self.detect_language(t)
                
                # Translate
                translation = self.translate(t, source_lang, target_lang, batch_size, max_length)
                translations.append(translation)
                
            return translations


class TranslationPipeline:
    """
    Pipeline for translation tasks using Hugging Face's pipeline API.
    """
    
    def __init__(
        self,
        en_to_ko_model: str = "Helsinki-NLP/opus-mt-en-ko",
        ko_to_en_model: str = "Helsinki-NLP/opus-mt-ko-en",
        device: int = -1 if not torch.cuda.is_available() else 0,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the translation pipeline.
        
        Args:
            en_to_ko_model: Model name for English to Korean translation
            ko_to_en_model: Model name for Korean to English translation
            device: Device to run the pipeline on (-1 for CPU, 0+ for GPU)
            logger: Logger instance
        """
        self.en_to_ko_model = en_to_ko_model
        self.ko_to_en_model = ko_to_en_model
        self.device = device
        
        # Set up logging
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize pipelines
        self.en_to_ko_pipeline = None
        self.ko_to_en_pipeline = None
        
        self.logger.info(f"Initialized translation pipeline with models: {en_to_ko_model}, {ko_to_en_model}")
        
    def load_pipelines(self):
        """Load translation pipelines."""
        self.logger.info("Loading translation pipelines")
        
        # Load English to Korean pipeline
        self.en_to_ko_pipeline = pipeline(
            "translation",
            model=self.en_to_ko_model,
            device=self.device
        )
        
        # Load Korean to English pipeline
        self.ko_to_en_pipeline = pipeline(
            "translation",
            model=self.ko_to_en_model,
            device=self.device
        )
        
        self.logger.info("Translation pipelines loaded")
        
    def translate_en_to_ko(
        self,
        text: Union[str, List[str]],
        max_length: int = 512
    ) -> Union[str, List[str]]:
        """
        Translate English text to Korean.
        
        Args:
            text: English text or list of texts
            max_length: Maximum sequence length
            
        Returns:
            Korean translation(s)
        """
        # Load pipeline if not loaded
        if self.en_to_ko_pipeline is None:
            self.load_pipelines()
            
        # Translate
        result = self.en_to_ko_pipeline(
            text,
            max_length=max_length
        )
        
        # Extract translations
        if isinstance(text, str):
            return result[0]["translation_text"]
        else:
            return [r["translation_text"] for r in result]
            
    def translate_ko_to_en(
        self,
        text: Union[str, List[str]],
        max_length: int = 512
    ) -> Union[str, List[str]]:
        """
        Translate Korean text to English.
        
        Args:
            text: Korean text or list of texts
            max_length: Maximum sequence length
            
        Returns:
            English translation(s)
        """
        # Load pipeline if not loaded
        if self.ko_to_en_pipeline is None:
            self.load_pipelines()
            
        # Translate
        result = self.ko_to_en_pipeline(
            text,
            max_length=max_length
        )
        
        # Extract translations
        if isinstance(text, str):
            return result[0]["translation_text"]
        else:
            return [r["translation_te
(Content truncated due to size limit. Use line ranges to read in chunks)