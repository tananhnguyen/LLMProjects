"""
Benchmark runner for evaluating LLM models.
Provides utilities for running benchmarks and collecting results.
"""

import os
import json
import logging
import time
import torch
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
from tqdm import tqdm

from src.models.teacher_model import TeacherLLM
from src.models.student_model import StudentLLM
from src.evaluation.metrics import EvaluationMetrics, SQuADEvaluator, MTBenchEvaluator


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runner."""
    
    # Benchmark datasets
    squad_path: Optional[str] = "data/evaluation/squad/dev-v2.0.json"
    mtbench_path: Optional[str] = "data/evaluation/korean/komt-bench.json"
    
    # Models to evaluate
    teacher_model_name: Optional[str] = None
    student_model_name: Optional[str] = None
    
    # Output directory for results
    output_dir: str = "results/benchmarks"
    
    # Whether to save predictions
    save_predictions: bool = True
    
    # Batch size for inference
    batch_size: int = 8
    
    # Maximum sequence length
    max_length: int = 512
    
    # Device for inference
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class BenchmarkRunner:
    """
    Runner for LLM benchmarks.
    """
    
    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the benchmark runner.
        
        Args:
            config: Benchmark configuration
            logger: Logger instance
        """
        self.config = config or BenchmarkConfig()
        
        # Set up logging
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize evaluation metrics
        self.metrics = EvaluationMetrics()
        
        # Initialize evaluators
        self.squad_evaluator = SQuADEvaluator(logger=self.logger)
        self.mtbench_evaluator = MTBenchEvaluator(logger=self.logger)
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        self.logger.info(f"Initialized benchmark runner with config: {self.config}")
        
    def load_models(
        self,
        teacher_model_name: Optional[str] = None,
        student_model_name: Optional[str] = None
    ) -> Tuple[Optional[TeacherLLM], Optional[StudentLLM]]:
        """
        Load teacher and student models.
        
        Args:
            teacher_model_name: Name of the teacher model
            student_model_name: Name of the student model
            
        Returns:
            Tuple of (teacher_model, student_model)
        """
        teacher_model = None
        student_model = None
        
        # Use config values if not provided
        teacher_model_name = teacher_model_name or self.config.teacher_model_name
        student_model_name = student_model_name or self.config.student_model_name
        
        # Load teacher model if specified
        if teacher_model_name:
            self.logger.info(f"Loading teacher model: {teacher_model_name}")
            teacher_model = TeacherLLM(
                model_name=teacher_model_name,
                device=self.config.device
            )
            
        # Load student model if specified
        if student_model_name:
            self.logger.info(f"Loading student model: {student_model_name}")
            student_model = StudentLLM(
                model_name=student_model_name,
                device=self.config.device
            )
            
        return teacher_model, student_model
        
    def run_squad_benchmark(
        self,
        model: Union[TeacherLLM, StudentLLM],
        dataset_path: Optional[str] = None,
        output_prefix: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run SQuAD benchmark on a model.
        
        Args:
            model: Model to evaluate
            dataset_path: Path to SQuAD dataset
            output_prefix: Prefix for output files
            
        Returns:
            Dictionary with benchmark results
        """
        # Use config value if not provided
        dataset_path = dataset_path or self.config.squad_path
        
        # Generate output prefix if not provided
        if output_prefix is None:
            output_prefix = f"{model.model_name.replace('/', '_')}_squad"
            
        self.logger.info(f"Running SQuAD benchmark on {model.model_name}")
        
        # Load dataset
        dataset = self.squad_evaluator.load_squad_dataset(dataset_path)
        
        # Generate predictions
        predictions = {}
        
        for article in tqdm(dataset["data"], desc="Processing articles"):
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                
                for qa in paragraph["qas"]:
                    qa_id = qa["id"]
                    question = qa["question"]
                    
                    # Generate answer
                    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
                    answer = model.generate(prompt, max_length=self.config.max_length)
                    
                    # Store prediction
                    predictions[qa_id] = answer
                    
        # Evaluate predictions
        results = self.squad_evaluator.evaluate(predictions, dataset)
        
        # Save predictions if enabled
        if self.config.save_predictions:
            predictions_file = os.path.join(self.config.output_dir, f"{output_prefix}_predictions.json")
            self.squad_evaluator.save_predictions(predictions, predictions_file)
            
        # Save results
        results_file = os.path.join(self.config.output_dir, f"{output_prefix}_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
            
        self.logger.info(f"SQuAD benchmark results: {results}")
        
        return results
        
    def run_mtbench_benchmark(
        self,
        model: Union[TeacherLLM, StudentLLM],
        dataset_path: Optional[str] = None,
        output_prefix: Optional[str] = None,
        language: str = "ko"
    ) -> Dict[str, Any]:
        """
        Run MT-Bench benchmark on a model.
        
        Args:
            model: Model to evaluate
            dataset_path: Path to MT-Bench dataset
            output_prefix: Prefix for output files
            language: Language of the dataset
            
        Returns:
            Dictionary with benchmark results
        """
        # Use config value if not provided
        dataset_path = dataset_path or self.config.mtbench_path
        
        # Generate output prefix if not provided
        if output_prefix is None:
            output_prefix = f"{model.model_name.replace('/', '_')}_mtbench_{language}"
            
        self.logger.info(f"Running MT-Bench benchmark on {model.model_name}")
        
        # Load dataset
        dataset = self.mtbench_evaluator.load_mtbench_dataset(dataset_path)
        
        # Generate predictions
        predictions = []
        
        for example in tqdm(dataset, desc="Processing examples"):
            instruction = example.get("instruction", "")
            
            # Generate response
            response = model.generate(instruction, max_length=self.config.max_length)
            
            # Store prediction
            predictions.append({
                "instruction": instruction,
                "output": response,
                "category": example.get("category", "general")
            })
            
        # Evaluate predictions
        results = self.mtbench_evaluator.evaluate(predictions, dataset)
        
        # Save predictions if enabled
        if self.config.save_predictions:
            predictions_file = os.path.join(self.config.output_dir, f"{output_prefix}_predictions.json")
            self.mtbench_evaluator.save_predictions(predictions, predictions_file)
            
        # Save results
        results_file = os.path.join(self.config.output_dir, f"{output_prefix}_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
            
        self.logger.info(f"MT-Bench benchmark results: {results}")
        
        return results
        
    def run_all_benchmarks(
        self,
        teacher_model_name: Optional[str] = None,
        student_model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run all benchmarks on teacher and student models.
        
        Args:
            teacher_model_name: Name of the teacher model
            student_model_name: Name of the student model
            
        Returns:
            Dictionary with all benchmark results
        """
        self.logger.info("Running all benchmarks")
        
        # Load models
        teacher_model, student_model = self.load_models(teacher_model_name, student_model_name)
        
        results = {
            "teacher": {},
            "student": {}
        }
        
        # Run benchmarks on teacher model
        if teacher_model:
            self.logger.info(f"Running benchmarks on teacher model: {teacher_model.model_name}")
            
            # SQuAD benchmark
            squad_results = self.run_squad_benchmark(
                teacher_model,
                output_prefix="teacher_squad"
            )
            results["teacher"]["squad"] = squad_results
            
            # MT-Bench benchmark
            mtbench_results = self.run_mtbench_benchmark(
                teacher_model,
                output_prefix="teacher_mtbench"
            )
            results["teacher"]["mtbench"] = mtbench_results
            
        # Run benchmarks on student model
        if student_model:
            self.logger.info(f"Running benchmarks on student model: {student_model.model_name}")
            
            # SQuAD benchmark
            squad_results = self.run_squad_benchmark(
                student_model,
                output_prefix="student_squad"
            )
            results["student"]["squad"] = squad_results
            
            # MT-Bench benchmark
            mtbench_results = self.run_mtbench_benchmark(
                student_model,
                output_prefix="student_mtbench"
            )
            results["student"]["mtbench"] = mtbench_results
            
        # Save combined results
        combined_results_file = os.path.join(self.config.output_dir, "combined_results.json")
        with open(combined_results_file, "w") as f:
            json.dump(results, f, indent=2)
            
        self.logger.info(f"All benchmark results saved to {combined_results_file}")
        
        return results
        
    def compare_models(
        self,
        results: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Compare teacher and student models based on benchmark results.
        
        Args:
            results: Dictionary with benchmark results
            
        Returns:
            DataFrame with model comparison
        """
        self.logger.info("Comparing teacher and student models")
        
        # Extract metrics for comparison
        comparison_data = []
        
        # SQuAD metrics
        if "squad" in results.get("teacher", {}) and "squad" in results.get("student", {}):
            teacher_em = results["teacher"]["squad"].get("exact_match", 0)
            teacher_f1 = results["teacher"]["squad"].get("f1", 0)
            student_em = results["student"]["squad"].get("exact_match", 0)
            student_f1 = results["student"]["squad"].get("f1", 0)
            
            comparison_data.append({
                "Benchmark": "SQuAD",
                "Metric": "Exact Match",
                "Teacher": teacher_em,
                "Student": student_em,
                "Difference": student_em - teacher_em,
                "Relative": f"{(student_em / teacher_em * 100) - 100:.2f}%" if teacher_em > 0 else "N/A"
            })
            
            comparison_data.append({
                "Benchmark": "SQuAD",
                "Metric": "F1 Score",
                "Teacher": teacher_f1,
                "Student": student_f1,
                "Difference": student_f1 - teacher_f1,
                "Relative": f"{(student_f1 / teacher_f1 * 100) - 100:.2f}%" if teacher_f1 > 0 else "N/A"
            })
            
        # MT-Bench metrics
        if "mtbench" in results.get("teacher", {}) and "mtbench" in results.get("student", {}):
            teacher_score = results["teacher"]["mtbench"].get("overall_score", 0)
            student_score = results["student"]["mtbench"].get("overall_score", 0)
            
            comparison_data.append({
                "Benchmark": "MT-Bench",
                "Metric": "Overall Score",
                "Teacher": teacher_score,
                "Student": student_score,
                "Difference": student_score - teacher_score,
                "Relative": f"{(student_score / teacher_score * 100) - 100:.2f}%" if teacher_score > 0 else "N/A"
            })
            
            # Category scores
            teacher_categories = results["teacher"]["mtbench"].get("category_scores", {})
            student_categories = results["student"]["mtbench"].get("category_scores", {})
            
            for category in set(teacher_categories.keys()) | set(student_categories.keys()):
                teacher_cat_score = teacher_categories.get(category, 0)
                student_cat_score = student_categories.get(category, 0)
                
                comparison_data.append({
                    "Benchmark": "MT-Bench",
                    "Metric": f"Category: {category}",
                    "Teacher": teacher_cat_score,
                    "Student": student_cat_score,
                    "Difference": student_cat_score - teacher_cat_score,
                    "Relative": f"{(student_cat_score / teacher_cat_score * 100) - 100:.2f}%" if teacher_cat_score > 0 else "N/A"
                })
                
        # Create DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison
        comparison_file = os.path.join(self.config.output_dir, "model_comparison.csv")
        comparison_df.to_csv(comparison_file, index=False)
        
        self.logger.info(f"Model comparison saved to {comparison_file}")
        
        return comparison_df
