"""
End-to-end testing script for the LLM optimization and knowledge distillation project.
Tests all major components and their integration.
"""

import os
import sys
import logging
import argparse
import torch
from typing import Dict, List, Optional, Union, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("test_results.log")
    ]
)
logger = logging.getLogger("end_to_end_test")

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.models.teacher_model import TeacherLLM
from src.models.student_model import StudentLLM
from src.distillation.knowledge_distillation import KnowledgeDistillation
from src.rag.rag_pipeline import RAGPipeline
from src.rag.document_processor import DocumentProcessor
from src.rag.embedding_generator import EmbeddingGenerator
from src.rag.vector_database import VectorDatabase
from src.optimization.peft_optimizer import PEFTOptimizer
from src.optimization.quantizer import Quantizer
from src.optimization.pruner import Pruner
from src.agent.llm_agent import LLMAgent, AgentTool
from src.translation.translator import Translator
from src.pubmed.pubmed_client import PubMedClient
from src.pubmed.medical_rag import MedicalRAGPipeline
from src.evaluation.metrics import EvaluationMetrics
from src.evaluation.benchmark import BenchmarkRunner


def test_teacher_student_models():
    """Test teacher and student LLM models."""
    logger.info("Testing teacher and student models")
    
    # Initialize teacher model
    teacher_model = TeacherLLM(
        model_name="google/flan-t5-base",  # Using a smaller model for testing
        device="cpu"
    )
    
    # Initialize student model
    student_model = StudentLLM(
        model_name="google/flan-t5-small",  # Using a smaller model for testing
        device="cpu"
    )
    
    # Test generation
    test_prompt = "Translate the following English text to French: 'Hello, how are you?'"
    
    logger.info(f"Testing generation with prompt: {test_prompt}")
    
    teacher_output = teacher_model.generate(test_prompt)
    logger.info(f"Teacher model output: {teacher_output}")
    
    student_output = student_model.generate(test_prompt)
    logger.info(f"Student model output: {student_output}")
    
    return {
        "teacher_model": teacher_model,
        "student_model": student_model,
        "teacher_output": teacher_output,
        "student_output": student_output
    }


def test_knowledge_distillation(teacher_model=None, student_model=None):
    """Test knowledge distillation pipeline."""
    logger.info("Testing knowledge distillation")
    
    # Initialize models if not provided
    if teacher_model is None:
        teacher_model = TeacherLLM(
            model_name="google/flan-t5-base",
            device="cpu"
        )
        
    if student_model is None:
        student_model = StudentLLM(
            model_name="google/flan-t5-small",
            device="cpu"
        )
        
    # Initialize knowledge distillation
    distillation = KnowledgeDistillation(
        teacher_model=teacher_model,
        student_model=student_model,
        alpha=0.5,
        temperature=2.0
    )
    
    # Test distillation with a small batch
    test_inputs = [
        "Translate to French: Hello world",
        "Summarize: The quick brown fox jumps over the lazy dog."
    ]
    
    logger.info(f"Testing distillation with {len(test_inputs)} examples")
    
    # Run a single step of distillation (without actual training)
    loss = distillation.compute_distillation_loss(test_inputs)
    
    logger.info(f"Distillation loss: {loss}")
    
    return {
        "distillation": distillation,
        "loss": loss
    }


def test_rag_pipeline():
    """Test RAG pipeline."""
    logger.info("Testing RAG pipeline")
    
    # Initialize components
    document_processor = DocumentProcessor()
    embedding_generator = EmbeddingGenerator(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Small model for testing
        device="cpu"
    )
    vector_database = VectorDatabase(dimension=384)  # Dimension for all-MiniLM-L6-v2
    
    # Initialize RAG pipeline
    rag_pipeline = RAGPipeline(
        document_processor=document_processor,
        embedding_generator=embedding_generator,
        vector_database=vector_database
    )
    
    # Test document processing and retrieval
    test_docs = [
        {
            "content": "The capital of France is Paris. It is known for the Eiffel Tower.",
            "metadata": {"source": "test", "id": "1"}
        },
        {
            "content": "Berlin is the capital of Germany. It has a rich history.",
            "metadata": {"source": "test", "id": "2"}
        },
        {
            "content": "Tokyo is the capital of Japan and is a major economic center.",
            "metadata": {"source": "test", "id": "3"}
        }
    ]
    
    logger.info(f"Adding {len(test_docs)} documents to RAG pipeline")
    
    # Add documents to RAG pipeline
    doc_ids = rag_pipeline.add_documents(test_docs)
    
    # Test retrieval
    test_query = "What is the capital of France?"
    
    logger.info(f"Testing retrieval with query: {test_query}")
    
    results = rag_pipeline.retrieve(test_query)
    
    logger.info(f"Retrieved {len(results)} documents")
    for i, result in enumerate(results):
        logger.info(f"Result {i+1}: {result['content']} (Score: {result['score']})")
        
    return {
        "rag_pipeline": rag_pipeline,
        "doc_ids": doc_ids,
        "results": results
    }


def test_optimization_techniques():
    """Test optimization techniques (PEFT, quantization, pruning)."""
    logger.info("Testing optimization techniques")
    
    # Initialize a small model for testing
    model_name = "google/flan-t5-small"
    
    # Test PEFT
    logger.info("Testing PEFT optimization")
    peft_optimizer = PEFTOptimizer()
    
    # Just initialize the config (don't actually apply PEFT to save time)
    peft_config = peft_optimizer.create_lora_config(
        r=8,
        lora_alpha=16,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none"
    )
    
    logger.info(f"Created PEFT config: {peft_config}")
    
    # Test quantization
    logger.info("Testing quantization")
    quantizer = Quantizer()
    
    # Just initialize the config (don't actually apply quantization)
    quant_config = {
        "bits": 8,
        "group_size": 128,
        "method": "dynamic"
    }
    
    logger.info(f"Created quantization config: {quant_config}")
    
    # Test pruning
    logger.info("Testing pruning")
    pruner = Pruner()
    
    # Just initialize the config (don't actually apply pruning)
    prune_config = pruner.config
    
    logger.info(f"Created pruning config: {prune_config}")
    
    return {
        "peft_config": peft_config,
        "quant_config": quant_config,
        "prune_config": prune_config
    }


def test_agent_and_translation():
    """Test LLM agent and translation system."""
    logger.info("Testing LLM agent and translation")
    
    # Initialize a small model for the agent
    model_name = "google/flan-t5-small"
    
    # Initialize agent
    agent = LLMAgent(
        model_name=model_name,
        device="cpu"
    )
    
    # Add a simple tool
    def calculator_tool(expression):
        try:
            return eval(expression)
        except Exception as e:
            return f"Error: {str(e)}"
            
    agent.add_tool(AgentTool(
        name="calculator",
        description="Calculate mathematical expressions",
        function=calculator_tool
    ))
    
    # Test agent with a simple query
    test_query = "What is 2 + 2?"
    
    logger.info(f"Testing agent with query: {test_query}")
    
    # Note: We're not actually running the agent to avoid LLM API calls
    # In a real test, we would call agent.run(test_query)
    
    # Initialize translator
    translator = Translator(device="cpu")
    
    # Test translation
    test_text = "Hello, how are you?"
    
    logger.info(f"Testing translation with text: {test_text}")
    
    # Note: We're not actually running the translation to avoid loading large models
    # In a real test, we would call translator.translate_en_to_ko(test_text)
    
    return {
        "agent": agent,
        "translator": translator
    }


def test_pubmed_and_medical_rag():
    """Test PubMed integration and medical RAG."""
    logger.info("Testing PubMed integration and medical RAG")
    
    # Initialize PubMed client
    pubmed_client = PubMedClient()
    
    # Test PubMed search (without actually making API calls)
    test_query = "diabetes treatment"
    
    logger.info(f"Testing PubMed search with query: {test_query}")
    
    # Note: We're not actually making API calls to avoid rate limits
    # In a real test, we would call pubmed_client.search_and_fetch(test_query, max_results=5)
    
    # Initialize medical RAG pipeline
    medical_rag = MedicalRAGPipeline()
    
    # Note: We're not actually running the medical RAG pipeline to avoid loading large models
    # In a real test, we would add documents and test retrieval
    
    return {
        "pubmed_client": pubmed_client,
        "medical_rag": medical_rag
    }


def test_evaluation_and_benchmarking():
    """Test evaluation metrics and benchmarking."""
    logger.info("Testing evaluation metrics and benchmarking")
    
    # Initialize evaluation metrics
    metrics = EvaluationMetrics()
    
    # Test metrics with simple examples
    predictions = ["The capital of France is Paris.", "Berlin is in Germany."]
    references = ["Paris is the capital of France.", "Berlin is the capital of Germany."]
    
    logger.info("Testing evaluation metrics")
    
    results = metrics.evaluate(predictions, references, metrics=["exact_match", "f1"])
    
    logger.info(f"Evaluation results: {results}")
    
    # Initialize benchmark runner
    benchmark_runner = BenchmarkRunner()
    
    # Note: We're not actually running benchmarks to avoid loading large models and datasets
    # In a real test, we would load models and run benchmarks
    
    return {
        "metrics": metrics,
        "results": results,
        "benchmark_runner": benchmark_runner
    }


def run_end_to_end_test():
    """Run end-to-end test of all components."""
    logger.info("Starting end-to-end test")
    
    results = {}
    
    # Test teacher and student models
    try:
        results["models"] = test_teacher_student_models()
    except Exception as e:
        logger.error(f"Error testing models: {str(e)}")
        results["models"] = {"error": str(e)}
        
    # Test knowledge distillation
    try:
        results["distillation"] = test_knowledge_distillation(
            results.get("models", {}).get("teacher_model"),
            results.get("models", {}).get("student_model")
        )
    except Exception as e:
        logger.error(f"Error testing distillation: {str(e)}")
        results["distillation"] = {"error": str(e)}
        
    # Test RAG pipeline
    try:
        results["rag"] = test_rag_pipeline()
    except Exception as e:
        logger.error(f"Error testing RAG: {str(e)}")
        results["rag"] = {"error": str(e)}
        
    # Test optimization techniques
    try:
        results["optimization"] = test_optimization_techniques()
    except Exception as e:
        logger.error(f"Error testing optimization: {str(e)}")
        results["optimization"] = {"error": str(e)}
        
    # Test agent and translation
    try:
        results["agent_translation"] = test_agent_and_translation()
    except Exception as e:
        logger.error(f"Error testing agent and translation: {str(e)}")
        results["agent_translation"] = {"error": str(e)}
        
    # Test PubMed and medical RAG
    try:
        results["pubmed_medical"] = test_pubmed_and_medical_rag()
    except Exception as e:
        logger.error(f"Error testing PubMed and medical RAG: {str(e)}")
        results["pubmed_medical"] = {"error": str(e)}
        
    # Test evaluation and benchmarking
    try:
        results["evaluation"] = test_evaluation_and_benchmarking()
    except Exception as e:
        logger.error(f"Error testing evaluation: {str(e)}")
        results["evaluation"] = {"error": str(e)}
        
    # Summarize results
    success_count = sum(1 for k, v in results.items() if "error" not in v)
    total_count = len(results)
    
    logger.info(f"End-to-end test completed: {success_count}/{total_count} components passed")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run end-to-end tests for LLM project")
    parser.add_argument("--component", type=str, help="Test specific component (models, distillation, rag, optimization, agent_translation, pubmed_medical, evaluation)")
    args = parser.parse_args()
    
    if args.component:
        # Run specific component test
        if args.component == "models":
            test_teacher_student_models()
        elif args.component == "distillation":
            test_knowledge_distillation()
        elif args.component == "rag":
            test_rag_pipeline()
        elif args.component == "optimization":
            test_optimization_techniques()
        elif args.component == "agent_translation":
            test_agent_and_translation()
        elif args.component == "pubmed_medical":
            test_pubmed_and_medical_rag()
        elif args.component == "evaluation":
            test_evaluation_and_benchmarking()
        else:
            logger.error(f"Unknown component: {args.component}")
    else:
        # Run full end-to-end test
        run_end_to_end_test()
