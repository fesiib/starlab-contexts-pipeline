"""
Integration example showing how to modify your existing _tech_eval.py
to use the LLM-as-a-Judge framework
"""

import json
import os
from typing import Dict, List, Any

# Import your existing modules
from helpers.dataset import get_dataset
from src.rag import generic_call_rag
from src.cim_methods import generic_call_context_similarity, generic_call_shortest_path

# Import the LLM judge framework
from llm_judge import (
    TestCase, Response, EvaluationRunner, MetricRegistry,
    convert_tech_eval_data_to_test_cases, convert_responses_to_response_objects
)
from metric_configs import create_tech_eval_registry


def enhanced_run_eval_abs(dataset_config, method_config, eval_config, use_llm_judge=True):
    """
    Enhanced version of your run_eval_abs function with LLM-as-a-Judge integration
    
    Args:
        dataset_config: Your existing dataset configuration
        method_config: Your existing method configuration  
        eval_config: Your existing evaluation configuration
        use_llm_judge: Whether to use LLM evaluation (default: True)
    """
    dataset_label = dataset_config["label"]
    eval_label = eval_config["label"]
    method_label = method_config["label"]

    results_path = os.path.join("./static/results/tech_eval/", 
                               f"{dataset_label}_{method_label}_{eval_label}.json")

    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            return json.load(f)

    print("Constructing test dataset...")
    test_dataset = construct_test_dataset(dataset_config)
    test_dataset_statistics(test_dataset)

    print("Running method...")
    responses = run_method(method_config, test_dataset)
    
    # Your existing evaluation
    print("Running eval...")
    eval_func = eval_config["func"]
    results = eval_func(responses, test_dataset)
    
    # Add LLM-as-a-Judge evaluation if requested
    llm_results = {}
    if use_llm_judge:
        print("Running LLM-as-a-Judge evaluation...")
        try:
            # Convert data to framework format
            test_cases = convert_tech_eval_data_to_test_cases(test_dataset, responses)
            response_objects = convert_responses_to_response_objects(responses)
            
            # Set up LLM evaluation
            registry = create_tech_eval_registry()
            runner = EvaluationRunner(registry)
            
            # Run LLM evaluation
            llm_metrics = ["relevance", "quality", "faithfulness", "comprehensiveness"]
            llm_eval_results = runner.run_evaluation(test_cases, response_objects, llm_metrics)
            llm_aggregated = runner.aggregate_results(llm_eval_results)
            
            llm_results = {
                "llm_evaluation": {
                    "results": llm_eval_results,
                    "aggregated": llm_aggregated
                }
            }
            
            # Save LLM results separately
            llm_results_path = os.path.join("./static/results/tech_eval/", 
                                          f"{dataset_label}_{method_label}_{eval_label}_llm.json")
            runner.save_results(llm_eval_results, llm_aggregated, llm_results_path)
            
        except Exception as e:
            print(f"LLM evaluation failed: {e}")
            llm_results = {"llm_evaluation": {"error": str(e)}}
    
    # Combine results
    final_results = {
        "responses": responses,
        "dataset_config": dataset_config,
        "method_config": method_config,
        "eval_config": eval_config,
        "original_results": results,
        **llm_results
    }
    
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=4)

    return final_results


def run_llm_evaluation_only(dataset_config, method_config, test_dataset, responses):
    """
    Run only LLM evaluation on existing data
    
    Args:
        dataset_config: Dataset configuration
        method_config: Method configuration
        test_dataset: Your existing test dataset
        responses: Your existing responses
    """
    print("Running LLM-as-a-Judge evaluation...")
    
    # Convert data to framework format
    test_cases = convert_tech_eval_data_to_test_cases(test_dataset, responses)
    response_objects = convert_responses_to_response_objects(responses)
    
    # Set up LLM evaluation
    registry = create_tech_eval_registry()
    runner = EvaluationRunner(registry)
    
    # Run LLM evaluation
    llm_metrics = ["relevance", "quality", "faithfulness", "comprehensiveness"]
    llm_eval_results = runner.run_evaluation(test_cases, response_objects, llm_metrics)
    llm_aggregated = runner.aggregate_results(llm_eval_results)
    
    # Save results
    dataset_label = dataset_config["label"]
    method_label = method_config["label"]
    llm_results_path = os.path.join("./static/results/tech_eval/", 
                                   f"{dataset_label}_{method_label}_llm_evaluation.json")
    runner.save_results(llm_eval_results, llm_aggregated, llm_results_path)
    
    return llm_eval_results, llm_aggregated


def compare_methods_with_llm_judge(dataset_config, method_configs, test_dataset):
    """
    Compare multiple methods using LLM-as-a-Judge
    
    Args:
        dataset_config: Dataset configuration
        method_configs: List of method configurations to compare
        test_dataset: Test dataset
    """
    print("Comparing methods with LLM-as-a-Judge...")
    
    all_results = {}
    
    for method_name, method_config in method_configs.items():
        print(f"Evaluating method: {method_name}")
        
        # Run method
        responses = run_method(method_config, test_dataset)
        
        # Run LLM evaluation
        try:
            llm_results, llm_aggregated = run_llm_evaluation_only(
                dataset_config, method_config, test_dataset, responses
            )
            
            all_results[method_name] = {
                "method_config": method_config,
                "llm_results": llm_results,
                "llm_aggregated": llm_aggregated
            }
            
        except Exception as e:
            print(f"Error evaluating {method_name}: {e}")
            all_results[method_name] = {"error": str(e)}
    
    # Save comparison results
    comparison_path = os.path.join("./static/results/tech_eval/", 
                                 f"{dataset_config['label']}_method_comparison_llm.json")
    with open(comparison_path, "w") as f:
        json.dump(all_results, f, indent=4)
    
    return all_results


# Helper functions (you'll need to implement these based on your existing code)
def construct_test_dataset(dataset_config):
    """Your existing construct_test_dataset function"""
    # Implementation from your _tech_eval.py
    pass

def test_dataset_statistics(test_dataset):
    """Your existing test_dataset_statistics function"""
    # Implementation from your _tech_eval.py
    pass

def run_method(method_config, test_dataset):
    """Your existing run_method function"""
    # Implementation from your _tech_eval.py
    pass


# Example usage
def main():
    """Example of how to use the enhanced evaluation"""
    
    # Your existing configurations
    DATASETS = [
        {
            "label": "test_q0_n5",
            "tasks": ["muffin_task"],  # Your existing tasks
            "query_idx": 0,
            "N_idx": 0,
        }
    ]
    
    METHODS = {
        "rag": {
            "label": "RAG-bert-10-0.7",
            "embedding_method": "bert",
            "k": 10,
            "doc_score_threshold": 0.7,
            "func": generic_call_rag,
        },
        "vanilla": {
            "label": "vanilla-bert",
            "embedding_method": "bert",
            "k": None,
            "doc_score_threshold": None,
            "func": generic_call_rag,
        }
    }
    
    EVALS = {
        "relevance_abs": {
            "label": "relevance-analysis-abs",
            "func": None,  # Your existing evaluation function
        }
    }
    
    # Run enhanced evaluation
    dataset_config = DATASETS[0]
    method_config = METHODS["rag"]
    eval_config = EVALS["relevance_abs"]
    
    # Run with LLM evaluation
    results = enhanced_run_eval_abs(dataset_config, method_config, eval_config, use_llm_judge=True)
    
    print("Evaluation completed!")
    print(f"LLM evaluation results: {results.get('llm_evaluation', 'Not available')}")


if __name__ == "__main__":
    main()
