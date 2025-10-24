"""
Example usage of the LLM-as-a-Judge framework

This file demonstrates how to use the framework with your existing evaluation pipeline.
"""

import json
import os
from typing import List, Dict, Any

from llm_judge import (
    TestCase, Response, EvaluationRunner, MetricRegistry,
    create_relevance_judge, create_quality_judge, create_faithfulness_judge,
    setup_default_registry, ScaleType
)


def convert_tech_eval_data_to_test_cases(test_dataset: List[Dict[str, Any]], 
                                       responses: List[Dict[str, Any]]) -> List[TestCase]:
    """Convert your existing test dataset format to TestCase objects"""
    test_cases = []
    
    for test_item, response_item in zip(test_dataset, responses):
        test_case = TestCase(
            query=test_item["query"],
            context=test_item.get("tutorial", {}).get("content", ""),
            expected_output=None,  # You can add this if available
            metadata={
                "task": test_item.get("task"),
                "segment": test_item.get("segment"),
                "n": test_item.get("n")
            }
        )
        test_cases.append(test_case)
    
    return test_cases


def convert_responses_to_response_objects(responses: List[Dict[str, Any]]) -> List[Response]:
    """Convert your existing response format to Response objects"""
    response_objects = []
    
    for response_item in responses:
        response = Response(
            content=response_item.get("content", ""),
            metadata={
                "method": response_item.get("method"),
                "retrieved_docs": response_item.get("retrieved_docs", []),
                "scores": response_item.get("scores", [])
            }
        )
        response_objects.append(response)
    
    return response_objects


def run_llm_evaluation_example():
    """Example of running LLM evaluation on your existing data"""
    
    # Load your existing test data (replace with actual paths)
    test_dataset_path = "./static/results/tech_eval/test_q0_n5.json"
    responses_path = "./static/results/tech_eval/test_q0_n5_rag_relevance-analysis-abs.json"
    
    # Load data
    with open(test_dataset_path, 'r') as f:
        test_dataset = json.load(f)
    
    with open(responses_path, 'r') as f:
        responses_data = json.load(f)
        responses = responses_data.get("responses", [])
    
    # Convert to framework format
    test_cases = convert_tech_eval_data_to_test_cases(test_dataset, responses)
    response_objects = convert_responses_to_response_objects(responses)
    
    # Set up evaluation
    registry = setup_default_registry()
    runner = EvaluationRunner(registry)
    
    # Define metrics to run
    metrics_to_run = ["relevance", "quality", "faithfulness", "comprehensiveness"]
    
    # Run evaluation
    print("Running LLM evaluation...")
    results = runner.run_evaluation(test_cases, response_objects, metrics_to_run)
    
    # Aggregate results
    aggregated = runner.aggregate_results(results)
    
    # Print results
    print("\n=== Evaluation Results ===")
    for metric_name, stats in aggregated.items():
        print(f"\n{metric_name.upper()}:")
        if "error" in stats:
            print(f"  Error: {stats['error']}")
        else:
            print(f"  Mean Score: {stats['mean_score']:.2f}")
            print(f"  Std Score: {stats['std_score']:.2f}")
            print(f"  Min Score: {stats['min_score']}")
            print(f"  Max Score: {stats['max_score']}")
            print(f"  Mean Confidence: {stats['mean_confidence']:.2f}")
            print(f"  Error Rate: {stats['error_rate']:.2%}")
    
    # Save results
    output_path = "./static/results/llm_evaluation_results.json"
    runner.save_results(results, aggregated, output_path)
    print(f"\nResults saved to: {output_path}")


def create_custom_metric_example():
    """Example of creating a custom metric"""
    
    from llm_judge import AbsoluteMetricJudge, ScaleType
    
    class CustomAccuracyJudge(AbsoluteMetricJudge):
        """Custom judge for accuracy evaluation"""
        
        def __init__(self):
            super().__init__("custom_accuracy", ScaleType.LIKERT_7)
        
        def get_evaluation_prompt(self, test_case: TestCase, response: Response) -> str:
            return f"""
Evaluate the accuracy of this response for the given query.

Query: {test_case.query}
Context: {test_case.context or 'No context provided'}
Response: {response.content}

Rate the accuracy on a scale of 1-7:
1: Completely inaccurate
2: Mostly inaccurate
3: Somewhat inaccurate
4: Neutral
5: Somewhat accurate
6: Mostly accurate
7: Completely accurate

Provide your evaluation as JSON:
{{
    "rating": 1-7,
    "confidence": 0.0-1.0,
    "reasoning": "Your explanation"
}}
"""
        
        def parse_evaluation_response(self, llm_response: str) -> EvaluationResult:
            # Implementation similar to LikertJudge
            pass
    
    # Register the custom metric
    registry = MetricRegistry()
    registry.register_metric("custom_accuracy", CustomAccuracyJudge())
    
    return registry


def run_comparative_evaluation_example():
    """Example of how to extend for comparative evaluation (future)"""
    
    # This is a placeholder for future comparative evaluation
    # You would create a ComparativeMetricJudge class that inherits from BaseLLMJudge
    # and implements comparison logic between multiple responses
    
    pass


def integrate_with_existing_tech_eval():
    """Integration example with your existing _tech_eval.py"""
    
    # This shows how to modify your existing run_eval_abs function
    def enhanced_run_eval_abs(dataset_config, method_config, eval_config):
        """Enhanced version of run_eval_abs with LLM evaluation"""
        
        # Your existing code...
        test_dataset = construct_test_dataset(dataset_config)
        responses = run_method(method_config, test_dataset)
        
        # Add LLM evaluation
        if eval_config.get("use_llm_judge", False):
            # Convert data
            test_cases = convert_tech_eval_data_to_test_cases(test_dataset, responses)
            response_objects = convert_responses_to_response_objects(responses)
            
            # Set up LLM evaluation
            registry = setup_default_registry()
            runner = EvaluationRunner(registry)
            
            # Run LLM evaluation
            llm_metrics = eval_config.get("llm_metrics", ["relevance", "quality"])
            llm_results = runner.run_evaluation(test_cases, response_objects, llm_metrics)
            llm_aggregated = runner.aggregate_results(llm_results)
            
            # Add LLM results to your existing results
            results = {
                "responses": responses,
                "dataset_config": dataset_config,
                "method_config": method_config,
                "eval_config": eval_config,
                "llm_evaluation": {
                    "results": llm_results,
                    "aggregated": llm_aggregated
                }
            }
        else:
            # Your existing results structure
            results = {
                "responses": responses,
                "dataset_config": dataset_config,
                "method_config": method_config,
                "eval_config": eval_config,
            }
        
        return results


if __name__ == "__main__":
    # Run the example
    run_llm_evaluation_example()
