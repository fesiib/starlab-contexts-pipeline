"""
Configuration file for LLM-as-a-Judge metrics

This file contains predefined metric configurations that can be easily customized
and extended for different evaluation scenarios.
"""

from typing import Dict, Any, List
from llm_judge import (
    MetricRegistry, BinaryJudge, LikertJudge, 
    ScaleType, create_relevance_judge, create_quality_judge,
    create_faithfulness_judge, create_comprehensiveness_judge
)


# Predefined metric configurations
METRIC_CONFIGS = {
    "relevance": {
        "type": "likert_7",
        "description": "How relevant is the response to the query?",
        "scale_description": {
            1: "Not relevant at all",
            2: "Slightly relevant", 
            3: "Somewhat relevant",
            4: "Moderately relevant",
            5: "Quite relevant",
            6: "Very relevant",
            7: "Extremely relevant"
        }
    },
    "quality": {
        "type": "likert_7",
        "description": "What is the overall quality of the response?",
        "scale_description": {
            1: "Very poor quality",
            2: "Poor quality",
            3: "Below average quality",
            4: "Average quality",
            5: "Good quality",
            6: "Very good quality",
            7: "Excellent quality"
        }
    },
    "faithfulness": {
        "type": "likert_7",
        "description": "How faithful is the response to the source context?",
        "scale_description": {
            1: "Completely unfaithful",
            2: "Mostly unfaithful",
            3: "Somewhat unfaithful",
            4: "Neutral",
            5: "Somewhat faithful",
            6: "Mostly faithful",
            7: "Completely faithful"
        }
    },
    "comprehensiveness": {
        "type": "likert_7",
        "description": "How comprehensive is the response?",
        "scale_description": {
            1: "Very incomplete",
            2: "Incomplete",
            3: "Somewhat incomplete",
            4: "Moderately complete",
            5: "Quite complete",
            6: "Very complete",
            7: "Extremely complete"
        }
    },
    "clarity": {
        "type": "likert_7",
        "description": "How clear and understandable is the response?",
        "scale_description": {
            1: "Very unclear",
            2: "Unclear",
            3: "Somewhat unclear",
            4: "Moderately clear",
            5: "Quite clear",
            6: "Very clear",
            7: "Extremely clear"
        }
    },
    "helpfulness": {
        "type": "likert_7",
        "description": "How helpful is the response for the user's query?",
        "scale_description": {
            1: "Not helpful at all",
            2: "Slightly helpful",
            3: "Somewhat helpful",
            4: "Moderately helpful",
            5: "Quite helpful",
            6: "Very helpful",
            7: "Extremely helpful"
        }
    },
    # Binary metrics
    "relevance_binary": {
        "type": "binary",
        "description": "Is the response relevant to the query?",
        "options": {
            "PASS": "Response is relevant",
            "FAIL": "Response is not relevant"
        }
    },
    "quality_binary": {
        "type": "binary", 
        "description": "Is the response of acceptable quality?",
        "options": {
            "PASS": "Response meets quality standards",
            "FAIL": "Response does not meet quality standards"
        }
    }
}


def create_metric_registry_from_config(config_names: List[str] = None) -> MetricRegistry:
    """Create a metric registry from configuration names"""
    if config_names is None:
        config_names = ["relevance", "quality", "faithfulness", "comprehensiveness"]
    
    registry = MetricRegistry()
    
    for config_name in config_names:
        if config_name not in METRIC_CONFIGS:
            raise ValueError(f"Unknown metric configuration: {config_name}")
        
        config = METRIC_CONFIGS[config_name]
        
        if config["type"] == "likert_7":
            judge = LikertJudge(
                metric_name=config_name,
                scale_type=ScaleType.LIKERT_7,
                model_name="gpt-4"
            )
        elif config["type"] == "likert_5":
            judge = LikertJudge(
                metric_name=config_name,
                scale_type=ScaleType.LIKERT_5,
                model_name="gpt-4"
            )
        elif config["type"] == "binary":
            judge = BinaryJudge(
                metric_name=config_name,
                model_name="gpt-4"
            )
        else:
            raise ValueError(f"Unknown metric type: {config['type']}")
        
        registry.register_metric(config_name, judge)
    
    return registry


def get_metric_description(metric_name: str) -> Dict[str, Any]:
    """Get description and configuration for a metric"""
    if metric_name not in METRIC_CONFIGS:
        raise ValueError(f"Unknown metric: {metric_name}")
    
    return METRIC_CONFIGS[metric_name]


def list_available_metrics() -> List[str]:
    """List all available metric configurations"""
    return list(METRIC_CONFIGS.keys())


def create_custom_metric_config(name: str, metric_type: str, description: str, 
                              scale_description: Dict[int, str] = None) -> Dict[str, Any]:
    """Create a custom metric configuration"""
    config = {
        "type": metric_type,
        "description": description
    }
    
    if metric_type in ["likert_7", "likert_5"] and scale_description:
        config["scale_description"] = scale_description
    elif metric_type == "binary" and scale_description:
        config["options"] = scale_description
    
    METRIC_CONFIGS[name] = config
    return config


# Example usage functions
def setup_ir_evaluation_metrics() -> MetricRegistry:
    """Set up metrics specifically for Information Retrieval evaluation"""
    ir_metrics = [
        "relevance",
        "quality", 
        "faithfulness",
        "comprehensiveness",
        "helpfulness"
    ]
    return create_metric_registry_from_config(ir_metrics)


def setup_binary_evaluation_metrics() -> MetricRegistry:
    """Set up binary evaluation metrics"""
    binary_metrics = [
        "relevance_binary",
        "quality_binary"
    ]
    return create_metric_registry_from_config(binary_metrics)


def setup_comprehensive_evaluation_metrics() -> MetricRegistry:
    """Set up comprehensive evaluation with all available metrics"""
    all_metrics = list(METRIC_CONFIGS.keys())
    return create_metric_registry_from_config(all_metrics)


# Integration with your existing evaluation framework
def get_tech_eval_metrics() -> List[str]:
    """Get metrics that align with your existing EVALS configuration"""
    return [
        "relevance",  # Maps to "relevance_abs"
        "faithfulness",  # Maps to "faithfulness"
        "comprehensiveness",  # Maps to "comprehensiveness"
        "quality"  # Additional quality metric
    ]


def create_tech_eval_registry() -> MetricRegistry:
    """Create registry with metrics that match your existing evaluation setup"""
    return create_metric_registry_from_config(get_tech_eval_metrics())
