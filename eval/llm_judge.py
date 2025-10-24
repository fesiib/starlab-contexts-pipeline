"""
LLM-as-a-Judge Framework

A modular framework for evaluating responses using LLM judges with support for:
- Absolute metrics (ranking 1-7, binary decisions)
- Comparative metrics (future expansion)
- Easy metric registration and extension
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of evaluation metrics"""
    ABSOLUTE = "absolute"
    COMPARATIVE = "comparative"


class ScaleType(Enum):
    """Types of rating scales"""
    BINARY = "binary"
    LIKERT_7 = "likert_7"
    LIKERT_5 = "likert_5"
    CUSTOM = "custom"


@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    metric_name: str
    score: Union[float, int, str]
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TestCase:
    """Container for a single test case"""
    query: str
    context: Optional[str] = None
    expected_output: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Response:
    """Container for a model response"""
    content: str
    metadata: Optional[Dict[str, Any]] = None


class BaseLLMJudge(ABC):
    """Base class for LLM judges"""
    
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def evaluate(self, test_case: TestCase, response: Response) -> EvaluationResult:
        """Evaluate a single response against a test case"""
        pass
    
    def batch_evaluate(self, test_cases: List[TestCase], responses: List[Response]) -> List[EvaluationResult]:
        """Evaluate multiple responses in batch"""
        if len(test_cases) != len(responses):
            raise ValueError("Number of test cases must match number of responses")
        
        results = []
        for test_case, response in zip(test_cases, responses):
            try:
                result = self.evaluate(test_case, response)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error evaluating test case: {e}")
                # Create error result
                error_result = EvaluationResult(
                    metric_name=self.__class__.__name__,
                    score=0.0,
                    confidence=0.0,
                    reasoning=f"Evaluation failed: {str(e)}",
                    metadata={"error": True}
                )
                results.append(error_result)
        
        return results


class AbsoluteMetricJudge(BaseLLMJudge):
    """Base class for absolute metric evaluation"""
    
    def __init__(self, metric_name: str, scale_type: ScaleType, 
                 model_name: str = "gpt-4", temperature: float = 0.0):
        super().__init__(model_name, temperature)
        self.metric_name = metric_name
        self.scale_type = scale_type
        self.metric_type = MetricType.ABSOLUTE
    
    @abstractmethod
    def get_evaluation_prompt(self, test_case: TestCase, response: Response) -> str:
        """Generate the evaluation prompt"""
        pass
    
    @abstractmethod
    def parse_evaluation_response(self, llm_response: str) -> EvaluationResult:
        """Parse the LLM's evaluation response"""
        pass
    
    def evaluate(self, test_case: TestCase, response: Response) -> EvaluationResult:
        """Evaluate using LLM judge"""
        prompt = self.get_evaluation_prompt(test_case, response)
        
        # TODO: Replace with actual LLM call
        # For now, return a placeholder
        llm_response = self._call_llm(prompt)
        
        return self.parse_evaluation_response(llm_response)
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt"""
        # TODO: Implement actual LLM call
        # This is a placeholder that should be replaced with actual LLM API call
        return f"LLM response for: {prompt[:100]}..."


class BinaryJudge(AbsoluteMetricJudge):
    """Binary decision judge (pass/fail, relevant/irrelevant, etc.)"""
    
    def __init__(self, metric_name: str, model_name: str = "gpt-4", temperature: float = 0.0):
        super().__init__(metric_name, ScaleType.BINARY, model_name, temperature)
    
    def get_evaluation_prompt(self, test_case: TestCase, response: Response) -> str:
        """Generate binary evaluation prompt"""
        prompt = f"""
You are an expert evaluator. Your task is to make a binary decision about the quality of a response.

Query: {test_case.query}
Context: {test_case.context or 'No context provided'}
Response: {response.content}

Please evaluate the response and provide:
1. Decision: PASS or FAIL
2. Confidence: A number between 0 and 1
3. Reasoning: Brief explanation of your decision

Format your response as JSON:
{{
    "decision": "PASS" or "FAIL",
    "confidence": 0.0-1.0,
    "reasoning": "Your explanation"
}}
"""
        return prompt
    
    def parse_evaluation_response(self, llm_response: str) -> EvaluationResult:
        """Parse binary evaluation response"""
        try:
            # Try to parse JSON response
            data = json.loads(llm_response)
            decision = data.get("decision", "FAIL")
            confidence = float(data.get("confidence", 0.0))
            reasoning = data.get("reasoning", "")
            
            # Convert to numeric score
            score = 1.0 if decision == "PASS" else 0.0
            
            return EvaluationResult(
                metric_name=self.metric_name,
                score=score,
                confidence=confidence,
                reasoning=reasoning,
                metadata={"decision": decision}
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.error(f"Failed to parse binary evaluation: {e}")
            return EvaluationResult(
                metric_name=self.metric_name,
                score=0.0,
                confidence=0.0,
                reasoning=f"Failed to parse response: {str(e)}",
                metadata={"error": True}
            )


class LikertJudge(AbsoluteMetricJudge):
    """Likert scale judge (1-7, 1-5, etc.)"""
    
    def __init__(self, metric_name: str, scale_type: ScaleType = ScaleType.LIKERT_7,
                 model_name: str = "gpt-4", temperature: float = 0.0):
        super().__init__(metric_name, scale_type, model_name, temperature)
        self.scale_max = 7 if scale_type == ScaleType.LIKERT_7 else 5
    
    def get_evaluation_prompt(self, test_case: TestCase, response: Response) -> str:
        """Generate Likert scale evaluation prompt"""
        scale_description = {
            1: "Poor",
            2: "Below Average", 
            3: "Average",
            4: "Good",
            5: "Very Good",
            6: "Excellent",
            7: "Outstanding"
        } if self.scale_max == 7 else {
            1: "Poor",
            2: "Below Average",
            3: "Average", 
            4: "Good",
            5: "Excellent"
        }
        
        scale_desc = "\n".join([f"{k}: {v}" for k, v in scale_description.items()])
        
        prompt = f"""
You are an expert evaluator. Your task is to rate a response on a {self.scale_max}-point scale.

Query: {test_case.query}
Context: {test_case.context or 'No context provided'}
Response: {response.content}

Rating Scale:
{scale_desc}

Please evaluate the response and provide:
1. Rating: A number from 1 to {self.scale_max}
2. Confidence: A number between 0 and 1
3. Reasoning: Brief explanation of your rating

Format your response as JSON:
{{
    "rating": 1-{self.scale_max},
    "confidence": 0.0-1.0,
    "reasoning": "Your explanation"
}}
"""
        return prompt
    
    def parse_evaluation_response(self, llm_response: str) -> EvaluationResult:
        """Parse Likert scale evaluation response"""
        try:
            data = json.loads(llm_response)
            rating = int(data.get("rating", 1))
            confidence = float(data.get("confidence", 0.0))
            reasoning = data.get("reasoning", "")
            
            # Ensure rating is within valid range
            rating = max(1, min(rating, self.scale_max))
            
            return EvaluationResult(
                metric_name=self.metric_name,
                score=rating,
                confidence=confidence,
                reasoning=reasoning,
                metadata={"rating": rating, "scale_max": self.scale_max}
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.error(f"Failed to parse Likert evaluation: {e}")
            return EvaluationResult(
                metric_name=self.metric_name,
                score=1,
                confidence=0.0,
                reasoning=f"Failed to parse response: {str(e)}",
                metadata={"error": True}
            )


class MetricRegistry:
    """Registry for managing evaluation metrics"""
    
    def __init__(self):
        self.metrics: Dict[str, BaseLLMJudge] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def register_metric(self, name: str, judge: BaseLLMJudge) -> None:
        """Register a new metric"""
        self.metrics[name] = judge
        self.logger.info(f"Registered metric: {name}")
    
    def get_metric(self, name: str) -> BaseLLMJudge:
        """Get a registered metric"""
        if name not in self.metrics:
            raise ValueError(f"Metric '{name}' not found in registry")
        return self.metrics[name]
    
    def list_metrics(self) -> List[str]:
        """List all registered metrics"""
        return list(self.metrics.keys())
    
    def remove_metric(self, name: str) -> None:
        """Remove a metric from registry"""
        if name in self.metrics:
            del self.metrics[name]
            self.logger.info(f"Removed metric: {name}")


class EvaluationRunner:
    """Orchestrates the evaluation process"""
    
    def __init__(self, registry: Optional[MetricRegistry] = None):
        self.registry = registry or MetricRegistry()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def run_evaluation(self, test_cases: List[TestCase], responses: List[Response], 
                      metric_names: List[str]) -> Dict[str, List[EvaluationResult]]:
        """Run evaluation with specified metrics"""
        if len(test_cases) != len(responses):
            raise ValueError("Number of test cases must match number of responses")
        
        results = {}
        
        for metric_name in metric_names:
            try:
                judge = self.registry.get_metric(metric_name)
                self.logger.info(f"Running evaluation with metric: {metric_name}")
                
                metric_results = judge.batch_evaluate(test_cases, responses)
                results[metric_name] = metric_results
                
                self.logger.info(f"Completed evaluation for {metric_name}: {len(metric_results)} results")
                
            except Exception as e:
                self.logger.error(f"Error running metric {metric_name}: {e}")
                results[metric_name] = []
        
        return results
    
    def aggregate_results(self, results: Dict[str, List[EvaluationResult]]) -> Dict[str, Any]:
        """Aggregate evaluation results"""
        aggregated = {}
        
        for metric_name, metric_results in results.items():
            if not metric_results:
                aggregated[metric_name] = {"error": "No results available"}
                continue
            
            # Extract scores
            scores = [r.score for r in metric_results if isinstance(r.score, (int, float))]
            confidences = [r.confidence for r in metric_results if r.confidence is not None]
            
            if scores:
                aggregated[metric_name] = {
                    "mean_score": sum(scores) / len(scores),
                    "min_score": min(scores),
                    "max_score": max(scores),
                    "std_score": self._calculate_std(scores),
                    "count": len(scores),
                    "mean_confidence": sum(confidences) / len(confidences) if confidences else None,
                    "error_rate": sum(1 for r in metric_results if r.metadata and r.metadata.get("error")) / len(metric_results)
                }
            else:
                aggregated[metric_name] = {"error": "No valid scores found"}
        
        return aggregated
    
    def _calculate_std(self, scores: List[float]) -> float:
        """Calculate standard deviation"""
        if len(scores) < 2:
            return 0.0
        
        mean = sum(scores) / len(scores)
        variance = sum((x - mean) ** 2 for x in scores) / len(scores)
        return variance ** 0.5
    
    def save_results(self, results: Dict[str, List[EvaluationResult]], 
                    aggregated: Dict[str, Any], output_path: str) -> None:
        """Save evaluation results to file"""
        output_data = {
            "aggregated_results": aggregated,
            "detailed_results": {
                metric_name: [
                    {
                        "metric_name": r.metric_name,
                        "score": r.score,
                        "confidence": r.confidence,
                        "reasoning": r.reasoning,
                        "metadata": r.metadata
                    }
                    for r in metric_results
                ]
                for metric_name, metric_results in results.items()
            }
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        self.logger.info(f"Results saved to: {output_path}")


# Convenience functions for common use cases
def create_relevance_judge(scale_type: ScaleType = ScaleType.LIKERT_7) -> AbsoluteMetricJudge:
    """Create a relevance evaluation judge"""
    if scale_type == ScaleType.BINARY:
        return BinaryJudge("relevance", model_name="gpt-4")
    else:
        return LikertJudge("relevance", scale_type, model_name="gpt-4")


def create_quality_judge(scale_type: ScaleType = ScaleType.LIKERT_7) -> AbsoluteMetricJudge:
    """Create a quality evaluation judge"""
    if scale_type == ScaleType.BINARY:
        return BinaryJudge("quality", model_name="gpt-4")
    else:
        return LikertJudge("quality", scale_type, model_name="gpt-4")


def create_faithfulness_judge(scale_type: ScaleType = ScaleType.LIKERT_7) -> AbsoluteMetricJudge:
    """Create a faithfulness evaluation judge"""
    if scale_type == ScaleType.BINARY:
        return BinaryJudge("faithfulness", model_name="gpt-4")
    else:
        return LikertJudge("faithfulness", scale_type, model_name="gpt-4")


def create_comprehensiveness_judge(scale_type: ScaleType = ScaleType.LIKERT_7) -> AbsoluteMetricJudge:
    """Create a comprehensiveness evaluation judge"""
    if scale_type == ScaleType.BINARY:
        return BinaryJudge("comprehensiveness", model_name="gpt-4")
    else:
        return LikertJudge("comprehensiveness", scale_type, model_name="gpt-4")


# Example usage and integration
def setup_default_registry() -> MetricRegistry:
    """Set up a registry with common metrics"""
    registry = MetricRegistry()
    
    # Register common metrics
    registry.register_metric("relevance", create_relevance_judge())
    registry.register_metric("quality", create_quality_judge())
    registry.register_metric("faithfulness", create_faithfulness_judge())
    registry.register_metric("comprehensiveness", create_comprehensiveness_judge())
    
    # Register binary versions
    registry.register_metric("relevance_binary", create_relevance_judge(ScaleType.BINARY))
    registry.register_metric("quality_binary", create_quality_judge(ScaleType.BINARY))
    
    return registry
