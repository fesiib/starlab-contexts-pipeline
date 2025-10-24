# LLM-as-a-Judge Framework

A modular framework for evaluating responses using LLM judges with support for absolute metrics (ranking 1-7, binary decisions) and easy expansion for comparative metrics.

## Features

- **Modular Design**: Easy to add new evaluation metrics
- **Multiple Scale Types**: Support for binary decisions, Likert scales (1-7, 1-5)
- **Batch Evaluation**: Efficient evaluation of multiple responses
- **Result Aggregation**: Automatic calculation of statistics and confidence scores
- **Easy Integration**: Simple integration with existing evaluation pipelines
- **Extensible**: Ready for future comparative evaluation features

## Quick Start

### Basic Usage

```python
from llm_judge import TestCase, Response, EvaluationRunner, setup_default_registry

# Create test cases and responses
test_cases = [
    TestCase(query="What is machine learning?", context="ML is a subset of AI..."),
    # ... more test cases
]

responses = [
    Response(content="Machine learning is a subset of artificial intelligence..."),
    # ... more responses
]

# Set up evaluation
registry = setup_default_registry()
runner = EvaluationRunner(registry)

# Run evaluation
metrics = ["relevance", "quality", "faithfulness"]
results = runner.run_evaluation(test_cases, responses, metrics)

# Get aggregated results
aggregated = runner.aggregate_results(results)
print(aggregated)
```

### Integration with Existing Code

```python
from eval.integration_example import enhanced_run_eval_abs

# Use your existing configurations
results = enhanced_run_eval_abs(
    dataset_config, 
    method_config, 
    eval_config, 
    use_llm_judge=True
)
```

## Architecture

### Core Components

1. **BaseLLMJudge**: Abstract base class for all judges
2. **AbsoluteMetricJudge**: Base class for absolute metrics (ranking, binary)
3. **BinaryJudge**: Binary decision evaluation (pass/fail)
4. **LikertJudge**: Likert scale evaluation (1-7, 1-5)
5. **MetricRegistry**: Manages available metrics
6. **EvaluationRunner**: Orchestrates evaluation process

### Data Structures

- **TestCase**: Contains query, context, expected output, metadata
- **Response**: Contains response content and metadata
- **EvaluationResult**: Contains score, confidence, reasoning, metadata

## Available Metrics

### Predefined Metrics

- **relevance**: How relevant is the response to the query? (1-7 scale)
- **quality**: What is the overall quality of the response? (1-7 scale)
- **faithfulness**: How faithful is the response to the source context? (1-7 scale)
- **comprehensiveness**: How comprehensive is the response? (1-7 scale)
- **clarity**: How clear and understandable is the response? (1-7 scale)
- **helpfulness**: How helpful is the response for the user's query? (1-7 scale)

### Binary Metrics

- **relevance_binary**: Is the response relevant? (PASS/FAIL)
- **quality_binary**: Does the response meet quality standards? (PASS/FAIL)

## Custom Metrics

### Creating a Custom Judge

```python
from llm_judge import AbsoluteMetricJudge, ScaleType, EvaluationResult
import json

class CustomAccuracyJudge(AbsoluteMetricJudge):
    def __init__(self):
        super().__init__("custom_accuracy", ScaleType.LIKERT_7)
    
    def get_evaluation_prompt(self, test_case, response):
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
    
    def parse_evaluation_response(self, llm_response):
        try:
            data = json.loads(llm_response)
            rating = int(data.get("rating", 1))
            confidence = float(data.get("confidence", 0.0))
            reasoning = data.get("reasoning", "")
            
            return EvaluationResult(
                metric_name=self.metric_name,
                score=rating,
                confidence=confidence,
                reasoning=reasoning,
                metadata={"rating": rating, "scale_max": 7}
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return EvaluationResult(
                metric_name=self.metric_name,
                score=1,
                confidence=0.0,
                reasoning=f"Failed to parse response: {str(e)}",
                metadata={"error": True}
            )

# Register the custom metric
registry = MetricRegistry()
registry.register_metric("custom_accuracy", CustomAccuracyJudge())
```

## Configuration

### Using Metric Configurations

```python
from metric_configs import create_metric_registry_from_config

# Create registry with specific metrics
registry = create_metric_registry_from_config([
    "relevance", 
    "quality", 
    "faithfulness"
])

# Or use predefined configurations
from metric_configs import setup_ir_evaluation_metrics
registry = setup_ir_evaluation_metrics()
```

### Custom Configurations

```python
from metric_configs import create_custom_metric_config

# Create a custom metric configuration
create_custom_metric_config(
    name="custom_metric",
    metric_type="likert_7",
    description="Custom evaluation metric",
    scale_description={
        1: "Very poor",
        2: "Poor",
        3: "Below average",
        4: "Average",
        5: "Good",
        6: "Very good",
        7: "Excellent"
    }
)
```

## Integration with Existing Code

### Modifying _tech_eval.py

Replace your existing `run_eval_abs` function with the enhanced version:

```python
from eval.integration_example import enhanced_run_eval_abs

# Your existing code
results = enhanced_run_eval_abs(
    dataset_config, 
    method_config, 
    eval_config, 
    use_llm_judge=True
)
```

### Converting Existing Data

```python
from eval.integration_example import convert_tech_eval_data_to_test_cases, convert_responses_to_response_objects

# Convert your existing data format
test_cases = convert_tech_eval_data_to_test_cases(test_dataset, responses)
response_objects = convert_responses_to_response_objects(responses)
```

## Output Format

### Evaluation Results

```json
{
  "aggregated_results": {
    "relevance": {
      "mean_score": 5.2,
      "std_score": 1.1,
      "min_score": 3,
      "max_score": 7,
      "mean_confidence": 0.85,
      "error_rate": 0.05
    }
  },
  "detailed_results": {
    "relevance": [
      {
        "metric_name": "relevance",
        "score": 6,
        "confidence": 0.9,
        "reasoning": "The response directly addresses the query...",
        "metadata": {"rating": 6, "scale_max": 7}
      }
    ]
  }
}
```

## Future Extensions

The framework is designed to easily support:

1. **Comparative Metrics**: Compare multiple responses side-by-side
2. **Custom LLM Models**: Support for different LLM providers
3. **Advanced Prompting**: More sophisticated prompt engineering
4. **Multi-turn Evaluation**: Evaluate conversational responses
5. **Domain-specific Metrics**: Specialized metrics for different domains

## Examples

See the following files for complete examples:

- `example_usage.py`: Basic usage examples
- `integration_example.py`: Integration with existing code
- `metric_configs.py`: Metric configuration examples

## Requirements

- Python 3.7+
- JSON support
- Logging support
- (Future) LLM API integration (OpenAI, Anthropic, etc.)

## TODO

- [ ] Implement actual LLM API calls
- [ ] Add support for different LLM providers
- [ ] Implement comparative evaluation
- [ ] Add more sophisticated prompt engineering
- [ ] Add support for multi-turn conversations
- [ ] Add visualization tools for results
