"""
Evaluation prompts for LLM-as-a-Judge framework
TODO: evaluate one retrieved information at a time instead of full response
"""
from pydantic_models.evaluation import MetricScale
from pydantic_models.evaluation import Likert7EvaluationResponse, Likert5EvaluationResponse, BinaryEvaluationResponse, ComparisonEvaluationResponse, Likert3EvaluationResponse
from prompts import tutorial_to_str, response_to_str

def get_response_for_metric_request(messages, metric, judge_model):
    if metric == MetricScale.LIKERT_7:
        response_format = Likert7EvaluationResponse
    elif metric == MetricScale.LIKERT_5:
        response_format = Likert5EvaluationResponse
    elif metric == MetricScale.LIKERT_3:
        response_format = Likert3EvaluationResponse
    elif metric == MetricScale.BINARY:
        response_format = BinaryEvaluationResponse
    elif metric == MetricScale.COMPARISON:
        response_format = ComparisonEvaluationResponse
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return {
        "messages": messages,
        "model": judge_model,
        "response_format": response_format,
    }

def get_response_for_metric_response(response, **kwargs):
    return {
        **response,
        "confidence": response["confidence"] / 10,
        "reasoning": response["reasoning"]
    }

SYSTEM_PROMPT_EVAL = """
You are a helpful assistant that can comprehensively evaluate the responses to a query about a procedural task `{task}`.
"""

USER_PROMPT_EVAL_FULL_TUTORIAL = """
You are given a query and a response to the query. The query was originally asked in the context of the current tutorial.
Evaluate the response based on the following criteria:
{criteria}

Current tutorial:
{cur_tutorial}

Original query:
{query}

Response:
{response}
"""

def eval_relevance_absolute_full_tutorial_request(task, tutorial, query, eval_response, metric, criteria, judge_model):
    if metric == MetricScale.COMPARISON:
        raise ValueError("Comparison metrics are not supported for absolute evaluation.")
    cur_tutorial_str = tutorial_to_str(tutorial)
    response_str = response_to_str(eval_response)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_EVAL.format(task=task),
        },
        {
            "role": "user",
            "content": USER_PROMPT_EVAL_FULL_TUTORIAL.format( cur_tutorial=cur_tutorial_str, query=query, response=response_str, criteria=criteria),
        },
    ]

    return get_response_for_metric_request(messages, metric, judge_model)

USER_PROMPT_EVAL_TUTORIAL_SEGMENT = """
You are given a query and a response to the query. The query was originally asked in the context of the current tutorial and its segment.
Evaluate the response based on the following criteria:
{criteria}

Current tutorial:
{cur_tutorial}

Current segment:
{cur_segment}

Original query:
{query}

Response:
{response}
"""

def eval_relevance_absolute_tutorial_segment_request(task, tutorial, segment, query, eval_response, metric, criteria, judge_model):
    if metric == MetricScale.COMPARISON:
        raise ValueError("Comparison metrics are not supported for absolute evaluation.")
    cur_tutorial_str = tutorial_to_str(tutorial)
    cur_segment_str = segment["content"]
    response_str = response_to_str(eval_response)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_EVAL.format(task=task),
        },
        {
            "role": "user",
            "content": USER_PROMPT_EVAL_TUTORIAL_SEGMENT.format(cur_tutorial=cur_tutorial_str, cur_segment=cur_segment_str, query=query, response=response_str, criteria=criteria),
        },
    ]

    return get_response_for_metric_request(messages, metric, judge_model)

def eval_relevance_absolute_request(task, tutorial, segment, query, eval_response, metric, criteria, judge_model):
    if segment is None:
        return eval_relevance_absolute_full_tutorial_request(task, tutorial, query, eval_response, metric, criteria, judge_model)
    else:
        return eval_relevance_absolute_tutorial_segment_request(task, tutorial, segment, query, eval_response, metric, criteria, judge_model)

def eval_relevance_absolute_response(response, **kwargs):
    return get_response_for_metric_response(response, **kwargs)

def eval_relevance_comparison_full_tutorial_request(metric, criteria, judge_model):
    if metric != MetricScale.COMPARISON:
        raise ValueError("Metric must be comparison for comparison evaluation.")
    ### random shuffle the order of the responses

def eval_relevance_comparison_tutorial_segment_request(metric, criteria, judge_model):
    if metric != MetricScale.COMPARISON:
        raise ValueError("Metric must be comparison for comparison evaluation.")
    pass

def eval_comprehensiveness_comparison_full_tutorial_request(metric, criteria, judge_model):
    pass

def eval_comprehensiveness_comparison_tutorial_segment_request(metric, criteria, judge_model):
    pass