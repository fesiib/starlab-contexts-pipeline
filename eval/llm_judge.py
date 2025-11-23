"""
Simplified LLM-as-a-Judge Framework

A streamlined framework for evaluating responses using LLM judges with support for:
- Binary decisions (pass/fail, relevant/irrelevant)
- Likert scale ratings (1-7, 1-5)
- Head-to-head comparisons (which is better)

References:
@inproceedings{10.1145/3626772.3657707,
author = {Thomas, Paul and Spielman, Seth and Craswell, Nick and Mitra, Bhaskar},
title = {Large Language Models can Accurately Predict Searcher Preferences},
year = {2024},
}
@article{Sebastian2025ValidatingLR,
  title={Validating LLM-Generated Relevance Labels for Educational Resource Search},
  author={Ratan J. Sebastian and Anett Hoppe},
  year={2025},
}
"""

import json
import os

from prompts.evaluation import (
    eval_relevance_absolute_request,
    eval_relevance_absolute_response,
)

from pydantic_models.evaluation import MetricScale

from prompts.framework_batch import batch_run_lm_calls

RELEVANCE_CRITERIA_LIKERT_3 = """
Imagine that you are learning about the task based on the given current tutorial and received an additinal list of information. Evaluate the additional information based on the following criteria:
- 3: Highly relevant and helpful information — crucial for learning and completing the task.
- 2: Relevant, but not helpful — contributes somewhat to learning and completing the task but is not essential.
- 1: Not relevant or already present in the current tutorial — not useful for learning and completing the task.

Give a score between 1 and 3.
"""

RELEVANCE_CRITERIA_LIKERT_5 = """
Imagine that you are learning about the task based on the given current tutorial and received an additinal list of information. Evaluate the additional information based on the following criteria:
- 5: Extremely relevant and highly helpful information — crucial for learning and completing the task.
- 4: Highly relevant and helpful information — significantly supports learning and completing the task.
- 3: Relevant but moderately helpful information — contributes somewhat to learning and completing the task but is not essential.
- 2: Marginally relevant — information is related but not useful for learning and completing the task.
- 1: Not relevant or already present in the current tutorial — not useful for learning and completing the task.

Give a score between 1 and 5.
"""

RELEVANCE_CRITERIA_BINARY = """
Identify if the information in the response are relevant to the query or not:
yes: The information in the response are relevant to the query and is missing from the context tutorial.
no: The information in the response are not relevant to the query or is present in the context tutorial.

Assume that you are trying to learn based on the given context tutorial. If the information in the response is relevant to the query and is missing from the context tutorial, then say yes, otherwise say no.
"""

RELEVANCE_CRITERIA_COMPARISON = """
"""

COMPREHENSIVENESS_CRITERIA_COMPARISON = """
"""

def relevance_absolute_evaluation(eval_responses, test_dataset, metric, judge_model):
    criteria = ""
    if metric == MetricScale.LIKERT_3:
        criteria = RELEVANCE_CRITERIA_LIKERT_3
    elif metric == MetricScale.LIKERT_5:
        criteria = RELEVANCE_CRITERIA_LIKERT_5
    elif metric == MetricScale.BINARY:
        criteria = RELEVANCE_CRITERIA_BINARY
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    
    request_args = []
    req_idx_to_source = []
    for i, (eval_response, test_case) in enumerate(zip(eval_responses, test_dataset)):
        if eval_response is None or len(eval_response) == 0:
            continue
        task = test_case["task"]
        tutorial = test_case["tutorial"]
        segment = test_case["segment"]
        query = test_case["query"]

        request_args.append({
            "task": task,
            "tutorial": tutorial,
            "segment": segment,
            "query": query,
            "eval_response": eval_response,
            "metric": metric,
            "criteria": criteria,
            "judge_model": judge_model,
        })
        req_idx_to_source.append(i)


    batch_results = batch_run_lm_calls(request_args, eval_relevance_absolute_request, eval_relevance_absolute_response)

    results = [None] * len(test_dataset)
    for result, test_idx in zip(batch_results, req_idx_to_source):
        results[test_idx] = result

    print("Aggregated results:")
    print(json.dumps(aggregate_results(results, metric), indent=4))
    print("-"*20)
    return results

def faithfulness_absolute_evaluation(eval_responses, test_dataset, metric, judge_model):
    ### likely different approach (cosine similarity with the source texts?)
    pass

def relevance_comparative_evaluation(eval_responses_A, eval_responses_B, test_dataset, metric, judge_model):
    pass

def comprehensiveness_comparative_evaluation(eval_responses_A, eval_responses_B, test_dataset, metric, judge_model):
    pass


def aggregate_results(results, metric):
    ### return average decision/rating and average confidence
    aggregated_result = None
    if metric == MetricScale.COMPARISON:
        aggregated_result = aggregate_results_comparison(results)
    else:
        aggregated_result = aggregate_results_absolute(results)
    return {
        **aggregated_result,
        "metric": metric,
    }

def aggregate_results_comparison(results):
    a_win = 0
    b_win = 0
    a_win_confidence = 0
    b_win_confidence = 0
    available_results = 0
    for result in results:
        if result == None:
            continue
        total_reavailable_resultssults += 1
        if "decision" in result and result["decision"] == "A":
            a_win += 1
            a_win_confidence += result["confidence"]
        else:
            b_win += 1
            b_win_confidence += result["confidence"]
    
    a_win_rate = -1
    a_win_confidence = -1
    b_win_rate = -1
    b_win_confidence = -1
    if available_results > 0:
        a_win_rate = a_win / available_results
        a_win_confidence = a_win_confidence / a_win
        b_win_rate = b_win / available_results
        b_win_confidence = b_win_confidence / b_win
    return {
        "a_win_rate": a_win_rate,
        "a_win_confidence": a_win_confidence,
        "b_win_rate": b_win_rate,
        "b_win_confidence": b_win_confidence,
        "none_results": len(results) - available_results,
    }

def aggregate_results_absolute(results):
    total_score = 0
    total_confidence = 0
    available_results = 0
    for result in results:
        if result == None:
            continue
        available_results += 1
        if "decision" in result and result["decision"] == "yes":
            total_score += 1
        if "rating" in result:
            total_score += result["rating"]
        total_confidence += result["confidence"]
    average_score = -1
    average_confidence = -1
    if len(results) > 0:
        average_score = total_score / len(results)
        average_confidence = total_confidence / len(results)
    return {
        "average_score": average_score,
        "average_confidence": average_confidence,
        "none_results": len(results) - available_results,
    }

def get_scores_absolute(results):
    scores = []
    for result in results:
        if result is None:
            scores.append(1.0) ## minimum score
        else:
            scores.append(result["rating"])
    return scores