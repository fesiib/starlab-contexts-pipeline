"""
Script that runs the technical evaluation on IR tasks

- Sample test dataset & save them
    - Test data components:
        - Task: task name
        - Tutorial: the target tutorial
        - Segment: (if needed) the target segment
        - Query: the query to be answered
- Run all the conditions on the selected inputs
- Rate/compare the results with LLM-as-a-judge (human evaluation)
- Randomly sample 10% of the results for later human evaluation
"""

import random
import os
import json
import itertools
import numpy as np
from collections import defaultdict
from scipy.stats import ttest_ind, ttest_rel
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from scipy.stats import wilcoxon

from helpers.dataset import IMPORTANT_TYPES_FINE, IMPORTANT_TYPE_DESCRIPTIONS_FINE
from helpers.dataset import get_dataset

from eval import DatasetConfig, MethodConfig, EvalConfig

from helpers.dataset import MUFFIN_TASK, CUSTOM_TASKS, CROSS_TASK_TASKS, BIG_CUSTOM_TASKS

from src.rag import generic_call_rag
from src.cim_methods import generic_call_context_similarity, generic_call_shortest_path

from eval.llm_judge import relevance_absolute_evaluation
from eval.llm_judge import get_scores_absolute

from pydantic_models.evaluation import MetricScale

from helpers.video_scripts import get_transcript_segment

def sample_test_dataset(tasks, sample_per_task, query, n):
    test_dataset = []
    for task in tasks:
        dataset = get_dataset(task)
        sampled_tutorials = random.sample(dataset, sample_per_task)
        for tutorial in sampled_tutorials:
            info_type = random.choice(IMPORTANT_TYPES_FINE)
            info_type_description = IMPORTANT_TYPE_DESCRIPTIONS_FINE[info_type]
            info_type_description = info_type_description.strip()
            cur_query = query
            cur_query = cur_query.format(n=n, info_type=info_type, info_type_description=info_type_description)

            cur_segment = None
            if query == QUERIES[2] or query == QUERIES[3]:
                ## TODO: ensure that the `segments` attribute is present for crosstask dataset.
                selected_segment = random.choice(tutorial["segments"])
                cur_segment = {
                    "label": selected_segment["label"],
                    "content": get_transcript_segment(tutorial["transcript"], selected_segment["start"], selected_segment["end"], include_intersecting=True),
                    "start": selected_segment["start"],
                    "end": selected_segment["end"],
                }
                cur_query = cur_query.format(segment=cur_segment["label"])

            test_dataset.append({
                "task": task,
                "tutorial": tutorial,
                "segment": cur_segment,
                "query": cur_query,
                "info_type": info_type,
                "n": n,
            })

    return test_dataset

def construct_test_dataset(dataset_config):
    label = dataset_config.label
    test_dataset_path = os.path.join(TECH_EVAL_PATH, label + ".json")
    if os.path.exists(test_dataset_path):
        with open(test_dataset_path, "r") as f:
            return json.load(f)
    ### run sanity checks
    if dataset_config.query == QUERIES[2] and dataset_config.query == QUERIES[3]:
        for task in dataset_config.tasks:
            if task not in CROSS_TASK_TASKS:
                raise ValueError(f"Segment sampling is only supported for cross-task tasks. {task} is not a cross-task task.")

    test_dataset = sample_test_dataset(dataset_config.tasks, dataset_config.sample_per_task, dataset_config.query, dataset_config.n)
    with open(test_dataset_path, "w") as f:
        json.dump(test_dataset, f, indent=4)
    return test_dataset

def test_dataset_statistics(test_dataset):
    stats_per_query = {}
    for test in test_dataset:
        query = test["query"]
        if query not in stats_per_query:
            stats_per_query[query] = {
                "count": 0,
                "tasks": [],
                "segments": [],
            }
        stats_per_query[query]["count"] += 1
        stats_per_query[query]["tasks"].append(test["task"])
        stats_per_query[query]["segments"].append(test["segment"])
    print("Size: ", len(test_dataset))
    for query, stats in stats_per_query.items():
        print("-"*20)
        print(query)
        print(f"count: {stats['count']}")
        print(f"tasks: {stats['tasks']}")
        print(f"segments: {stats['segments']}")
        print("-"*20)

def run_method(method_config, test_dataset):
    method_func = method_config.func
    embedding_method = method_config.embedding_method
    k = method_config.k
    doc_score_threshold = method_config.doc_score_threshold
    version = method_config.version
    gen_model = method_config.gen_model
    responses = []

    ### group tests by task & restore the initial order
    tests_per_task = defaultdict(list)
    for idx, test in enumerate(test_dataset):
        task = test["task"]
        tests_per_task[task].append((idx, test))
        responses.append(None) ### placeholder for the response

    for task in tests_per_task.keys():
        dataset = get_dataset(task)
        cur_tests = [test for _, test in tests_per_task[task]]
        task_responses = method_func(task, version, dataset, cur_tests, embedding_method, gen_model, k, doc_score_threshold)
        if task_responses == None:
            continue
        initial_idxs = [initial_idx for initial_idx, _ in tests_per_task[task]]
        for idx, response in zip(initial_idxs, task_responses):
            responses[idx] = response
    return responses

def run_absolute_eval(dataset_config, method_config, eval_config):
    dataset_label = dataset_config.label
    eval_label = eval_config.label
    method_label = method_config.label

    results_path = os.path.join(TECH_EVAL_PATH, f"{dataset_label}_{method_label}_{eval_label}.json")

    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            return json.load(f)

    print("Constructing test dataset...")
    test_dataset = construct_test_dataset(dataset_config)
    test_dataset_statistics(test_dataset)

    print("Running method...")
    eval_responses = run_method(method_config, test_dataset)

    print("Running eval...")
    eval_func = eval_config.func
    eval_metric = eval_config.metric
    eval_judge_model = eval_config.judge_model
    results = eval_func(eval_responses, test_dataset, eval_metric, eval_judge_model)
    
    results = {
        "evaluation_type": "absolute",
        "results": results,
        "eval_responses": eval_responses,
        "dataset_config": dataset_config.to_dict(),
        "method_config": method_config.to_dict(),
        "eval_config": eval_config.to_dict(),
    }
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    return results

def run_comparative_eval(dataset_config, method_config_A, method_config_B, eval_config):
    dataset_label = dataset_config["label"]
    eval_label = eval_config["label"]
    method_labels = [method_config_A["label"], method_config_B["label"]]
    ### sort method_labels alphabetically
    method_labels.sort()
    method_labels = '_'.join(method_labels)
    results_path = os.path.join(TECH_EVAL_PATH, f"{dataset_label}_{method_labels}_{eval_label}.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            return json.load(f)
    
    print("Constructing test dataset...")
    test_dataset = construct_test_dataset(dataset_config)
    test_dataset_statistics(test_dataset)

    print("Running methods...")
    eval_responses_A = run_method(method_config_A, test_dataset)
    eval_responses_B = run_method(method_config_B, test_dataset)

    print("Running eval...")
    eval_func = eval_config["func"]
    results = eval_func(eval_responses_A, eval_responses_B, test_dataset, eval_config.metric)
    
    results = {
        "evaluation_type": "comparative",
        "results": results,
        "eval_responses_A": eval_responses_A,
        "eval_responses_B": eval_responses_B,
        "dataset_config": dataset_config,
        "method_config_A": method_config_A,
        "method_config_B": method_config_B,
        "eval_config": eval_config,
    }
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    return results

def main(dataset_idx, method_idx, eval_idx):
    dataset_config = DATASETS[dataset_idx]
    method_config = METHODS[method_idx]
    eval_config = EVALS[eval_idx]
    results = run_absolute_eval(dataset_config, method_config, eval_config)
    return results

TECH_EVAL_PATH = "./static/results/tech_eval/"

QUERIES = [
    "Given a tutorial, retrieve top-{n} missing, but relevant `{info_type}` information for the entire tutorial.\nFollowing is the definition of `{info_type}` information: \n{info_type_description}",
    "Given a tutorial, retrieve all missing, but relevant `{info_type}` information.\nFollowing is the definition of `{info_type}` information: \n{info_type_description}",
    "Given a tutorial and a highlighted segment, retrieve top-{n} missing, but relevant `{info_type}` information for the highlighted segment.\nFollowing is the definition of `{info_type}` information: \n{info_type_description}",
    "Given a tutorial, retrieve top-{n} missing, but relevant `{info_type}` information about the `{segment}`.\nFollowing is the definition of `{info_type}` information: \n{info_type_description}",
]

DATASETS = [
    DatasetConfig(
        label="test_2_q1_n2",
        tasks=[MUFFIN_TASK],
        sample_per_task=2,
        query=QUERIES[0],
        n=5,
    ),
    DatasetConfig(
        label="custom_test_10_q1_n5",
        tasks=[CUSTOM_TASKS[4]],
        sample_per_task=10,
        query=QUERIES[0],
        n=5,
    ),
    DatasetConfig(
        label="cross_10_q1_n5",
        tasks=CROSS_TASK_TASKS,
        sample_per_task=10,
        query=QUERIES[0],
        n=5,
    ),
    DatasetConfig(
        label="custom4_50_q1_n5",
        tasks=BIG_CUSTOM_TASKS,
        sample_per_task=50,
        query=QUERIES[0],
        n=5,
    ),
    DatasetConfig(
        label="cross4_50_q1_n5",
        tasks=CROSS_TASK_TASKS,
        sample_per_task=50,
        query=QUERIES[0],
        n=5,
    ),
    DatasetConfig(
        label="cross_10_q3_n5",
        tasks=CROSS_TASK_TASKS,
        sample_per_task=10,
        query=QUERIES[2],
        n=5,
    ),
    DatasetConfig(
        label="cross4_50_q3_n5",
        tasks=CROSS_TASK_TASKS,
        sample_per_task=50,
        query=QUERIES[2],
        n=5,
    ),
]

METHODS = [
    MethodConfig(
        label="RAG-openai-4-0.0",
        embedding_method="openai",
        k=4,
        doc_score_threshold=None,
        func=generic_call_rag,
        version=None,
        gen_model="gpt-4.1-mini-2025-04-14",
    ),
    MethodConfig(
        label="RAG-openai-8-0.0",
        embedding_method="openai",
        k=8,
        doc_score_threshold=None,
        func=generic_call_rag,
        version=None,
        gen_model="gpt-4.1-mini-2025-04-14",
    ),
    MethodConfig(
        label="RAG-openai-16-0.0",
        embedding_method="openai",
        k=16,
        doc_score_threshold=None,
        func=generic_call_rag,
        version=None,
        gen_model="gpt-4.1-mini-2025-04-14",
    ),
    MethodConfig(
        label="RAG-openai-2-0.0",
        embedding_method="openai",
        k=2,
        doc_score_threshold=None,
        func=generic_call_rag,
        version=None,
        gen_model="gpt-4.1-mini-2025-04-14",
    ),
    MethodConfig(
        label="vanilla-openai",
        embedding_method="openai",
        k=None,
        doc_score_threshold=None,
        func=generic_call_rag,
        version=None,
        gen_model="gpt-4.1-mini-2025-04-14",
    ),
    MethodConfig(
        label="context-similarity-openai",
        embedding_method="openai",
        k=None,
        doc_score_threshold=None,
        func=generic_call_context_similarity,
        version="full_run_11",
        gen_model="gpt-4.1-mini-2025-04-14",
    ),
    MethodConfig(
        label="shortest-path-openai",
        embedding_method="openai",
        k=None,
        doc_score_threshold=None,
        func=generic_call_shortest_path,
        version="full_run_11",
        gen_model="gpt-4.1-mini-2025-04-14",
    ),
]

EVALS = [
    EvalConfig(
        label="relevance-absolute-evaluation-gpt-5-3",
        func=relevance_absolute_evaluation,
        metric=MetricScale.LIKERT_3,
        judge_model="gpt-5-mini-2025-08-07",
    ),
    EvalConfig(
        label="relevance-absolute-evaluation-gpt-5-5",
        func=relevance_absolute_evaluation,
        metric=MetricScale.LIKERT_5,
        judge_model="gpt-5-mini-2025-08-07",
    ),
    EvalConfig(
        label="relevance-absolute-evaluation-gpt-5-binary",
        func=relevance_absolute_evaluation,
        metric=MetricScale.BINARY,
        judge_model="gpt-5-mini-2025-08-07",
    ),
    EvalConfig(
        label="faithfulness-absolute-evaluation-gpt-5",
        func=None,
        metric=None,
        judge_model="gpt-5-mini-2025-08-07",
    ),
    EvalConfig(
        label="relevance-comparative-evaluation-gpt-5",
        func=None,
        metric=MetricScale.COMPARISON,
        judge_model="gpt-5-mini-2025-08-07",
    ),
    EvalConfig(
        label="comprehensiveness-comparative-evaluation-gpt-5",
        func=None,
        metric=MetricScale.COMPARISON,
        judge_model="gpt-5-mini-2025-08-07",
    ),
]

def t_test(scores_a, scores_b, path):
    
    np_a = np.array(scores_a)
    np_b = np.array(scores_b)
    print(np.mean(np_a), np.std(np_a))
    print(np.mean(np_b), np.std(np_b))

    ### draw the distribution of score_a-score_b
    diff = np_a - np_b
    plt.hist(diff)
    plt.savefig(path)
    plt.close()
    ### check if diff is normally distributed
    shapiro_test = shapiro(diff)
    if shapiro_test.pvalue > 0.05:
        ### Paired t-test
        print("Diff is normally distributed", shapiro_test)
        t_stat, p_value = ttest_rel(np_a, np_b, nan_policy="omit", alternative="two-sided")
        return t_stat, p_value, "paired t-test"
    else:
        ### Wilcoxon Signed-Rank Test
        print("Diff is not normally distributed", shapiro_test)
        t_stat, p_value = wilcoxon(np_a, np_b, correction=True, alternative="two-sided")
        return t_stat, p_value, "wilcoxon"

if __name__ == "__main__":
    # dataset_idxs = [5, 6]
    dataset_idxs = [3]
    # method_idxs = [0, 1, 2, 3] ### RAGs 2 is best
    method_idxs = [3, 0, 1, 2, 5] ### Context Similarity & Shortest Path
    eval_idxs = [1]

    all_combinations = list(itertools.product(dataset_idxs, method_idxs, eval_idxs))
    
    organize_results = defaultdict(lambda: defaultdict(list))
    
    for combination in all_combinations:
        dataset_idx, method_idx, eval_idx = combination
        cur_results = main(dataset_idx, method_idx, eval_idx)
        organize_results[method_idx][f"{dataset_idx}_{eval_idx}"] = cur_results

    print(json.dumps(organize_results, indent=4))

    # dataset_idx, method_idx, eval_idx = all_combinations[0]
    # main(dataset_idx, method_idx, eval_idx)

    for method_1 in organize_results.keys():
        for method_2 in organize_results.keys():
            if method_1 == method_2:
                continue
            common_results = organize_results[method_1].keys() & organize_results[method_2].keys()
            if len(common_results) == 0:
                continue
            scores_a = []
            scores_b = []
            for common_result_key in common_results:
                cur_results_1 = organize_results[method_1][common_result_key]
                cur_scores_1 = get_scores_absolute(cur_results_1["results"])
                cur_results_2 = organize_results[method_2][common_result_key]
                cur_scores_2 = get_scores_absolute(cur_results_2["results"])
                scores_a.extend(cur_scores_1)
                scores_b.extend(cur_scores_2)
            folder_path = os.path.join(TECH_EVAL_PATH, "diff_distribution")
            file_name = f"{method_1}_{method_2}.png"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            path = os.path.join(folder_path, file_name)
            stat, p_value, test_type = t_test(scores_a, scores_b, path)
            print(f"Method: {method_1}, Method: {method_2}, Stat: {stat}, P-value: {p_value}, Test Type: {test_type}")
            print("-"*20)
    ### print representative results per score
    sampled_indexes = range(200)
    results_per_index = defaultdict(list)
    for method_idx in organize_results.keys():
        for eval_idx in organize_results[method_idx].keys():
            cur_results = organize_results[method_idx][eval_idx]
            cur_dataset_config = DatasetConfig.from_dict(cur_results["dataset_config"])
            test_dataset = construct_test_dataset(cur_dataset_config)
            for idx in sampled_indexes:
                cur_test = test_dataset[idx]
                cur_result = cur_results["results"][idx]
                if cur_result is None:
                    cur_result = "<none>"
                else:
                    cur_result = f"{cur_result['rating']}<br>({cur_result['reasoning']})"
                cur_response = cur_results["eval_responses"][idx]
                if cur_response is None:
                    cur_response = "<none>"
                else:
                    cur_response = "<br><br>".join([item["content"] for item in cur_response])
                results_per_index[idx].append({
                    "method": method_idx,
                    "eval": eval_idx,
                    "task": cur_test["task"],
                    "tutorial": cur_test["tutorial"]["url"],
                    "segment": cur_test["segment"]["label"] if cur_test["segment"] is not None else None,
                    "query": cur_test["query"],
                    "result": cur_result,
                    "response": cur_response,
                })
    
    ### print as .md table
    print("| Index | Task | Tutorial | Segment | Query | RAG(2) | RAG(4) | RAG(8) | RAG(16) | Context Similarity |")
    print("|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|")
    for idx, results in results_per_index.items():
        index = idx + 1
        task = results[0]["task"]
        tutorial = results[0]["tutorial"]
        segment = results[0]["segment"]
        query = results[0]['query'].replace("\n", "<br>")
        rag2_result = results[0]["result"]
        rag4_result = results[1]["result"]
        rag8_result = results[2]["result"]
        rag16_result = results[3]["result"]
        context_similarity_result = results[4]["result"]
        rag2_response = results[0]["response"]
        rag4_response = results[1]["response"]
        rag8_response = results[2]["response"]
        rag16_response = results[3]["response"]
        context_similarity_response = results[4]["response"]
        print(f"| {index} | {task} | {tutorial} | {segment} | {query} | {rag2_result} | {rag4_result} | {rag8_result} | {rag16_result} | {context_similarity_result} |")
        print(f"| | | | | | {rag2_response} | {rag4_response} | {rag8_response} | {rag16_response} | {context_similarity_response} |")
            
    