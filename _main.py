import json
import os
import random
from contextlib import redirect_stdout, redirect_stderr

from helpers.dataset import MUFFIN_TASK, CUSTOM_TASKS, CROSS_TASK_TASKS, BIG_CUSTOM_TASKS
from helpers.dataset import get_dataset
from helpers.cim_scripts import FRAMEWORK_PATH

from src.framework_split import construct_cim_split
from src.framework_iter import construct_cim_iter

from src.cim_methods import context_similarity_retrieval, run_cim_method

from helpers.video_scripts import get_transcript_segment

def run_framework(task, version):
    output_dir = os.path.join(FRAMEWORK_PATH, version)
    os.makedirs(output_dir, exist_ok=True)

    output_file = f"split_{task.replace(' ', '_').lower()}.txt"
    output_path = os.path.join(output_dir, output_file)

    # open once per task; capture BOTH stdout and stderr
    with open(output_path, "w", buffering=1) as f, redirect_stdout(f), redirect_stderr(f):
        dataset = get_dataset(task)  # your function
        _ = construct_cim_split(task, dataset, version)  # your function
        print("COMPLETED", task)

def run_framework_small_custom_tasks(version):
    SMALL_CUSTOM_TASKS = []
    for task in CUSTOM_TASKS:
        if task in BIG_CUSTOM_TASKS:
            continue
        SMALL_CUSTOM_TASKS.append(task)
    for task in SMALL_CUSTOM_TASKS:
        run_framework(task, version)
    print("COMPLETED FULL RUN SMALL CUSTOM TASKS!")

def run_framework_big_custom_tasks(version):
    for task in BIG_CUSTOM_TASKS:
        run_framework(task, version)
    print("COMPLETED FULL RUN BIG CUSTOM TASKS!")

def run_framework_cross_tasks(version):
    for task in CROSS_TASK_TASKS:
        run_framework(task, version)
    print("COMPLETED FULL RUN CROSS TASK TASKS!")

def run_context_similarity(task, version):
    dataset = get_dataset(task)
    embedding_method = "openai"
    
    tutorial = dataset[0]
    segment = None
    if "segments" in tutorial:
        selected_segment = random.choice(tutorial["segments"])
        segment = {
            "label": selected_segment["label"],
            "content": get_transcript_segment(tutorial["transcript"], selected_segment["start"], selected_segment["end"], include_intersecting=True),
            "start": selected_segment["start"],
            "end": selected_segment["end"],
        }
        print(json.dumps(tutorial["content"], indent=4))
        print(json.dumps(tutorial["segments"], indent=4))
        print(json.dumps(segment, indent=4))

    tests = [{
        "tutorial": tutorial,
        "segment": segment,
        "info_type": "Supplementary - Tip",
        "n": 5
    }]
    func = context_similarity_retrieval

    response = run_cim_method(task, version, dataset, tests, embedding_method, func)

    print(json.dumps(response, indent=4))

def run_rag():
    pass

def run_vanilla():
    pass

if __name__ == "__main__":
    for task in BIG_CUSTOM_TASKS:
        run_context_similarity(task, "full_run_11")
    # run_framework(MUFFIN_TASK, "full_run_0")
    # run_framework_small_custom_tasks("full_run_0")
    # for task in BIG_CUSTOM_TASKS:
    #     if task == BIG_CUSTOM_TASKS[2]:
    #         continue
    #     run_framework(task, "full_run_11")
    # run_framework_big_custom_tasks("full_run_10")
    # run_framework({"task": BIG_CUSTOM_TASKS[0], "version": "full_run_2"})
    # run_framework_cross_tasks("full_run_4")
    # run_framework({"task": CROSS_TASK_TASKS[0], "version": "full_run_3"})