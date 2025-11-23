import sys
import os
from contextlib import redirect_stdout, redirect_stderr

from helpers.cim_scripts import get_cell_to_units, calc_discriminativeness

from helpers.dataset import MUFFIN_TASK, CUSTOM_TASKS, CROSS_TASK_TASKS, BIG_CUSTOM_TASKS, IMPORTANT_TYPES_FINE

from analysis.frontier import plot_frontiers_facets, plot_frontiers_labels
from analysis.frontier import get_available_results, classify_facet_candidates
from analysis.frontier import plot_size_vs_complexity

from analysis.display import show_task_stats, display_tutorial_context_deltas

from analysis import ANALYSIS_PATH

def plot_results(output_folder, results, piece_types, common_threshold=None):
    
    elbow_d = 1
    y_axis = "discriminativeness"

    plot_frontiers_facets(results, piece_types, elbow_d, y_axis, output_folder)
    plot_frontiers_labels(results, piece_types, elbow_d, y_axis, output_folder)
    plot_size_vs_complexity(results, piece_types, elbow_d, output_folder)
    
    ### classify the facet candidates into common vs unique to the task
    if common_threshold is not None:
        sim_thresh = 0.9
        embedding_method = "openai"
        results = classify_facet_candidates(results, sim_thresh, common_threshold, embedding_method)
    
    plot_frontiers_facets(results, piece_types, elbow_d, y_axis, output_folder)
    plot_frontiers_labels(results, piece_types, elbow_d, y_axis, output_folder)
    
    plot_size_vs_complexity(results, piece_types, elbow_d, output_folder)

def get_corpus_last_version():
    task_to_version = {}

    for task in CUSTOM_TASKS:
        task_to_version[task] = "full_run_0"

    for task in CROSS_TASK_TASKS:
        task_to_version[task] = "full_run_4"
    
    for task in BIG_CUSTOM_TASKS:
        task_to_version[task] = "full_run_11"

    return list(task_to_version.keys()), list(task_to_version.values())

def main_run(results, piece_types, output_folder, common_threshold):
    print("Tasks: ", len(results))
    for task, result in results.items():
        output_dir = os.path.join(output_folder, f"{task.replace(' ', '_').lower()}_info")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "task_info.txt")
        with open(output_path, "w", buffering=1) as f, redirect_stdout(f), redirect_stderr(f):
            show_task_stats(task, result, piece_types)
        key_to_name = {}
        for facet in result["context_schema"]:
            key_to_name[facet["id"]] = facet["title"]
        for idx, tutorial in enumerate(result["labeled_dataset"]):
            output_path = os.path.join(output_dir, f"tutorial_{idx}_deltas.md")
            with open(output_path, "w", buffering=1) as f, redirect_stdout(f), redirect_stderr(f):
                display_tutorial_context_deltas(tutorial, key_to_name, include_content_types=piece_types)



    desired_d = 0.5
    ## filter out results that have not reached the desired discriminativeness
    filtered_results = {}
    for task, result in results.items():
        cell_to_units, relevant_units_count = get_cell_to_units(result["facet_candidates"], result["labeled_dataset"], piece_types)
        cur_d = calc_discriminativeness(cell_to_units, relevant_units_count)
        if cur_d < desired_d:
            filtered_results[task] = result

    print(f"Only {len(filtered_results)} tasks left out of {len(results)}")
    results = filtered_results

    plot_results(output_folder, results, piece_types, common_threshold)

def main():
    args = sys.argv[1:]
    folder = "noname_frontiers"
    if len(args) > 0 and len(args[0]) > 3:
        folder = args[0]
    print(f"Plotting results to folder: {folder}")
    output_folder = os.path.join(ANALYSIS_PATH, folder)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    piece_types = IMPORTANT_TYPES_FINE

    ### smaller vs larger
    tasks, versions = get_corpus_last_version()
    tasks_small = []
    versions_small = []
    tasks_large = []
    versions_large = []
    for task, version in zip(tasks, versions):
        if version == "full_run_0":
            tasks_small.append(task)
            versions_small.append(version)
        else:
            tasks_large.append(task)
            versions_large.append(version)
    print(tasks_small, tasks_large)
    results_small = get_available_results(tasks_small, versions_small)
    results_large = get_available_results(tasks_large, versions_large)
    output_folder_small = os.path.join(output_folder, "smaller")
    output_folder_large = os.path.join(output_folder, "larger")
    os.makedirs(output_folder_small, exist_ok=True)
    os.makedirs(output_folder_large, exist_ok=True)

    main_run(results_small, piece_types, output_folder_small, 0.5)
    main_run(results_large, piece_types, output_folder_large, 0.5)

    #tasks, versions = get_corpus_last_version()
    for task in BIG_CUSTOM_TASKS:
        tasks = [task, task]
        versions = ["full_run_5", "full_run_11"]
        cur_output_folder = os.path.join(output_folder, f"{task.replace(' ', '_').lower()}")
        os.makedirs(cur_output_folder, exist_ok=True)

        results = get_available_results(tasks, versions)
        main_run(results, piece_types, cur_output_folder, 0.5)

if __name__ == "__main__":
    main()