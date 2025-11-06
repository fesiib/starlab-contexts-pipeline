import sys
import os

from helpers.cim_scripts import get_cell_to_units, calc_discriminativeness

from helpers.dataset import MUFFIN_TASK, CUSTOM_TASKS, CROSS_TASK_TASKS

from analysis.frontier import plot_frontiers_facets, plot_frontiers_labels
from analysis.frontier import get_available_results, classify_facet_candidates
from analysis.frontier import plot_size_vs_complexity

from analysis.display import show_task_stats

from analysis import ANALYSIS_PATH

def plot_results(results, piece_types, common_threshold=None):
    args = sys.argv[1:]
    folder = "noname_frontiers"
    if len(args) > 0 and len(args[0]) > 3:
        folder = args[0]

    output_folder = os.path.join(ANALYSIS_PATH, folder)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    elbow_d = 1
    y_axis = "discriminativeness"

    # plot_frontiers_facets(results, piece_types, elbow_d, y_axis, output_folder)
    # plot_frontiers_labels(results, piece_types, elbow_d, y_axis, output_folder)
    plot_size_vs_complexity(results, piece_types, elbow_d, output_folder)
    
    ### classify the facet candidates into common vs unique to the task
    if common_threshold is not None:
        sim_thresh = 0.9
        embedding_method = "openai"
        results = classify_facet_candidates(results, sim_thresh, common_threshold, embedding_method)
    
    # plot_frontiers_facets(results, piece_types, elbow_d, y_axis, output_folder)
    # plot_frontiers_labels(results, piece_types, elbow_d, y_axis, output_folder)
    
    plot_size_vs_complexity(results, piece_types, elbow_d, output_folder)

def main():

    piece_types = ["Method - Subgoal", "Method - Instruction", "Method - Tool",
        "Supplementary - Tip", "Supplementary - Warning",
        "Explanation - Justification", "Explanation - Effect",
        "Description - Status", "Description - Context", "Description - Tool Specification",
        "Conclusion - Outcome", "Conclusion - Reflection",
    ]

    # tasks = CUSTOM_TASKS + [MUFFIN_TASK]

    # hat_tasks = [CUSTOM_TASKS[14], CUSTOM_TASKS[14]]
    # hat_dummies = ["trial_1", "full_run_1"]
    # results = get_available_results(hat_tasks, hat_dummies)

    tasks_trial_1 = [CUSTOM_TASKS[14]] + [MUFFIN_TASK]
    dummies_trial_1 = ["trial_1"] * 2
    # results = get_available_results(tasks_trial_1, dummies_trial_1)

    tasks_full_run_1 = CUSTOM_TASKS
    dummies_full_run_1 = ["full_run_1"] * len(tasks_full_run_1)

    tasks_full_run_2 = CUSTOM_TASKS
    dummies_full_run_2 = ["full_run_2"] * len(tasks_full_run_2)

    tasks_full_run_3 = CROSS_TASK_TASKS
    dummies_full_run_3 = ["full_run_3"] * len(tasks_full_run_3)

    tasks = tasks_full_run_1 + tasks_full_run_2 + tasks_full_run_3
    dummies = dummies_full_run_1 + dummies_full_run_2 + dummies_full_run_3

    results = get_available_results(tasks, dummies)
    
    # key1 = list(results.keys())[0]
    # key2 = list(results.keys())[1]
    # small_results = {
    #     key1: results[key1],
    #     key2: results[key2],
    # }
    # results = small_results

    # print("Tasks: ", len(results))
    # for task, result in results.items():
    #     show_task_stats(task, result, piece_types)


    desired_d = 0.5
    common_threshold = 0.8
    ## filter out results that have not reached the desired discriminativeness
    filtered_results = {}
    for task, result in results.items():
        cell_to_units, relevant_units_count = get_cell_to_units(result["facet_candidates"], result["labeled_dataset"], piece_types)
        cur_d = calc_discriminativeness(cell_to_units, relevant_units_count)
        if cur_d < desired_d:
            filtered_results[task] = result

    print(f"Only {len(filtered_results)} tasks left out of {len(results)}")
    results = filtered_results

    plot_results(results, piece_types, common_threshold)

if __name__ == "__main__":
    main()