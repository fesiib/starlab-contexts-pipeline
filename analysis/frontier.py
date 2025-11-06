import json
import matplotlib.pyplot as plt
import numpy as np
import os
import math
from collections import defaultdict


from src.framework_split import construct_cim_split_conservative
from src.framework_split import compute_frontier_knapsack
from helpers.nlp import clustering_custom

from analysis.display import count_dataset_stats

Y_TITLES = {
    "discriminativeness": "Discriminativeness",
    "explained_norm": "Explained Variance",
}
X_TITLES = {
    "units": "Number of Units in the Corpus",
    "facets": "Number of Facets",
    "vocabulary_labels": "Number of Vocabulary Labels",
}

Y_DESIRED = {
    "discriminativeness": 0.8,
    "explained_norm": 0.9,
}

def find_piece_by_unit(dataset, unit_id, info_types):
    for tutorial in dataset:
        for piece in tutorial['pieces']:
            if piece['content_type'] not in info_types:
                continue
            if piece['unit_id'] == unit_id:
                return piece['content']
    return None

def get_facets(facet_candidates, facet_titles):
    facets = []
    for facet_title in facet_titles:
        for facet in facet_candidates:
            if facet["title"] == facet_title:
                facets.append(facet)
    return facets

def distribution_of_labels(results, top_k=5):
    labels_count = defaultdict(int)
    facet_name_count = defaultdict(int)
    facet_presence_count = defaultdict(set)
    for task, result in results.items():
        facet_candidates = result["facet_candidates"]
        for facet in facet_candidates:
            facet_name_count[facet["title"]] += 1
            labels_count[len(facet["vocabulary"])] += 1
            facet_presence_count[facet["title"]].add(task)
    
    for facet_name in facet_name_count.keys():
        ### average across tasks
        facet_name_count[facet_name] /= len(facet_presence_count[facet_name])
    
    top_k_facet_names = sorted(facet_name_count.items(), key=lambda x: x[1], reverse=True)[:top_k]
    

    top_k_facet_names = [x[0] for x in top_k_facet_names]
    plt.bar(top_k_facet_names, [facet_name_count[x] for x in top_k_facet_names])
    plt.show()

    plt.bar(labels_count.keys(), labels_count.values())
    plt.show()

def calc_sparsity(cell_to_units, display_size=-1):
    cell_sizes = defaultdict(int)
    for cell, units in cell_to_units.items():
        cell_sizes[len(units)] += 1
        if len(units) == display_size:
            print(cell)
            for unit in units:
                print(unit)

    cell_sizes = sorted(cell_sizes.items(), key=lambda x: x[0])
    
    for size, count in cell_sizes:
        print(f"{size} unit size: {count} cells")

def show_frontier_item(item, facet_candidates):
    agg_facets = []
    # agg_vocab = []
    for facet_id in item["facets"]:
        for facet in facet_candidates:
            if facet["id"] == facet_id:
                agg_facets.append(facet["title"])
                # for label in facet["vocabulary"]:
                #     agg_vocab.append(label["label"])
    print("ELBOW FACETS: ", agg_facets)
    # print("ELBOW VOCAB: ", agg_vocab)

def get_colors(n):
    colors = plt.cm.rainbow(np.linspace(0, 1, n))
    ### make the color 50% transparent
    # colors = [(color[0], color[1], color[2], 0.5) for color in colors]
    return colors

def plot_frontier(plt, x, y, label, color, linestyle, marker, markersize, axvlines=[]):
    # for _x, _y, _label in axvlines:
    #     plt.axvline(x=_x, color=color, linestyle=':')
        ## plt.text(_x, _y, _label)
    plt.plot(x, y, label=label, color=color, linestyle=linestyle, marker=marker, markersize=markersize)

def interpolate_frontier_precise(x, y, y_new):
    """
    Interpolate the frontier. Assumes x is non-decreasing.
    """
    precision = 1e-9
    for i in range(len(x)):
        if math.fabs(y[i] - y_new) < precision:
            return x[i]
    
    for i in range(len(x) - 1):
        p1 = i
        p2 = i + 1
        if y[p2] < y[p1]:
            p1, p2 = p2, p1
        if y_new > y[p1] and y[p2] > y_new:
            return x[i] + (x[i + 1] - x[i]) * (y_new - y[p1]) / (y[p2] - y[p1])
    return None

def extrapolate_frontier_precise(x, y, y_new):
    """
    Extrapolate the frontier. Assumes x is a non-decreasing list.
    Fit the data to y = a + b/x and return the value of x when y = y_new.
    """
    if len(x) < 2:
        return None ### not enough data to fit a line
    
    ### fit the data to y = a + b/x
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask = x > 0
    x, y = x[mask], y[mask] ### remove zeros
    A = np.column_stack([np.ones(len(x)), 1/x]) ### [1, 1/x]
    coef, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
    a, b = coef

    ### Diagnostics
    n = len(y)
    p = 2
    rss = residuals[0] if residuals.size else np.sum((y - A @ coef)**2)
    sigma2 = rss / max(n-p, 1)
    cov = sigma2 * np.linalg.inv(A.T @ A) ### covariance matrix of the coefficients
    se_a, se_b = np.sqrt(np.diag(cov))
    print(f"a: {a:.2f} (se: {se_a:.2f}), b: {b:.2f} (se: {se_b:.2f}) rss: {rss:.2f}, n: {n:.2f}")
    return b/(y_new - a) ### return the value of x when y = y_new

def interpolate_frontier_approx(x, y, y_new):
    best_x = -1
    best_dist = float("inf")
    for xx, yy in zip(x, y):
        cur_dist = abs(yy - y_new)
        if cur_dist < best_dist:
            best_x = xx
            best_dist = cur_dist
    if best_dist > 1e-1:
        return None
    return best_x

def plot_frontiers(
    frontiers, x_axis, y_axis, x_lims, y_lims, output_path
):
    plt.figure(figsize=(10, 10))

    colors = get_colors(len(frontiers))

    cur_y_title = Y_TITLES[y_axis]
    cur_y_desired = Y_DESIRED[y_axis]

    cur_x_title = X_TITLES[x_axis]

    for info, color in zip(frontiers, colors):
        frontier = info["frontier"]
        elbows = info["elbows"]
        label = info["task"]

        x = [item["compactness"] for item in frontier]
        y = [item[y_axis] for item in frontier]
        
        best_c = interpolate_frontier_approx(x, y, cur_y_desired)
        if best_c is None:
            best_c = 100 ### TODO: extrapolate the data later if needed
        axvlines = []
        axvlines.append((best_c, cur_y_desired, f"{cur_y_desired:.2f}"))
        for elbow, w_d in elbows:
            axvlines.append((elbow["compactness"], elbow[y_axis], f"{w_d:.2f}"))

        linestyle = "-"
        marker = "o"
        markersize = 3
        if label.startswith("common_"):
            linestyle = ":"
            marker = "D" ## diamond
        plot_frontier(plt, x, y, label, color, linestyle, marker, markersize, axvlines=axvlines)

    plt.axhline(y=cur_y_desired, color='r', linestyle='--')

    plt.legend()
    plt.xlabel(cur_x_title)
    plt.ylabel(cur_y_title)
    plt.title(f"{cur_x_title} vs {cur_y_title} Frontier")
    if x_lims is not None:
        plt.xlim(left=x_lims[0], right=x_lims[1])
        plt.xticks(np.arange(x_lims[0], x_lims[1] + 1, x_lims[2]))
    if y_lims is not None:
        plt.ylim(bottom=y_lims[0], top=y_lims[1])
        plt.yticks(np.arange(y_lims[0], y_lims[1] + 1, y_lims[2]))
    plt.savefig(output_path)
    plt.close()

def find_elbow(frontier, w_d):
    if len(frontier) == 0:
        return None
    w_c = 1 - w_d
    optimal_idx = -1
    best_score = float("inf")
    for idx, item in enumerate(frontier):
        cur_d = item["discriminativeness"]
        cur_c = item["compactness"]
        score = w_d * cur_d + w_c * cur_c
        if score < best_score:
            best_score = score
            optimal_idx = idx
    return frontier[optimal_idx]

def find_elbows(frontier, base_d, inc=0.01):
    cur_elbows = []
    w_d = base_d
    while w_d < 1.0:
        elbow_item = find_elbow(frontier, w_d)
        if elbow_item is None:
            continue
        cur_elbows.append((elbow_item, w_d))
        w_d += inc
    return cur_elbows

def get_info_for_results(results, cur_types, w_d, max_label_count=None, over_values=True):
    info_per_results = []
    for task, result in results.items():
        dataset = result["labeled_dataset"]
        facet_candidates = result["facet_candidates"]
        frontier = compute_frontier_knapsack(dataset, cur_types, facet_candidates, max_label_count, over_values)
        elbows = []
        dataset_stats = count_dataset_stats(dataset, cur_types)
        if w_d is not None:
            elbow_item = find_elbow(frontier, w_d)
            if elbow_item is not None:
                elbows.append((elbow_item, w_d))

        info_per_results.append({
            "task": task,
            "frontier": frontier,
            "elbows": elbows,
            "dataset_stats": dataset_stats,
        })
    return info_per_results

def plot_frontiers_facets(results, piece_types, elbow_d, y_axis, output_folder):
    max_label_count = None
    over_values = False
    x_axis = "facets"
    x_lims = (0, 20, 1)
    y_lims = (0, 4, 0.5)
    
    info_per_results = get_info_for_results(results, piece_types, elbow_d, max_label_count, over_values)
    
    output_path = os.path.join(output_folder, f"frontier_facets_{y_axis}_{str(elbow_d)}_{len(results)}.png")
    plot_frontiers(info_per_results, x_axis, y_axis, x_lims, y_lims, output_path)

def plot_frontiers_labels(results, piece_types, elbow_d, y_axis, output_folder):
    max_label_count = None
    over_values = True
    x_axis = "vocabulary_labels"
    x_lims = (10, 100, 5)
    y_lims = (0, 10, 0.5)
    
    info_per_results = get_info_for_results(results, piece_types, elbow_d, max_label_count, over_values)

    output_path = os.path.join(output_folder, f"frontier_labels_{y_axis}_{str(elbow_d)}_{len(results)}.png")
    plot_frontiers(info_per_results, x_axis, y_axis, x_lims, y_lims, output_path)

def get_available_results(tasks, dummies):
    results = {}
    for task, version in zip(tasks, dummies):
        result = construct_cim_split_conservative(task, version)
        if result is None:
            continue
        task_name = task.lower().replace(" ", "_") + "_" + version
        results[task_name] = result
    return results

def classify_facet_candidates(results, similarity_threshold, common_threshold, embedding_method):
    """
    Classify the facet candidates into common vs unique to the task.
    """
    updated_results = {}
    facet_id_to_class = {}
    facet_id_to_task = {}
    facet_texts_per_id = {}
    all_unique_tasks = set()
    for task, result in results.items():
        all_unique_tasks.add(task)
        facet_candidates = result["facet_candidates"]

        for facet in facet_candidates:
            ## facet_text = f"{facet['title']}: {facet['definition']}"
            facet_text = facet['title']
            facet_texts_per_id[facet['id']] = facet_text
            facet_id_to_task[facet['id']] = task

    ## cluster facet_texts into clusters
    ## assign a cluster a class `common` or `unique` depending on the size of the cluster (if cluster size = 1, it is unique, otherwise it is common)
    clusters = clustering_custom(facet_texts_per_id.values(), similarity_threshold, embedding_method)
    cluster_sizes = defaultdict(list)
    for facet_id, cluster in zip(facet_texts_per_id.keys(), clusters):
        cluster_sizes[cluster].append(facet_id)
    

    print("To be classified common, need", common_threshold * len(all_unique_tasks), "tasks")
    unique_tasks_cluster_count = defaultdict(int)
    for cluster, facet_ids in cluster_sizes.items():
        unique_tasks = set()
        for facet_id in facet_ids:
            unique_tasks.add(facet_id_to_task[facet_id])
        unique_tasks_cluster_count[len(unique_tasks)] += len(facet_ids)
        # cur_class = "common"
        # if len(unique_tasks) == 1:
        #     cur_class = "unique"
        cur_class = "unique"
        ratio = len(unique_tasks) / len(all_unique_tasks)
        if ratio > (common_threshold - 1e-9): ### Reasoning: the facet is common if it is present in at least half of the tasks
            cur_class = "common"
        
        for facet_id in facet_ids:
            facet_id_to_class[facet_id] = cur_class

    print(f"Total facet candidates: {len(facet_texts_per_id)}")
    print(json.dumps(unique_tasks_cluster_count, indent=4))

    for task, result in results.items():
        task_common = f"common_{task}"
        task_unique = f"unique+common_{task}"
        facet_candidates = result["facet_candidates"]

        candidates_common = []
        candidates_unique = []
        for facet in facet_candidates:
            if facet_id_to_class[facet['id']] == "common":
                ### add the common to unique as well
                candidates_common.append(facet)
                candidates_unique.append(facet)
                print("common", facet['title'], facet['definition'], task)
            else:
                candidates_unique.append(facet)
                print("unique+common", facet['title'], facet['definition'], task)

        updated_results[task_common] = {
            **result,
            "facet_candidates": candidates_common,
        }
        updated_results[task_unique] = {
            **result,
            "facet_candidates": candidates_unique,
        }
    return updated_results

def plot_size_vs_complexity(results, piece_types, elbow_d, output_folder):
    max_label_count = None
    over_values = True
    y_axis = "discriminativeness"
    
    info_per_results = get_info_for_results(results, piece_types, elbow_d, max_label_count, over_values)

    output_path = os.path.join(output_folder, f"size_vs_complexity_{str(elbow_d)}_{len(results)}.png")

    sizes = []
    complexities = []

    desired_y = Y_DESIRED[y_axis]

    for info in info_per_results:
        x = [item["compactness"] for item in info["frontier"]]
        y = [item["discriminativeness"] for item in info["frontier"]]
        best_c = interpolate_frontier_precise(x, y, desired_y)
        if best_c is None:
            continue ### TODO: later extrapolate the data based on the sizes x compactness
        complexities.append((best_c, info["task"]))
        sizes.append(info["dataset_stats"]["count_pieces"])
    
    if len(sizes) == 0:
        print("No data points found")
        return

    scatters = defaultdict(lambda: {"x": [], "y": []})
    for i, (best_c, task) in enumerate(complexities):
        kind = task[:6] ### common_ or unique+common_
        scatters[kind]["x"].append(sizes[i])
        scatters[kind]["y"].append(best_c)

    colors = get_colors(len(scatters))
    x_lims = (0, 0, 500)
    y_lims = (0, 0, 5)

    plt.figure(figsize=(10, 10))
    markers = ["o", "D", "s", "v", "^", "x", "P", "H", "8", "p", "d", "|", "_"]
    markersize = 10

    for i, (kind, scatter) in enumerate(scatters.items()):
        color = colors[i]
        marker = markers[i]
        print(f"Optimal Avg. Compactness for d={elbow_d}: {np.average(scatter['y']):.2f}")
        print(f"Optimal Std. Compactness for d={elbow_d}: {np.std(scatter['y']):.2f}")
        x = scatter['x']
        y = scatter['y']
        x_lims = (min(x_lims[0], np.min(x)), max(x_lims[1], np.max(scatter['x'])), x_lims[2])
        y_lims = (min(y_lims[0], np.min(y)), max(y_lims[1], np.max(scatter['y'])), y_lims[2])

        plt.scatter(x, y, label=f"{kind}(d={elbow_d:.2f})", marker=marker, s=markersize, color=color)
    plt.legend()
    plt.xlabel("#units in the corpus")
    plt.xticks(np.arange(x_lims[0], x_lims[1] + 1, x_lims[2]))
    plt.ylabel("#vocabulary labels (d=1)")
    plt.yticks(np.arange(y_lims[0], y_lims[1] + 1, y_lims[2]))
    plt.title("Trends wrt Size of the Corpus")
    plt.savefig(output_path)
    plt.close()