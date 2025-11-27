from pydantic_models.evaluation import MetricScale
from typing import Optional

from helpers.dataset import CROSS_TASK_TASKS, BIG_CUSTOM_TASKS, CUSTOM_TASKS

from src.rag import generic_call_rag
from src.cim_methods import generic_call_context_similarity, generic_call_shortest_path

from eval.llm_judge import relevance_absolute_evaluation, relevance_comparative_evaluation

class DatasetConfig():
    label: str
    tasks: list[str]
    sample_per_task: int
    query: int
    n: int

    def __init__(self, label, tasks, sample_per_task, query, n):
        self.label = label
        self.tasks = tasks
        self.sample_per_task = sample_per_task
        self.query = query
        self.n = n
    
    def __str__(self):
        return f"DatasetConfig(label={self.label}, tasks={self.tasks}, sample_per_task={self.sample_per_task}, query={self.query}, n={self.n})"
    
    def to_dict(self):
        return {
            "label": self.label,
            "tasks": self.tasks,
            "sample_per_task": self.sample_per_task,
            "query": self.query,
            "n": self.n,
        }

    def from_dict(obj):
        return DatasetConfig(
            label=obj["label"],
            tasks=obj["tasks"],
            sample_per_task=obj["sample_per_task"],
            query=obj["query"],
            n=obj["n"],
        )

class MethodConfig():
    label: str
    embedding_method: str
    k: int
    doc_score_threshold: float
    func: callable
    version: str
    gen_model: Optional[str]
    
    def __init__(self, label, embedding_method, k, doc_score_threshold, func, version, gen_model):
        self.label = label
        self.embedding_method = embedding_method
        self.k = k
        self.doc_score_threshold = doc_score_threshold
        self.func = func
        self.version = version
        self.gen_model = gen_model

    def __str__(self):
        return f"MethodConfig(label={self.label}, embedding_method={self.embedding_method}, k={self.k}, doc_score_threshold={self.doc_score_threshold}, func={self.func.__name__}, version={self.version}, gen_model={self.gen_model})"

    def to_dict(self):
        return {
            "label": self.label,
            "embedding_method": self.embedding_method,
            "k": self.k,
            "doc_score_threshold": self.doc_score_threshold,
            "func": "function",
            "version": self.version,
            "gen_model": self.gen_model,
        }


class EvalConfig():
    label: str
    func: callable
    metric: MetricScale
    joint: bool
    judge_model: Optional[str]
    
    def __init__(self, label, func, metric, joint, judge_model):
        self.label = label
        self.func = func
        self.metric = metric
        self.joint = joint
        self.judge_model = judge_model
    
    def __str__(self):
        return f"EvalConfig(label={self.label}, func={self.func.__name__}, metric={self.metric}, joint={self.joint}, judge_model={self.judge_model})"

    def to_dict(self):
        return {
            "label": self.label,
            "func": "function",
            "metric": self.metric,
            "joint": self.joint,
            "judge_model": self.judge_model,
        }


TECH_EVAL_PATH = "./static/results/tech_eval/"

QUERIES = {
    "full-top-n": "Given a tutorial, retrieve top-{n} missing, but relevant `{info_type}` information for the entire tutorial.\nFollowing is the definition of `{info_type}` information: \n{info_type_description}",
    "full-all": "Given a tutorial, retrieve all missing, but relevant `{info_type}` information.\nFollowing is the definition of `{info_type}` information: \n{info_type_description}",
    "segment-top-n": "Given a tutorial and a highlighted segment, retrieve top-{n} missing, but relevant `{info_type}` information for the highlighted segment.\nFollowing is the definition of `{info_type}` information: \n{info_type_description}",
    "segment-all": "Given a tutorial and a highlighted segment, retrieve all missing, but relevant `{info_type}` information for the highlighted segment.\nFollowing is the definition of `{info_type}` information: \n{info_type_description}",
}

### naming convention: {dataset_name}_{query_key}_{sample_per_task}_n{n}
DATASETS = {
    "test_full-top-n_10_n5": DatasetConfig(
        label="test_full-top-n_10_n5",
        tasks=[BIG_CUSTOM_TASKS[0]],
        sample_per_task=10,
        query=QUERIES["full-top-n"],
        n=5,
    ),
    "test_segment-top-n_10_n5": DatasetConfig(
        label="test_segment-top-n_10_n5",
        tasks=[CROSS_TASK_TASKS[0]],
        sample_per_task=10,
        query=QUERIES["segment-top-n"],
        n=5,
    ),
    "custom_full-top-n_50_n5": DatasetConfig(
        label="custom_full-top-n_50_n5",
        tasks=BIG_CUSTOM_TASKS,
        sample_per_task=50,
        query=QUERIES["full-top-n"],
        n=5,
    ),
    "cross_full-top-n_10_n5": DatasetConfig(
        label="cross_full-top-n_10_n5",
        tasks=CROSS_TASK_TASKS,
        sample_per_task=50,
        query=QUERIES["full-top-n"],
        n=5,
    ),
    "cross_segment-top-n_10_n5": DatasetConfig(
        label="cross_segment-top-n_10_n5",
        tasks=CROSS_TASK_TASKS,
        sample_per_task=50,
        query=QUERIES["segment-top-n"],
        n=5,
    ),
}

### naming convention: {method_name}_{embedding_method}_v{version?}_k{k?}_{gen_model?}
METHODS = {
    ### RAG GPT-4.1-mini
    "vanilla_gpt-4-1-mini": MethodConfig(
        label="vanilla_gpt-4-1-mini",
        embedding_method="openai",
        k=None,
        doc_score_threshold=None,
        func=generic_call_rag,
        version=None,
        gen_model="gpt-4.1-mini-2025-04-14",
    ),
    "RAG_openai_k2_gpt-4-1-mini": MethodConfig(
        label="RAG_openai_k2_gpt-4-1-mini",
        embedding_method="openai",
        k=2,
        doc_score_threshold=None,
        func=generic_call_rag,
        version=None,
        gen_model="gpt-4.1-mini-2025-04-14",
    ),
    "RAG_openai_k4_gpt-4-1-mini": MethodConfig(
        label="RAG_openai_k4_gpt-4-1-mini",
        embedding_method="openai",
        k=4,
        doc_score_threshold=None,
        func=generic_call_rag,
        version=None,
        gen_model="gpt-4.1-mini-2025-04-14",
    ),
    "RAG_openai_k8_gpt-4-1-mini": MethodConfig(
        label="RAG_openai_k8_gpt-4-1-mini",
        embedding_method="openai",
        k=8,
        doc_score_threshold=None,
        func=generic_call_rag,
        version=None,
        gen_model="gpt-4.1-mini-2025-04-14",
    ),
    "RAG_openai_k16_gpt-4-1-mini": MethodConfig(
        label="RAG_openai_k16_gpt-4-1-mini",
        embedding_method="openai",
        k=16,
        doc_score_threshold=None,
        func=generic_call_rag,
        version=None,
        gen_model="gpt-4.1-mini-2025-04-14",
    ),
    "RAG_openai_k32_gpt-4-1-mini": MethodConfig(
        label="RAG_openai_k32_gpt-4-1-mini",
        embedding_method="openai",
        k=32,
        doc_score_threshold=None,
        func=generic_call_rag,
        version=None,
        gen_model="gpt-4.1-mini-2025-04-14",
    ),
    
    ### OURS
    "ours_similarity_openai_vlatest": MethodConfig(
        label="ours_similarity_openai_vlatest",
        embedding_method="openai",
        k=None,
        doc_score_threshold=None,
        func=generic_call_context_similarity,
        version=None,
        gen_model=None,
    ),
    "ours_distance_openai_vlatest": MethodConfig(
        label="ours_distance_openai_vlatest",
        embedding_method="openai",
        k=None,
        doc_score_threshold=None,
        func=generic_call_shortest_path,
        version=None,
        gen_model=None,
    ),

    ### RAG GPT-5-mini
    "vanilla_gpt-5-mini": MethodConfig(
        label="vanilla_gpt-5-mini",
        embedding_method="openai",
        k=None,
        doc_score_threshold=None,
        func=generic_call_rag,
        version=None,
        gen_model="gpt-5-mini-2025-08-07",
    ),
    "RAG_openai_k2_gpt-5-mini": MethodConfig(
        label="RAG_openai_k2_gpt-5-mini",
        embedding_method="openai",
        k=2,
        doc_score_threshold=None,
        func=generic_call_rag,
        version=None,
        gen_model="gpt-5-mini-2025-08-07",
    ),
    "RAG_openai_k4_gpt-5-mini": MethodConfig(
        label="RAG_openai_k4_gpt-5-mini",
        embedding_method="openai",
        k=4,
        doc_score_threshold=None,
        func=generic_call_rag,
        version=None,
        gen_model="gpt-5-mini-2025-08-07",
    ),
    "RAG_openai_k8_gpt-5-mini": MethodConfig(
        label="RAG_openai_k8_gpt-5-mini",
        embedding_method="openai",
        k=8,
        doc_score_threshold=None,
        func=generic_call_rag,
        version=None,
        gen_model="gpt-5-mini-2025-08-07",
    ),
    "RAG_openai_k16_gpt-5-mini": MethodConfig(
        label="RAG_openai_k16_gpt-5-mini",
        embedding_method="openai",
        k=16,
        doc_score_threshold=None,
        func=generic_call_rag,
        version=None,
        gen_model="gpt-5-mini-2025-08-07",
    ),
    "RAG_openai_k32_gpt-5-mini": MethodConfig(
        label="RAG_openai_k32_gpt-5-mini",
        embedding_method="openai",
        k=32,
        doc_score_threshold=None,
        func=generic_call_rag,
        version=None,
        gen_model="gpt-5-mini-2025-08-07",
    ),
}

### naming convention: {criteria}_{joint/apiece}_{metric type}_{judge_model}_{scale}
EVALS = {
    "relevance_apiece_absolute_gpt-5-mini_binary": EvalConfig(
        label="relevance_apiece_absolute_gpt-5-mini_binary",
        func=relevance_absolute_evaluation,
        metric=MetricScale.BINARY,
        joint=False,
        judge_model="gpt-5-mini-2025-08-07",
    ),
    "relevance_apiece_absolute_gpt-5-mini_3": EvalConfig(
        label="relevance_apiece_absolute_gpt-5-mini_3",
        func=relevance_absolute_evaluation,
        metric=MetricScale.LIKERT_3,
        joint=False,
        judge_model="gpt-5-mini-2025-08-07",
    ),
    "relevance_apiece_absolute_gpt-5-mini_5": EvalConfig(
        label="relevance_apiece_absolute_gpt-5-mini_5",
        func=relevance_absolute_evaluation,
        metric=MetricScale.LIKERT_5,
        joint=False,
        judge_model="gpt-5-mini-2025-08-07",
    ),
    "relevance_joint_absolute_gpt-5-mini_5": EvalConfig(
        label="relevance_joint_absolute_gpt-5-mini_5",
        func=relevance_absolute_evaluation,
        metric=MetricScale.LIKERT_5,
        joint=True,
        judge_model="gpt-5-mini-2025-08-07",
    ),
    "relevance_joint_comparative_gpt-5-mini_comparison": EvalConfig(
        label="relevance_joint_comparative_gpt-5-mini_comparison",
        func=relevance_comparative_evaluation,
        metric=MetricScale.COMPARISON,
        joint=True,
        judge_model="gpt-5-mini-2025-08-07",
    ),
}