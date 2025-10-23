"""
RAG baseline for evaluating information retrieval performance.

- Includes a generic function that (1) can encode the knowledge base (i.e., a corpus of tutorial videos) into a vector database, and (2) encode the query (which is likely the target tutorial + query) into a vector. Uses TF-IDF or Sentence-BERT or OpenAI embeddings in `./helpers/`.

- Includes a function that can perform RAG retrieval on the vector database using the query vector and respond to the query.
"""
import os
import pickle
import numpy as np

from helpers.bert import bert_embedding, tfidf_embedding, mccs
from prompts.rag import get_rag_response_full_tutorial, get_rag_response_tutorial_segment

EMBEDDINGS_PATH = "./static/results/rag/"
DOCUMENT_SCORE_THRESHOLD = 0.7

def perform_embedding(embedding_method, texts):
    """
    Embed the texts using the appropriate embedding method.
    """
    if embedding_method == "tfidf":
        return tfidf_embedding(texts)
    elif embedding_method == "bert":
        return bert_embedding(texts)
    else:
        ### TODO: implement OpenAI embeddings
        raise ValueError(f"Invalid embedding method: {embedding_method}")

def encode_dataset(embedding_method, task, dataset):
    """
    Encode the dataset into a vector database.
    """
    embeddings_path = EMBEDDINGS_PATH + f"{task}_{embedding_method}_dataset_embeddings.pkl"
    if os.path.exists(embeddings_path):
        embeddings = pickle.load(open(embeddings_path, "rb"))
    else:
        texts = [(
            "Tutorial: " + item["content"]
            ) for item in dataset]
        embeddings = perform_embedding(embedding_method, texts)
        pickle.dump(embeddings, open(embeddings_path, "wb"))
    return embeddings

def encode_query(embedding_method, tutorial, query, segment=None):
    """
    Encode the query into a vector.
    """
    if segment is None:
        texts = ["Tutorial: " + tutorial["content"] + "\n" + "Query: " + query]
    else:
        texts = ["Tutorial: " + tutorial["content"] + "\n" + "Segment: " + segment + "\n" + "Query: " + query]
    embeddings = perform_embedding(texts, embedding_method)
    return embeddings

def perform_retrieval(embedding_method, task, dataset, tutorial, query, segment=None, k=10):
    """
    Perform RAG retrieval on the vector database using the query vector and respond to the query.
    """
    dataset_embeddings = encode_dataset(embedding_method, task, dataset)
    query_embeddings = encode_query(embedding_method, tutorial, query, segment)
    document_idxs, scores = mccs(dataset_embeddings, query_embeddings, top_k=k)
    scores = scores.flatten()
    document_idxs = document_idxs.flatten()
    documents = []
    for idx, score in zip(document_idxs, scores):
        documents.append({
            "title": dataset[idx]["title"],
            "content": dataset[idx]["content"],
            "score": score,
        })
    return documents, scores

def respond_to_query_rag(embedding_method, task, dataset, tutorial, query, segment=None, k=10):
    documents, scores = perform_retrieval(embedding_method, task, dataset, tutorial, query, segment, k)

    ### filter out documents with score less than 0.7
    filtered_documents = []
    for document in documents:
        if document["score"] >= DOCUMENT_SCORE_THRESHOLD:
            filtered_documents.append(document)
    if len(filtered_documents) == 0:
        raise ValueError("No documents with score >= 0.7 found.")

    if segment is None:
        return get_rag_response_full_tutorial(task, filtered_documents, tutorial, query)
    else:
        return get_rag_response_tutorial_segment(task, filtered_documents, tutorial, segment, query)