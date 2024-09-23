import nltk

from sklearn.decomposition import PCA

def standardize(embeddings):
    std = embeddings.std(axis=0)
    mean = embeddings.mean(axis=0)
    if (std == 0).any():
        return embeddings
    return (embeddings - mean) / std

def reduce_dim(fit_embeddings, transform_embeddings, n_components=2):
    pca = PCA(n_components=n_components)
    pca.fit(fit_embeddings)
    reduced_embeddings = pca.transform(transform_embeddings)

    return reduced_embeddings, pca

from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import zeros
from nltk.corpus import stopwords

nltk.download('stopwords')

def tfidf_embedding(texts): 
    if len(texts) == 0:
        return []
    vectorizer = TfidfVectorizer(
        stop_words=stopwords.words('english'),
        ngram_range=(1, 2), max_features=1000
    )
    for i in range(len(texts)):
        if texts[i] == "":
            texts[i] = " "

    X = vectorizer.fit_transform(texts)
    embeddings = X.toarray()
    return embeddings

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from helpers.bert import bert_embedding


def k_means_clustering(
        embeddings,
        n_clusters=3,
):
    kmeans = KMeans(n_clusters=n_clusters, n_init=3, random_state=10).fit(embeddings)

    s_score = silhouette_score(embeddings, kmeans.labels_, metric='euclidean')
    
    return kmeans.labels_, kmeans.inertia_, s_score

def cluster_texts(texts):
    """
    Returns labels, inertia & silhouette score for each n_clusters
    """
    if len(texts) <= 1:
        return [0 for _ in range(len(texts))], [], []

    labels = []
    max_s_score = -2
    best_n_clusters = 0
    
    inertias = []
    s_scores = []
    embeddings = bert_embedding(texts)

    for n_clusters in range(2, len(texts)-1):
        labels_, inertia, s_score = k_means_clustering(embeddings, n_clusters=n_clusters)
        
        inertias.append(inertia)
        s_scores.append(s_score)

        if s_score > max_s_score:
            max_s_score = s_score
            labels = labels_
            best_n_clusters = n_clusters

    if len(labels) == 0:
        labels = [i for i in range(len(texts))]
        inertias = [0]
        s_scores = [1]
        best_n_clusters = len(texts)
    
    ### delete later
    print("# Clusters: ", best_n_clusters)
    print("Inertias: ", inertias)
    print("S Scores: ", s_scores)

    return labels, inertias, s_scores