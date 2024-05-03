from sklearn.decomposition import PCA

def standardize(embeddings):
    std = embeddings.std(axis=0)
    mean = embeddings.mean(axis=0)
    if (std == 0).any():
        return embeddings
    return (embeddings - mean) / std

def reduce_dim(embeddings, n_components=2):
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings