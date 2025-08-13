import numpy
import numpy as np


# import embedding model

# Embedding matrix

def get_embedding(word: str, embedding_model) -> np.ndarray:
    """
    Retrieve the embedding vector for a given word.

    Args:
        word (str): Word to get embedding vector for.
        embedding_model (torch.nn.Module): Embedding model to use.

    Returns:
        np.ndarray: Embedding vector.
    """
    if word in embedding_model:
        return embedding_model[word]
    else:
        return numpy.zeros(embedding_model.vecotor_size)

def tokenize_and_embed(word: str, embedding_model) -> list:
    """
    Tokenize the input sentence and obtain embedding for each token

    Args:
        word (str): Input sentence.
        embedding_model: Pre-trained embedding model.

    Returns:
        list: List of embedding vectors for each token.
    """
    tokens = word.split()
    embeddings = numpy.array([get_embedding(word, embedding_model) for word in tokens])
    return embeddings




