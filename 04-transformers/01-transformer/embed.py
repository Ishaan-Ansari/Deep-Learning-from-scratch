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

def add_positional_encoding(embeddings: numpy.ndarray) -> numpy.ndarray:
    """
    Add positional encoding to the input embedding.

    Args:
        embeddings (numpy.ndarray): Input embedding.

    Returns:
        numpy.ndarray: Embeddings with added positional encoding.
    """

    sequence_len = embeddings.shape[0]
    embedding_dim = embeddings.shape[1]

    # initialize positional encoding matrix
    pos_enc_matrix = np.zeros((sequence_len, embedding_dim))

    # Calculate the positional encodings
    for pos in range(sequence_len):
        for i in range(embedding_dim):
            if i % 2 == 0:
                pos_enc_matrix[pos, i] = numpy.sin(pos / (10000 ** (2 * i / embedding_dim)))
            else:
                pos_enc_matrix[pos, i] = numpy.cos(pos / (10000 ** (2 * i / embedding_dim)))

    # Add positional encodings
    embedding_matrix = embeddings + pos_enc_matrix

    return embedding_matrix
