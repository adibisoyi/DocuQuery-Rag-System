class VectorStore:
    """
    Handles FAISS index creation and querying.
    """

    def __init__(self):
        pass

    def add_embeddings(self, embeddings):
        raise NotImplementedError()

    def search(self, query_embedding, top_k=5):
        raise NotImplementedError()