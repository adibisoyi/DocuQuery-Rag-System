class Chunker:
    """
    Splits documents into chunks for embedding.
    """

    def __init__(self, chunk_size=600, overlap=100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document_text):
        raise NotImplementedError("Chunking not implemented yet.")