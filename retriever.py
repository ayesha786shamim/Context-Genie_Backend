from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class VectorStore:
    def __init__(self, chunks):
        """
        Initializes the retriever using:
        - a sentence transformer model to embed text
        - a FAISS index to enable fast similarity search
        """
        self.model = SentenceTransformer("all-MiniLM-L6-v2")  # Embedding model
        self.index = faiss.IndexFlatL2(384)  # FAISS index for 384-dim vectors
        self.chunks = chunks  # Store the original chunks for later reference

        # Generate vector embeddings for each chunk's text
        self.embeddings = self.model.encode([chunk['text'] for chunk in chunks])

        # Add these embeddings to the FAISS index for similarity search
        self.index.add(np.array(self.embeddings).astype('float32'))

    def search(self, query, top_k=3):
        """
        Given a user's query, returns the top_k most relevant text chunks.
        """
        # Convert the query into a vector
        query_embedding = self.model.encode([query])

        # Search for similar vectors in the FAISS index
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), top_k)

        # Return the matching text chunks
        return [self.chunks[i] for i in indices[0]]
