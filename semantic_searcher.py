import os, fitz, chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from chromadb.api.types import QueryResult
from sentence_transformers import SentenceTransformer


class SemanticSearcher:
    """
    Semantic search using Chroma as vector database.
    """

    def __init__(
        self, knowledge_path: str, top_k: int, emb_model: str = "BAAI/bge-base-en-v1.5"
    ):
        self.knowledge_path = knowledge_path
        self.top_k = top_k
        self.emb_model = SentenceTransformer(emb_model)

        # Create vectordb on initialization
        ### To do: file-checker that sees if the file has been vectorized already
        self.preprocess_vectordb()

    @staticmethod
    def get_document_chunks(text: str, chunk_size=300, overlap=50):
        """Split document into overlapping chunks, preserving sentences"""
        paragraphs = text.split("\n\n")
        words = []

        # Add chunks with max size = chunk_size
        for para in paragraphs:
            para = para.strip()
            if para:
                words.extend(para.split())

        chunks = []
        i = 0

        while i < len(words):
            chunk_words = words[i : i + chunk_size]
            chunks.append(" ".join(chunk_words))
            i += chunk_size - overlap

        return chunks

    def preprocess_vectordb(self):
        """Building Chroma vector database from all documents in knowledgebase"""
        file_paths = []
        for path, dirs, files in os.walk(self.knowledge_path):
            for file in files:
                if file.endswith((".txt", ".pdf")):
                    file_paths.append(os.path.join(path, file))

        print(f"Found {len(file_paths)} files")
        print(f"Start building vector database")

        all_chunks = []
        all_embeddings = []
        all_ids = []
        all_metadata = []

        for file_idx, file_path in enumerate(file_paths):
            if file_path.endswith(".txt"):
                with open(file_path, "r") as file:
                    content = file.read()
            elif file_path.endswith(".pdf"):
                doc = fitz.open(file_path)
                content = ""
                for page in doc:
                    content += page.get_text()
                doc.close()

            print("Get chunks for document: " + file_path)
            chunks = self.get_document_chunks(content)
            embeddings = self.emb_model.encode(chunks).tolist()

            for chunk_idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                all_chunks.append(chunk)
                all_embeddings.append(emb)
                all_ids.append(f"{file_path}_chunk_{chunk_idx}")
                all_metadata.append(
                    {"source_doc": file_path, "chunk_id": chunk_idx, "doc_id": file_idx}
                )

        print("Chromadb: making client")
        client = chromadb.PersistentClient(path="./rag_db")

        print("Chromadb: making collection")
        collection = client.get_or_create_collection(
            name="knowledgebase", embedding_function=None
        )

        print("Adding all chunks to vector database")
        collection.add(
            documents=all_chunks,
            embeddings=all_embeddings,
            ids=all_ids,
            metadatas=all_metadata,
        )

    def semantic_scoring(self, prompt: str, top_k: int) -> QueryResult:
        """Scoring prompt by using preprocessed vector database"""
        prompt_emb = self.emb_model.encode(prompt)

        client = chromadb.PersistentClient(path="./rag_db")
        collection = client.get_or_create_collection(
            name="knowledgebase", embedding_function=None
        )

        results = collection.query(
            query_embeddings=[prompt_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        return results

    def print_score(self, results: QueryResult):
        print(f"Found {len(results['documents'][0])} matches")

        for i, (doc, meta, dist) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ):
            print(f"Rank {i+1} (dist: {dist:.3f}): {doc[:100]}...")
            print(f"  Source: {meta['source_doc']}")
