import os, re

from collections import Counter
import pandas as pd
import numpy as np
import fitz
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

from dotenv import dotenv_values

config = dotenv_values(".env")
HF_TOKEN = config.get("HUGGINGFACE_HUB_TOKEN")


class Retriever:
    """
    A retriever based on TF-IDF algorithm, scores documents by matching prompt words.
    Uses knowledge directory to build term matrix.
    """

    def __init__(self, knowledge_path: str = None):
        """
        Initializing Retriever with knowledge directory path.
        Defaults to './knowledgebase' if no path provided.
        """

        self.knowledge_path = knowledge_path or os.path.join(
            os.getcwd(), "knowledgebase"
        )
        self.document_matrix = self._getDocumentMatrix()
        self.weighted_matrix = self.getWeightedMatrix()

    ### KEYWORD-SEARCH SECTION:

    def _countWords(self, file_path: str) -> Counter:
        """Count word frequencies in a file"""
        if file_path.endswith(".txt"):
            with open(file_path, "r") as file:
                content = file.read()
        elif file_path.endswith(".pdf"):
            doc = fitz.open(file_path)
            content = ""
            for page in doc:
                content += page.get_text()
            doc.close()

        # Alphanumerical words only
        words = re.findall(r"\b[a-zA-Z0-9]+\b", content)
        return Counter(words)

    def _getDocumentMatrix(self) -> pd.DataFrame:
        """Build TF matrix by scanning all documents under knowledgePath"""
        file_paths = []
        for path, dirs, files in os.walk(self.knowledge_path):
            for file in files:
                if file.endswith((".txt", ".pdf")):
                    file_paths.append(os.path.join(path, file))

        print(f"Found {len(file_paths)} files")

        counters = [self._countWords(fp) for fp in file_paths]

        matrix = pd.DataFrame(counters, index=file_paths).fillna(0)
        return matrix.T

    def _inverse_document_frequency(self) -> pd.Series:
        """Calculate IDF: log(1 + total_docs / doc_freq) for each term"""
        total_docs = len(self.document_matrix.columns)
        doc_frequencies = (self.document_matrix > 0).sum(axis=1)
        return np.log1p(total_docs / doc_frequencies)

    def getWeightedMatrix(self):
        """multiplying TF matrix with IDF array, resulting in TF-IDF matrix"""
        return self.document_matrix.mul(self._inverse_document_frequency(), axis=0)

    def writeWeightedMatrixCSV(self):
        self.document_matrix.to_csv("weighted_matrix.csv")

    def file_scoring(self, prompt: str):
        """
        Score files based on TF-IDF weights of prompt words.
        Returns Counter with file paths as keys and relevance scores as values
        """
        prompt_words = re.findall(r"\b[a-zA-Z0-9]+\b", prompt)
        scores = Counter()

        for word in prompt_words:
            if word in self.weighted_matrix.index:
                row = self.weighted_matrix.loc[word]
                scores.update(dict(zip(row.index, row.values)))

        return scores

    ### SEMANTIC-SEARCH SECTION:
    @staticmethod
    def get_document_chunks(text: str, chunk_size=300, overlap=50):
        """Split document into overlapping chunks, preserving sentences"""
        paragraphs = text.split("\n\n")
        chunks = []

        current_chunk = ""

        # Add chunks with max size = chunk_size
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk += " " + para if current_chunk else para

        if current_chunk:
            chunks.append(current_chunk.strip())

        # Add overlap between chunks
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            overlapped_chunks.append(chunk)
            if i < len(chunks) - 1:
                # Add overlapping chunk (last N words of current + first N of next)
                overlap_text = chunks[i][-overlap:] + " " + chunks[i + 1][:overlap]
                overlapped_chunks.append(overlap_text)

        return overlapped_chunks

    def preprocess_vectordb(self):
        """Building Chroma vector database from all documents in knowledgebase"""
        file_paths = []
        for path, dirs, files in os.walk(self.knowledge_path):
            for file in files:
                if file.endswith((".txt", ".pdf")):
                    file_paths.append(os.path.join(path, file))

        print(f"Found {len(file_paths)} files")
        print(f"Start building vector database")

        model = SentenceTransformer("BAAI/bge-base-en-v1.5", token=HF_TOKEN)

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
            embeddings = model.encode(chunks).tolist()

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

    def semantic_scoring(self, prompt: str, top_k: int):
        """Scoring prompt by using preprocessed vector database"""
        model = SentenceTransformer("BAAI/bge-base-en-v1.5", token=HF_TOKEN)
        prompt_emb = model.encode(prompt)

        client = chromadb.PersistentClient(path="./rag_db")
        collection = client.get_or_create_collection(
            name="knowledgebase", embedding_function=None
        )

        results = collection.query(
            query_embeddings=[prompt_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

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

    ### To do: file-checker that sees if the file has been processed already
