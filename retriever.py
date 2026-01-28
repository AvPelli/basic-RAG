import os, re
from collections import Counter
import pandas as pd
import numpy as np
import fitz


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
