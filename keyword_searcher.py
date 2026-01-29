import os, re, fitz
from collections import Counter
import pandas as pd
import numpy as np


class KeywordSearcher:
    """
    TF-IDF algorithm implemented using pandas and numpy
    """

    def __init__(self, knowledge_path: str, top_k: int):
        self.knowledge_path = knowledge_path
        self.top_k = top_k
        self.document_matrix = self._getDocumentMatrix()
        self.weighted_matrix = self._getWeightedMatrix()

    @staticmethod
    def _readFile(file_path: str) -> str:
        if file_path.endswith(".txt"):
            with open(file_path, "r") as file:
                content = file.read()
        elif file_path.endswith(".pdf"):
            doc = fitz.open(file_path)
            content = ""
            for page in doc:
                content += page.get_text()
            doc.close()
        return content

    def _countWords(self, file_path: str) -> Counter:
        """Count word frequencies in a file"""
        content = self._readFile(file_path)
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

    def _getWeightedMatrix(self):
        """multiplying TF matrix with IDF array, resulting in TF-IDF matrix"""
        return self.document_matrix.mul(self._inverse_document_frequency(), axis=0)

    def writeWeightedMatrixCSV(self):
        self.document_matrix.to_csv("weighted_matrix.csv")

    def file_scoring(self, prompt: str, top_k: int) -> Counter:
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

    def print_score(self, scores: Counter):
        print("Scores :\n")
        print("\n".join(f"{k}: {v}" for k, v in scores.most_common(self.top_k)))
