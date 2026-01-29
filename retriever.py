import os, re
from keyword_searcher import KeywordSearcher
from semantic_searcher import SemanticSearcher


class Retriever:
    """
    Retriever that scans documents (txt,pdf) under a given directory.
    Currently supports: keyword-search, semantic-search
    """

    def __init__(
        self,
        knowledge_path: str = None,
        mode: str = "semantic",
        emb_model: str = "BAAI/bge-base-en-v1.5",
        top_k: int = 5,
    ):

        self.mode = mode
        self.top_k = top_k

        self.knowledge_path = knowledge_path or os.path.join(
            os.getcwd(), "knowledgebase"
        )

        if mode == "semantic":
            self.semantic_search = SemanticSearcher(
                self.knowledge_path, top_k, emb_model
            )
        else:
            self.keyword_search = KeywordSearcher(self.knowledge_path, top_k)

    def score(self, prompt: str):
        if self.mode == "semantic":
            results = self.semantic_search.semantic_scoring(prompt, self.top_k)
            self.semantic_search.print_score(results)
        else:
            results = self.keyword_search.file_scoring(prompt, self.top_k)
            self.keyword_search.print_score(results)
