import os
from collections import Counter
import pandas as pd

class Retriever:

    def __init__(self, path):
        """Retriever object has a path to a file"""

        self.documentpaths = path
        self.documentMatrix = self._getDocumentMatrix(path)


    def _countWords(self, path):
        """Return Counter containing wordcounts"""
        #Read file as 1 string:
        with open(path, 'r') as file:
            content = file.read()

        #Count words:
        counter = Counter()
        words = content.split(sep=None)
        for w in words:
            counter[w] += 1
        return counter

    def _getDocumentMatrix(self, path):
        wordcount = self._countWords(path)
        return pd.DataFrame.from_dict(wordcount, orient='index', columns=['count'])
    