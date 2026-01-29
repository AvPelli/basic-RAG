import os
from retriever import Retriever


def main():
    retriever = Retriever()

    # weightedMatrix = retriever.getWeightedMatrix()
    # retriever.writeWeightedMatrixCSV()

    # prompt = "Can machines think?"
    # scores = retriever.file_scoring(prompt)

    # print(f"Scores for prompt: {prompt}\n")
    # print("\n".join(f"{k}: {v}" for k, v in scores.most_common()))

    # retriever.preprocess_vectordb()

    retriever.semantic_scoring("can machines think?", 5)


if __name__ == "__main__":
    main()
