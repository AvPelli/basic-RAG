import os
from retriever import Retriever


def main():
    retriever = Retriever()

    weightedMatrix = retriever.getWeightedMatrix()
    print(weightedMatrix)

    scores = retriever.file_scoring("Can machines think?")
    print("Scores for prompt:\n")
    print("\n".join(f"{k}: {v}" for k, v in scores.most_common()))


if __name__ == "__main__":
    main()
