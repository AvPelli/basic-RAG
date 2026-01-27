import os
from retriever import Retriever


def main():
    current_dir = os.getcwd()
    print("Current directory: " + current_dir)

    retriever = Retriever(current_dir)

    weightedMatrix = retriever.getWeightedMatrix()
    print(weightedMatrix)

    scores = retriever.file_scoring("find me a test")
    print("Scores for prompt:\n")
    print("\n".join(f"{k}: {v}" for k, v in scores.most_common()))


if __name__ == "__main__":
    main()
