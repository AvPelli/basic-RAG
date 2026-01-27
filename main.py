import os
from retriever import Retriever


def main():
    current_dir = os.getcwd()
    print("Current directory: " + current_dir)

    retriever = Retriever(current_dir)

    IDF = retriever.inverseDocumentFrequency()
    print(IDF)

    weightedMatrix = retriever.getWeightedMatrix()
    print(weightedMatrix)


if __name__ == "__main__":
    main()
