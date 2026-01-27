import os
from retriever import Retriever


def main():
    current_dir = os.getcwd()
    print("Current directory: " + current_dir)

    retriever = Retriever(current_dir)
    print(retriever.documentMatrix.head(5))


if __name__ == "__main__":
    main()
