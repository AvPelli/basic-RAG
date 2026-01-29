from dotenv import dotenv_values

config = dotenv_values(".env")
HF_TOKEN = config.get("HUGGINGFACE_HUB_TOKEN")
from retriever import Retriever


def main():
    retriever = Retriever(top_k=5, mode="keyword")

    retriever.score("Can machines think?")

    # weightedMatrix = retriever.getWeightedMatrix()
    # retriever.writeWeightedMatrixCSV()


if __name__ == "__main__":
    main()
