from dotenv import dotenv_values

config = dotenv_values(".env")
HF_TOKEN = config.get("HUGGINGFACE_HUB_TOKEN")
from retriever import Retriever


def main():
    retriever_keyword = Retriever(top_k=5, mode="keyword")
    retriever_keyword.score("Can machines think?")

    # retriever_keyword.writeWeightedMatrixCSV()

    retriever_semantic = Retriever(top_k=5, mode="semantic")
    retriever_semantic.score("Can machines think?")


if __name__ == "__main__":
    main()
