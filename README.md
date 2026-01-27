# basic-RAG
Basic implementation of a RAG system.

The goal of this project is to improve my understanding of Retrieval Augmented Generation, by building such a system from the ground up. 

Inspiration drawn from a coursera course i'm following at the moment: [DeepLearning.AI - RAG](https://learn.deeplearning.ai/courses/retrieval-augmented-generation)


## Design choices

Here i will keep a list of choices i made during implementation:

* I'm building bottom up, starting from building retriever for one file, then scaling it to multiple files.

* I'm using the TF-IDF algorithm to score documents, this is a foundational algorithm on which BM25 is an improvement. BM25 is widely used in RAG systems and builds further upon TF-IDF by normalizing for document-length, Term Frequency (TF) and provides tunable parameters for these.

I try to follow clean code principles: 

* Keeping the code easy to read and understand 

* Keeping documentation short and to the point

* Limit the amount of work done by each function, the name of the function should contain enough information to know what happens inside the function