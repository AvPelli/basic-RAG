# basic-RAG
Basic implementation of a RAG system.

The goal of this project is improving my understanding of Retrieval Augmented Generation, by making such a system myself from the ground up. 

While building i made effort to not use any AI assist in coding, as this would hinder the learning process.

Inspiration drawn from a coursera course i'm following at the moment: [DeepLearning.AI - RAG](https://learn.deeplearning.ai/courses/retrieval-augmented-generation)


## Design choices

Here i will keep a list of choices i made during implementation:

* I chose to not do any document processing during creation of Retriever, and only do it when explicitly asked by calling the function.

* I'm building bottom up, starting from building retriever for one file, then scaling it to multiple files.

I try to follow clean code principles: 

* Keeping the code easy to read and understand 

* Keeping documentation short and to the point

* Limit the amount of work done by each function, the name of the function should 