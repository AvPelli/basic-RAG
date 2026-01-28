# basic-RAG
Basic implementation of a RAG system.

The goal of this project is to improve my understanding of Retrieval Augmented Generation, by building such a system from the ground up. 

Inspiration drawn from a coursera course i'm following at the moment: [DeepLearning.AI - RAG](https://learn.deeplearning.ai/courses/retrieval-augmented-generation)

## Roadmap

Basic RAG:

- [x] Implement keyword-search - _completed 27/01_
- [ ] Implement semantic-search 
- [ ] Implement LLM
- [ ] Test complete RAG system 


Production RAG:

- [ ] Study state of the art retrieval algorithms / libraries
- [ ] Study opensource LLM variations
- [ ] Test state of the art RAG system
- [ ] Compare with self-implemented basic RAG and reflect on the results

Modifications and tuning:

- [ ] Study pre-training of opensource LLM's
- [ ] Apply custom knowledgebase as training data
- [ ] Compare custom trained with untrained general LLM

## Design choices

Here I will keep a log of choices i'm making and thoughts i'm having during this project. To keep it structured i'm grouping them in subtitles following the roadmap (see above).

### General

In general, I try to follow clean code principles: 

* Keeping the code easy to read and understand 

* Keeping documentation short and to the point

* Limit the amount of work done by each function, the name of the function should contain enough information to know what happens inside the function.

### Basic RAG - implement keyword-search:

* I'm building bottom up, starting from building retriever for one file, then scaling it to multiple files. 

* I found that keeping changes small and intermittent testing made development easier as I got lost a couple times when making too many changes at once.

* I'm using the TF-IDF algorithm to score documents, this is a foundational algorithm on which BM25 is an improvement. BM25 is widely used in RAG systems and builds further upon TF-IDF by normalizing for document-length, Term Frequency (TF) and provides tunable parameters for these.

* Currently only supporting .txt and .pdf files in the knowledgebase