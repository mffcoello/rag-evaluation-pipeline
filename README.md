## What it does

A RAG pipeline that actually evaluates itself: checks if answers are grounded in the retrieved context, if they answer the question, and how much of what got retrieved was even relevant.

## Why

I built a RAG chatbot before (Miami Book Fair) This is me evaluating the same mechanism

## How it works

- Chunk docs, embed with `all-MiniLM-L6-v2`, store in ChromaDB.
- Generate answers with `google/flan-t5-base` (free, local, no API key).
- Judge each answer three ways: faithfulness (grounded or made up?), relevancy (does it answer the question?), context precision@k (was the retrieved stuff actually relevant?).
- Built precision@k two ways, keyword matching and LLM judgment. They disagreed on "What is chunking?" (0.67 vs 0.33), which is basically the whole argument for why keyword matching isn't enough.
- Judge only answers yes/no, not a 1-5 score. Small free models aren't reliable at scoring, they're okay at yes/no.

## How to run it

pip install langchain langchain-community chromadb sentence-transformers tiktoken transformers

Run rag_evaluation.ipynb top to bottom. Call evaluate_rag("question") for one question, or run the batch loop for aggregate numbers.

What I learned

- Low relevancy wasn't a generator problem, it traced back to low context precision instead.
- Keyword matching counts a chunk as relevant just for containing a word, doesn't check if it's actually about the topic.
- Small models judge yes/no fine, fall apart on scored ratings.
- Hit two real bugs: a Colab torch/transformers conflict from installing mid-session, and pipeline("text2text-generation") got removed, switched to AutoModelForSeq2SeqLM directly.
