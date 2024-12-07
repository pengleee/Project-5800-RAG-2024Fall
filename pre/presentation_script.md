### Page 1-6

[1 min] Good afternoon, everyone. Today, we'll be presenting our project on the End-to-End Retrieval-Augmented Generation (RAG) System. This project was developed by Yuting Fan, Peng Li, and Yiwei Qi for the DSAN 5800 course. Our goal is to create a scalable, efficient, and user-friendly system that integrates document retrieval with AI-generated responses. I’ll start by introducing the project and system overview, while my teammates will walk you through implementation, evaluation, and results.

[1 min] So, what is a RAG system? It stands for Retrieval-Augmented Generation. Unlike standard AI models that generate answers from internal knowledge, a RAG system retrieves relevant context from a document database and uses that context to produce accurate, fact-based responses. This makes it highly reliable and reduces hallucinations often seen in AI models. Our system supports dynamic document uploads, retrieval via FAISS, and generation using OpenAI's GPT models.

[1 min] Our main goal was to develop a system that automates the process of query handling, document retrieval, and AI-based response generation. By using FAISS for retrieval and OpenAI's GPT for language generation, we achieved a fast, flexible, and transparent system. Additionally, we ensured that users can upload their own documents to expand the system’s knowledge base.

[1 min] Here’s the overall system architecture. It consists of three main modules: document retrieval, language generation, and user interface.

[1 min] The process starts with a user query, which is converted into an embedding. The FAISS retriever searches for related documents, and those are combined with the query to generate a response using GPT. The answer is presented to the user in a simple interface built with Streamlit.


### Page 7-11

[1 min] Now let’s look at the system's core technical implementation. For retrieval, we use FAISS, a vector search tool, combined with Hugging Face's SentenceTransformers to create semantic embeddings.

[1 min] For generation, we use OpenAI's GPT models, with users having the option to select GPT-3.5 or GPT-4. The user input is combined with document content to generate accurate and relevant responses.

[1 min] We’ve also made the system easy to use. The user interface is built with Streamlit, allowing users to upload custom documents and select which GPT model they want to use. This allows for real-time responses and makes it easy to expand the knowledge base on demand.
[*Give a demo about how to use the system*]

[2 min] Some key features of our system include real-time document uploads, support for multiple GPT models, and transparency. The system displays which document it retrieved information from, providing credibility and context for users.


### Page 12-17

[2 min] To ensure system reliability, we tested file support, query handling, and performance. It can handle most file types like PDF and plain text. It performs well with large files up to 100MB, but files over 1GB require optimization. The system also handles garbled files and malformed queries with clear user feedback.

[1 min] We compared GPT-3.5 and GPT-4. Unsurprisingly, GPT-4 performs better, especially with complex queries. It produces more accurate, relevant, and logically coherent responses.

[2 min] In summary, we developed a functional and scalable RAG system. In the future, we hope to expand to multi-modal data and fine-tune GPT models for specialized fields like healthcare.