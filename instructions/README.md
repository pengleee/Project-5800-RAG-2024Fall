# Final Project Assignment

*ANLY-5800 (Fall '24)*

## Overview

The final project for this course will be a group-based assignment where students can choose between predefined project topics or propose a project of their own. This project is designed to allow students to apply the concepts learned throughout the course, exploring advanced NLP techniques with hands-on implementations. Each group will choose one of the predefined project options or submit a proposal for and execute on a custom project idea (details below).


### Project Guidelines

#### **Group Size**
 
1-4 students

#### **Deliverables**
- A well-documented GitHub repository containing your code
- A project report (4-6 pages) outlining your methodology, experiments, and results
- A 15 minute final presentation / demo showcasing your project, followed by a 5 minute Q&A session

#### **Timeline**
| **Milestone**                     | **Due Date**   |
|-----------------------------------|----------------|
| **Group Selection Due**           | Oct 29         |
| **Project Proposal Due**          | Nov 05         |
| **Milestone Checkpoint**          | Nov 19         |
| **Final Presentation & Due Date** | Dec 10         |

---

## Jetstream2 Compute Resources

Each group will have access to free compute resources via a Jupyter Lab notebook hosted on the Jetstream2 super computing cluster. Our cluster contains Nvidia A100 (40GB) GPUs. To access and use this resource, follow these [*instructions*](https://github.com/chrislarson1/GU-ANLY-5800-FALL-2024/blob/main/project/jetstream2-instructions.md).

---

## Project Options

### Option (a): Implement LLM from Scratch

#### Description:
In this project, you will build a simplified version of an open-source large language model, from scratch. The goal is to understand the key components that make up a transformer-based LLM, including tokenization, attention mechanisms, and training on a small dataset. You are expected to implement the architecture from the ground up using a NN library of your choosing, such as Pytorch or Tensorflow. You will very likely need to reduce the parameter count of the model.

#### Key Tasks:
- Implement tokenization using the appropriate tokenizer.
- Build a multi-head self-attention mechanism.
- Implement feedforward networks, layer normalization, and position encoding.
- Train your model on a token prediction task using a small dataset of your choice.
- Evaluate model performance using common NLP tasks and benchmarks.

#### Resources:
- [Llama GitHub repository](https://github.com/meta-llama)


### Option (b): Fine-tune a Large Language Model Using LoRA

#### Description:
Low-Rank Adaptation (LoRA) is a method for efficiently fine-tuning large language models by reducing the number of trainable parameters. In this project, you will fine-tune a pre-trained LLM on a specific NLP task using LoRA.

#### Key Tasks:
- Select a pre-trained LLM suitable for your NLP task (e.g., text classification, question answering).
- Implement LoRA to fine-tune the model with reduced computational resources.
- Train the model on a custom dataset or an existing dataset.
- Compare performance and efficiency with full model fine-tuning.

#### Resources:
- [LoRA paper](https://arxiv.org/abs/2106.09685)

### Option (c): End-to-End Retrieval-Augmented Generation (RAG) System

#### Description:
In this project, you will build a retrieval-augmented generation (RAG) system that can retrieve relevant documents from a knowledge base and use a language model to generate responses based on the retrieved information.

#### Key Tasks:
- Implement a document retriever using dense retrieval (e.g., FAISS).
- Combine the retriever with a language model (e.g., GPT) to generate responses.
- Build a simple web application that satisfies these functional requirements:
  - Chat interface
  - Document upload
  - Document source attribution
- Evaluate retrieval and response generation quality.

#### Resources:
- [HuggingFace RAG Implementation](https://huggingface.co/docs/transformers/model_doc/rag)

### Option (d): LLM Agent with Dynamic Tool Usage and Code Generation

#### Description:
In this project, you will create an LLM agent that can dynamically decide which tools to use based on the query it receives. The agent should be capable of generating and executing code when necessary to answer user queries.

#### Key Tasks:
- Implement an agent that can dynamically select tools (e.g., search engines, calculators, external APIs) to perform complex tasks.
- Add the ability for the agent to generate Python code to execute tasks, such as performing calculations or retrieving data.
- Use an LLM to parse user input and decide on the appropriate tool or generate code on the fly.
- Add the ability to inspect and visualize the agent's steps.
- Evaluate the system on a set of user queries and measure accuracy and effectiveness.

*Note: LLM and tool calling / orchestration must be implemented, not outsourced to a third party API.*

#### Resources:
- [LangChain GitHub repository](https://github.com/langchain-ai/langchain)
- [Llama-stack GitHub repository](https://github.com/meta-llama/llama-stack)
- [ReAct paper](https://arxiv.org/pdf/2210.03629)

### Option (e): Student-Defined Project

Students are welcome (and encouraged) to propose project ideas that align with their own interests, so long as they relate to the topics covered in this course. The proposed project should be ambitious but achievable within the given time frame. Here are some guidelines for the student-defined project:

#### Requirements:
- Must involve a significant component of NLP research or application (e.g., novel models, tasks, or datasets).
- Should incorporate methods learned throughout the course, including but not limited to language models, transformers, tokenization, fine-tuning, or retrieval systems.
- The project proposal must include a clear problem statement, methodology, expected outcomes, and datasets to be used.

---

## Evaluation Criteria

| **Category**                  | **Weight** | **Description**                                                                 |
|-------------------------------|------------|---------------------------------------------------------------------------------|
| **Approach & Implementation** | 40%        | Accuracy and correctness of the implemented model or system.                    |
| **Innovation**                | 20%        | Creativity in problem-solving and implementation.                               |
| **Experimental Results**      | 20%        | Quality of experiments, evaluation, and analysis.                               |
| **Clarity & Documentation**   | 10%        | Quality of the code, documentation, and project report.                         |
| **Presentation**              | 10%        | Clarity and effectiveness of the final presentation.                            |
