# Multi-Agent PDF Question Answering using FAISS and Google Gemini

This project implements an intelligent multi-agent system that reads and indexes a PDF document, enabling users to ask natural language questions and receive accurate, AI-generated answers based on document content. It uses a dynamic routing mechanism to decide whether a user's query requires document context or can be answered generally.

This is designed for learning, demo purposes, and as a foundation for production-grade document question answering tools.

## Features

- Semantic Search over PDF content using FAISS and Sentence Transformers
- Google Gemini API for summarization, question answering, critique, and routing
- Multi-Agent Architecture (Retrieval, Summarizer, Q&A, Critique, Router)
- Document-Based and General Knowledge Query Handling
- Built and tested in Google Colab

## Architecture Overview

The pipeline consists of the following intelligent agents:

| Agent           | Purpose                                                                 |
|----------------|-------------------------------------------------------------------------|
| Router Agent    | Detects if the query is document-based or general knowledge            |
| Retrieval Agent | Searches for relevant PDF chunks using FAISS                           |
| Summarizer Agent| Condenses the retrieved information into a concise summary             |
| Q&A Agent       | Answers the user query based on the summary using Gemini               |
| Critique Agent  | Reviews and refines the AI's response for professional quality         |
| General Agent   | Answers questions not related to the document                          |

## Tech Stack

| Component              | Library/Tool               |
|------------------------|----------------------------|
| PDF Text Extraction    | PyMuPDF (fitz)             |
| Text Embedding         | SentenceTransformers       |
| Vector Search          | FAISS                      |
| Text Chunking          | LangChain                  |
| Language Model         | Google Generative AI (Gemini) |
| Development Platform   | Google Colab               |

## Installation

If running locally (not in Colab), install the required libraries:

```bash
pip install pymupdf faiss-cpu sentence-transformers langchain google-generativeai
