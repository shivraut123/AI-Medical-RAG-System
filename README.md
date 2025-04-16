# 🩺 MediBot Pro: RAG-Based Medical Assistant

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![LangChain](https://img.shields.io/badge/Framework-LangChain-green)
![Gemini](https://img.shields.io/badge/AI-Google%20Gemini%202.5-orange)

## 📌 Project Overview
**MediBot Pro** is a specialized conversational AI system designed to provide accurate medical information, symptom analysis, and first-aid guidance. Unlike standard LLMs (like ChatGPT) which can hallucinate medical facts, this project utilizes **Retrieval-Augmented Generation (RAG)**.

It retrieves verified information exclusively from the *Gale Encyclopedia of Medicine* (PDF Dataset) before generating a response, ensuring that every answer is grounded in authoritative medical literature.

## 🚀 Key Features
- **Retrieval-Augmented Generation (RAG):** eliminates AI hallucinations by grounding answers in a vector database.
- **Multi-Mode Interaction:**
  - 🩺 **Consultant Mode:** Explains medical concepts and diseases.
  - 🔍 **Symptom Checker:** Maps user complaints to potential conditions based on the dataset.
  - 🚑 **First Aid Guide:** Provides structured, step-by-step emergency protocols.
- **Source Citation:** Every response includes a direct reference to the page number and text from the source PDF for verification.
- **Local Vector Storage:** Uses FAISS (Facebook AI Similarity Search) for efficient, offline similarity search.
- **Hospital Finder:** Integrated geolocation feature to find nearby medical facilities.

## 🛠️ Technical Architecture

The system follows a standard RAG pipeline:
1.  **Ingestion:** The medical encyclopedia (PDF) is loaded and split into chunks of 500 characters.
2.  **Embedding:** Text chunks are converted into vector embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
3.  **Storage:** Vectors are stored locally using the FAISS vector database.
4.  **Retrieval:** User queries are converted to vectors; the system searches FAISS for the top 3 most relevant context chunks.
5.  **Generation:** The context + user query are sent to **Google Gemini 2.5 Flash**, which generates the final natural language response.

## 💻 Tech Stack
* **Language:** Python 3.12
* **LLM:** Google Gemini 2.5 Flash (via API)
* **Orchestration:** LangChain
* **Vector Database:** FAISS (CPU)
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **Frontend:** Streamlit

## ⚙️ Installation & Setup

**Prerequisites:**
- Python 3.9 or higher installed.
- A Google Cloud API Key (for Gemini).

**1. Clone the Repository**
```bash
git clone [https://github.com/your-username/MediBot-Pro.git](https://github.com/your-username/MediBot-Pro.git)
cd MediBot-Pro