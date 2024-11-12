# 🌟 Baian System - 1st Place Winners at Datathon by Elm 🌟

## 👥 Team Members

- **Sara Alkhoneen** – [![LinkedIn](https://img.shields.io/badge/-Connect-blue?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sara-alkhoneen-5a14a514a/)
- **Layan Almousa** – [![LinkedIn](https://img.shields.io/badge/-Connect-blue?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/layanalmousa?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app)
- **Retaj Alghamdi** – [![LinkedIn](https://img.shields.io/badge/-Connect-blue?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ritaj-alghamdi-7594092a3?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app)
- **Anwar Alshamrani** – [![LinkedIn](https://img.shields.io/badge/-Connect-blue?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/anwar-alshamrani-606702243/)
- **Hend Alghamdi** – [![LinkedIn](https://img.shields.io/badge/-Connect-blue?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/hendalghamdi/)
---

## 🌐 Project Overview

The **Baian System** is an AI-powered information retrieval and question-answering (QA) system that efficiently parses, stores, and retrieves document data, specializing in Arabic language support. Built to handle real user queries with precision, it focuses on providing accurate answers without making up information.

### 🔑 Key Components & Technologies

- **LlamaParse API** 📄: Efficiently parses and stores documents in `.pkl` format for easy retrieval.

- **LangChain Library** 📚: Splits document data into chunks and vectors, with chunks stored in **Chroma Vector Database**.

- **Embedding Model (HuggingFace)** 🤖: Uses `BAAI\bge-m3` model for embedding, transforming document chunks into vector representations.

- **Groq Chat Model with Llama 3** 💬: Chatbot model, utilizing **Llama 3** for accurate Arabic language support, guided by prompts to ensure factual responses.

- **QA System with LangChain's RetrievalQA** 🔍: Merges vector database and chat model to power the QA system, providing contextually relevant answers from the document.

### ⚙️ How It Works

1. **Parsing & Storing**: LlamaParse API reads and stores document data for easy access.
2. **Data Chunking**: LangChain segments data into chunks, embedding them and storing in **Chroma**.
3. **QA System Creation**: LangChain’s `RetrievalQA` integrates the chat model and vector database, enabling the QA system.
4. **Query Handling**: On query submission, the system locates and returns document-based answers without generating fabricated responses.

---

## 🎉 Fun Fact

The name **Baian** has a clever play on words!  
- It includes "**AI**," highlighting the artificial intelligence focus.
- In Arabic, **"بيان"** means "clarity" or "statement," aligning perfectly with **Baian's** goal of making information clear and accessible from data (**بيانات**).

---

## 📌 Additional Resources

## 🖼️ System Overview Image


