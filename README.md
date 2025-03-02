# IntelliSearch using Crawler and RAG
**IntelliSearch web crawler** is an intelligent web crawler that leverages advanced AI language models (LLMs) along with modern search techniques to deliver precise, context-aware answers. The system employs dense vector retrieval (via Qdrant) and RAG Fusion for re-ranking, and it’s designed to be easily extended with advanced techniques such as Late Interaction and token-level refinement.

## Technologies
<img src="https://github.com/user-attachments/assets/09772236-6410-4b1f-bd66-66a7d8742be9" width="30" />
<img src="https://github.com/user-attachments/assets/0478f3b6-a895-422c-a4da-554d26c1cfc2" width="30"/>
<img src="https://github.com/user-attachments/assets/4955df76-7b5f-4590-b4a2-13a5abe90b6f" width="30"/>
<img src="https://github.com/user-attachments/assets/5b2b0e6c-520e-426e-af81-aea0a68c2556" width="30"/>

## Features

- **Hybrid LLM Integration:**
  - **Local LLMs:** Run directly on your machine for enhanced data privacy and control.
  - **Paid API LLMs:** Utilize cutting-edge models like OpenAI’s GPT-4 for superior performance and real-time capabilities.

- **Efficient Vector Search with Qdrant:**
  - **Search results:** SerpAPI free use tier for Google search results 
  - **Fast & Accurate:** Qdrant efficiently stores and retrieves dense embeddings to ensure quick and precise search results at scale.

- **Advanced Retrieval Techniques:**
  - **RAG Fusion Reranking:** Merges multiple search results using reciprocal rank fusion to prioritize the most relevant documents.
  - **Planned Enhancements:** Integrate Late Interaction techniques (e.g., ColBERT-style token-level re-ranking) and hybrid search methods (combining dense embeddings with BM25).

- **Enterprise-Ready:**
  - **Customizable for Closed Systems:** Easily tunable for internal databases and proprietary search systems, similar to industry-leading apps similar to Perplexity.

- **User-Friendly Interface:**
  - **Gradio UI:** A simple, interactive web-based interface for seamless user interactions.

---

## 📂 Folder Structure
```
WebcrawlerRAG/
  ├── components/
  │   ├── chat_logic.py          # Contains the main logic for handling chat interactions and RAG techniques
  │   ├── ranking_modes.py       # Contains functions for different ranking modes like reciprocal rank fusion and unique union
  ├── services/
  │   ├── search_service.py      # Handles document search and loading
  ├── utils/
  │   ├── config.py              # Configuration settings for the project
  ├── app.py                     # Main application file to launch the Gradio UI
  ├── models.properties          # Configuration file listing available models
  ├── requirements.txt           # List of dependencies required for the project
  ├── README.md                  # Project documentation and instructions
  ├── .env                       # Environment variables (e.g., API keys, database URLs)

```

##  How It Works
-  **Document Loading & Processing**
    -  The system fetches documents via the search service and splits them into manageable chunks.
  
-  **Vector Storage & Retrieval**
    -  Chunks are embedded using a dense embedding model and stored in Qdrant. Retrieval is performed using dense vector search.

-  **RAG Fusion Re-ranking**
    -  Multiple search queries are generated, and results are merged using reciprocal rank fusion or Unique union for broader search use-cases to prioritize accurate matching.

-  **Answer Synthesis**
    -  The retrieved context is fed into an LLM (local or API-based) to generate a final answer in markdown format with links to sources.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Feel free to star, fork and contribute to this project and share your feedback!
