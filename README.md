# LangChain Projects

A collection of projects exploring LangChain for building LLM-powered applications — 
covering retrieval-augmented generation (RAG), chatbots, agents, text summarization, 
and integrations with multiple LLM providers and vector stores.

## Contents

| Project | Description |
|---------|--------------|
| [Chat-SQL](./Chat-SQL) | Natural language interface for querying SQL databases |
| [Chatbot](./Chatbot) | Conversational chatbot built with LangChain |
| [Google-Gemma](./Google-Gemma) | LLM application using Google's Gemma model |
| [Hugging-Face](./Hugging-Face) | Text generation, Q&A, and embeddings using Hugging Face models |
| [Hybrid Search with...](./Hybrid_Search_with...) | Hybrid (keyword + semantic) search implementation |
| [Nvidia-NIM](./Nvidia-NIM) | LLM application using Nvidia NIM microservices |
| [RAG-Document](./RAG-Document) | Retrieval-augmented generation over document data |
| [Text-Summarization](./Text-Summarization) | Multiple approaches to LLM-based text summarization |
| [YouTube-Text-Summarization](./YoutTube-Text-Sum...) | Summarizing YouTube video transcripts |
| [agents](./agents) | LangChain agent-based applications |
| [AstraDB and LangChain](./AstraDB-and-LangC...) | Vector store integration using AstraDB |
| [Prompt Engineering](./PromptEngineering...) | Prompt design and engineering experiments |

## Tech Stack
- Python, LangChain
- LLM providers: Groq, Hugging Face, Google Gemma, Nvidia NIM
- Vector stores: FAISS, AstraDB
- python-dotenv for environment/API key management

## How to Use
1. Clone the repository
2. Navigate to the project folder you're interested in
3. Install dependencies: `pip install -r requirements.txt`
4. Set up required API keys in a `.env` file (see individual project READMEs for specifics)
5. Run the relevant notebook or script

## Note
Some projects are Jupyter notebooks (`.ipynb`) used for experimentation, while others 
are deployed as standalone Python applications (`.py`), some using Streamlit for the UI.


