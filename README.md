# TP3 - Inteligência Artificial

## Tema: Retrieval-Augmented Generation com Large Language Models (LLMs)

Este projeto consiste na adaptação da plataforma UROBOT para integrar o modelo LLM local **Ollama** e usar **Retrieval-Augmented Generation (RAG)** com PDFs específicos — incluindo o Regulamento Pedagógico da ESTG.

## Estrutura do Projeto

- `urobot/`: Código principal do chatbot com integração Ollama + RAG
- `resources/`: PDFs de referência para alimentar a base de dados vetorial
- `vector_store/`: Ficheiros gerados para a base de dados vetorial (ChromaDB)
- `setup/`: Instruções de instalação
- `screenshots/`: Prints para incluir no PowerPoint
- `presentation/`: Apresentação final em PPTX

## Requisitos
Python 3.10+, Ollama, ChromaDB, LangChain, PyPDF2
