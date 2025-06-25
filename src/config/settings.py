#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configurações globais para o sistema RAG com Ollama
"""

import os

# Caminhos de diretórios
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESOURCES_DIR = os.path.join(ROOT_DIR, "resources")
VECTOR_STORE_DIR = os.path.join(ROOT_DIR, "vector_store")

# Configurações do PDF
PDF_PATH = os.path.join(RESOURCES_DIR, "ESTG_Regulamento-Frequencia-Avaliacao2023.pdf")

# Configurações de processamento de documentos
CHUNK_SIZE = 800
CHUNK_OVERLAP = 80

# Configurações do modelo Ollama
OLLAMA_MODEL = "llama3"
OLLAMA_EMBEDDINGS_MODEL = "nomic-embed-text"
OLLAMA_TEMPERATURE = 0.1
OLLAMA_TOP_P = 0.9
OLLAMA_NUM_CTX = 2048
OLLAMA_NUM_THREAD = 4
OLLAMA_NUM_GPU = 1

# Configurações do retriever
RETRIEVER_K = 2  # Número de documentos a recuperar

# Cache
CACHE_TTL_VECTORSTORE = 3600  # 1 hora
CACHE_TTL_RESPONSES = 1800    # 30 minutos
