#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo para gerenciamento de embeddings e vectorstore
"""

import os
import logging
from typing import List, Optional

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

from src.config.settings import (
    VECTOR_STORE_DIR, 
    OLLAMA_EMBEDDINGS_MODEL,
    RETRIEVER_K
)

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_embeddings() -> OllamaEmbeddings:
    """
    Cria um objeto de embeddings usando o modelo Ollama
    
    Returns:
        Objeto OllamaEmbeddings configurado
    """
    logger.info(f"Criando embeddings com o modelo {OLLAMA_EMBEDDINGS_MODEL}")
    try:
        return OllamaEmbeddings(model=OLLAMA_EMBEDDINGS_MODEL)
    except Exception as e:
        logger.error(f"Erro ao criar embeddings: {str(e)}")
        raise

def create_vectorstore(documents: List[Document], recreate: bool = False) -> Chroma:
    """
    Cria ou carrega um vectorstore a partir de documentos
    
    Args:
        documents: Lista de documentos para criar embeddings
        recreate: Se True, recria o vectorstore mesmo se já existir
        
    Returns:
        Objeto Chroma vectorstore
    """
    try:
        embeddings = create_embeddings()
        
        # Verificar se já existe um vectorstore persistido
        if os.path.exists(VECTOR_STORE_DIR) and os.listdir(VECTOR_STORE_DIR) and not recreate:
            logger.info(f"Carregando vectorstore existente de: {VECTOR_STORE_DIR}")
            return Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embeddings)
        
        # Criar novo vectorstore
        logger.info(f"Criando novo vectorstore em: {VECTOR_STORE_DIR}")
        vectorstore = Chroma.from_documents(
            documents, 
            embedding=embeddings, 
            persist_directory=VECTOR_STORE_DIR
        )
        vectorstore.persist()
        logger.info("Vectorstore criado e persistido com sucesso")
        return vectorstore
    except Exception as e:
        logger.error(f"Erro ao criar vectorstore: {str(e)}")
        raise

def get_retriever(vectorstore: Chroma):
    """
    Configura um retriever a partir do vectorstore
    
    Args:
        vectorstore: Objeto Chroma vectorstore
        
    Returns:
        Retriever configurado
    """
    logger.info(f"Configurando retriever com k={RETRIEVER_K}")
    try:
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVER_K}
        )
    except Exception as e:
        logger.error(f"Erro ao configurar retriever: {str(e)}")
        raise
