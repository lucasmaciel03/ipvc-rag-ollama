#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo para carregamento e processamento de documentos
"""

import os
import logging
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from src.config.settings import PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_pdf(pdf_path: str = PDF_PATH) -> List[Document]:
    """
    Carrega um arquivo PDF e retorna uma lista de documentos
    
    Args:
        pdf_path: Caminho para o arquivo PDF
        
    Returns:
        Lista de documentos carregados
    """
    logger.info(f"Carregando PDF: {pdf_path}")
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        logger.info(f"PDF carregado com {len(documents)} páginas")
        return documents
    except Exception as e:
        logger.error(f"Erro ao carregar PDF: {str(e)}")
        raise

def split_documents(documents: List[Document], 
                   chunk_size: int = CHUNK_SIZE, 
                   chunk_overlap: int = CHUNK_OVERLAP) -> List[Document]:
    """
    Divide documentos em chunks menores
    
    Args:
        documents: Lista de documentos a serem divididos
        chunk_size: Tamanho de cada chunk
        chunk_overlap: Sobreposição entre chunks
        
    Returns:
        Lista de documentos divididos em chunks
    """
    logger.info(f"Dividindo documentos em chunks (tamanho={chunk_size}, overlap={chunk_overlap})")
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Documento dividido em {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Erro ao dividir documentos: {str(e)}")
        raise

def load_and_split_documents(pdf_path: str = PDF_PATH) -> List[Document]:
    """
    Carrega um PDF e divide em chunks em uma única função
    
    Args:
        pdf_path: Caminho para o arquivo PDF
        
    Returns:
        Lista de documentos divididos em chunks
    """
    documents = load_pdf(pdf_path)
    return split_documents(documents)
