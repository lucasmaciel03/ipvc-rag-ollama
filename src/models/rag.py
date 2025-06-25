#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo para configuração e execução do pipeline RAG com Ollama
"""

import logging
from typing import Dict, Any, List

from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from src.config.settings import (
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
    OLLAMA_TOP_P,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_THREAD,
    OLLAMA_NUM_GPU
)

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Template de prompt em português
PROMPT_TEMPLATE = """
Você é um assistente especializado no Regulamento Pedagógico da ESTG (Escola Superior de Tecnologia e Gestão).
Responda à pergunta em PORTUGUÊS com base nas informações fornecidas abaixo.
Se a informação não estiver presente nos documentos fornecidos, diga que não tem informações suficientes para responder.

Contexto:
{context}

Pergunta: {question}

Resposta em português:
"""

def create_llm() -> OllamaLLM:
    """
    Cria e configura o modelo Ollama LLM
    
    Returns:
        Modelo OllamaLLM configurado
    """
    logger.info(f"Configurando modelo Ollama ({OLLAMA_MODEL})")
    try:
        return OllamaLLM(
            model=OLLAMA_MODEL,
            temperature=OLLAMA_TEMPERATURE,
            stop=["\n\n"],
            top_p=OLLAMA_TOP_P,
            num_ctx=OLLAMA_NUM_CTX,
            num_thread=OLLAMA_NUM_THREAD,
            num_gpu=OLLAMA_NUM_GPU
        )
    except Exception as e:
        logger.error(f"Erro ao configurar modelo Ollama: {str(e)}")
        raise

def create_qa_chain(retriever):
    """
    Cria uma cadeia de QA com o retriever e o modelo LLM
    
    Args:
        retriever: Retriever configurado para buscar documentos relevantes
        
    Returns:
        Cadeia RetrievalQA configurada
    """
    logger.info("Configurando cadeia de QA")
    try:
        llm = create_llm()
        
        # Criar o template de prompt
        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        
        # Criar a cadeia de QA
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        logger.info("Cadeia de QA configurada com sucesso")
        return qa_chain
    except Exception as e:
        logger.error(f"Erro ao criar cadeia de QA: {str(e)}")
        raise

def process_query(query: str, qa_chain) -> Dict[str, Any]:
    """
    Processa uma consulta usando a cadeia de QA
    
    Args:
        query: Pergunta do usuário
        qa_chain: Cadeia de QA configurada
        
    Returns:
        Dicionário com a resposta e documentos fonte
    """
    logger.info(f"Processando consulta: {query}")
    try:
        # Executar a consulta
        result = qa_chain.invoke({"query": query})
        
        # Extrair a resposta e os documentos fonte
        if isinstance(result, dict) and "result" in result:
            resposta = result["result"]
            documentos = result.get("source_documents", [])
        else:
            resposta = str(result)
            documentos = []
            
        logger.info("Consulta processada com sucesso")
        return {
            "resposta": resposta,
            "documentos": documentos
        }
    except Exception as e:
        logger.error(f"Erro ao processar consulta: {str(e)}")
        raise
