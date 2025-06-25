#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interface Streamlit para o RAG com Ollama para o Regulamento Pedagógico da ESTG
Versão refatorada com arquitetura modular
"""

import streamlit as st
import os
import time
import logging
from typing import Dict, Any

# Importar módulos do projeto
from src.data.document_loader import load_pdf, split_documents
from src.models.embeddings import create_vectorstore, get_retriever
from src.models.rag import create_qa_chain, process_query
from src.utils.cache import SimpleCache, normalize_query, timed_execution
from src.config.settings import (
    PDF_PATH, 
    VECTOR_STORE_DIR,
    CACHE_TTL_VECTORSTORE,
    CACHE_TTL_RESPONSES
)

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuração da página Streamlit
st.set_page_config(
    page_title="UROBOT - Regulamento Pedagógico ESTG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-bottom: 1rem;
    }
    .response-box {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .document-box {
        background-color: #F0FDF4;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border-left: 4px solid #10B981;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Inicializar o estado da sessão
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "query_cache" not in st.session_state:
    st.session_state.query_cache = SimpleCache(ttl=CACHE_TTL_RESPONSES)

# Título principal
st.markdown('<h1 class="main-header">UROBOT - Assistente do Regulamento Pedagógico ESTG</h1>', unsafe_allow_html=True)

# Função para carregar o PDF e criar o vectorstore
@st.cache_resource(ttl=CACHE_TTL_VECTORSTORE)
def load_documents_and_create_vectorstore(recreate=False):
    """
    Carrega o PDF e cria o vectorstore com cache
    
    Args:
        recreate: Se True, recria o vectorstore mesmo se já existir
        
    Returns:
        Vectorstore configurado ou None em caso de erro
    """
    with st.spinner("Carregando o Regulamento Pedagógico..."):
        try:
            # Carregar o PDF
            documents = load_pdf(PDF_PATH)
            st.success(f"PDF carregado com {len(documents)} páginas")
            
            # Dividir em chunks
            chunks = split_documents(documents)
            st.success(f"Documento dividido em {len(chunks)} chunks")
            
            # Criar vectorstore
            vectorstore = create_vectorstore(chunks, recreate)
            st.success("Vectorstore criado/carregado com sucesso")
            
            return vectorstore
        except Exception as e:
            st.error(f"❌ Erro ao processar o documento: {str(e)}")
            st.exception(e)
            return None

# Função para configurar o modelo e o pipeline RAG
def setup_rag_pipeline(vectorstore):
    """
    Configura o pipeline RAG com o vectorstore
    
    Args:
        vectorstore: Vectorstore configurado
        
    Returns:
        Tupla (retriever, qa_chain) ou None em caso de erro
    """
    with st.spinner("Configurando o pipeline de RAG..."):
        try:
            # Configurar o retriever
            retriever = get_retriever(vectorstore)
            
            # Criar a cadeia de QA
            qa_chain = create_qa_chain(retriever)
            
            st.success("Pipeline RAG configurado com sucesso")
            return retriever, qa_chain
        except Exception as e:
            st.error(f"❌ Erro ao configurar o pipeline RAG: {str(e)}")
            st.exception(e)
            return None, None

# Função para processar a consulta e retornar a resposta com tempo de execução
@st.cache_data(ttl=CACHE_TTL_RESPONSES)
def get_cached_response(_query, _retriever, _qa_chain):
    """
    Processa uma consulta e retorna a resposta (com cache)
    
    Args:
        _query: Consulta do usuário
        _retriever: Retriever configurado
        _qa_chain: Cadeia de QA configurada
        
    Returns:
        Dicionário com resposta, documentos e tempo de execução
    """
    try:
        # Medir o tempo de execução
        start_time = time.time()
        
        # Processar a consulta
        result = process_query(_query, _qa_chain)
        
        # Calcular o tempo de execução
        execution_time = time.time() - start_time
        
        return {
            "resposta": result["resposta"],
            "documentos": result["documentos"],
            "tempo": execution_time
        }
    except Exception as e:
        logger.error(f"Erro ao processar consulta: {str(e)}")
        return {
            "resposta": f"Erro ao processar pergunta: {str(e)}",
            "documentos": [],
            "tempo": 0
        }

# Barra lateral
with st.sidebar:
    st.markdown('<h2 class="sub-header">Configurações</h2>', unsafe_allow_html=True)
    
    # Opção para recriar o vectorstore
    recriar_vectorstore = st.checkbox("Recriar Vectorstore", value=False, 
                                     help="Marque esta opção se quiser recriar o vectorstore do zero")
    
    # Botão para iniciar o sistema RAG
    if st.button("Iniciar Sistema RAG", use_container_width=True):
        # Carregar o PDF e criar o vectorstore
        vectorstore = load_documents_and_create_vectorstore(recreate=recriar_vectorstore)
        
        if vectorstore:
            # Configurar o pipeline RAG
            retriever, qa_chain = setup_rag_pipeline(vectorstore)
            
            if retriever and qa_chain:
                st.session_state.retriever = retriever
                st.session_state.qa_chain = qa_chain
                st.success("✅ Sistema RAG inicializado com sucesso!")
    
    # Informações sobre o sistema
    st.markdown("### Sobre o Sistema")
    st.markdown("""
    Este assistente utiliza **Retrieval-Augmented Generation (RAG)** com o modelo **Ollama (llama3)** para responder perguntas sobre o Regulamento Pedagógico da ESTG.
    
    O sistema:
    1. Carrega o PDF do regulamento
    2. Divide em chunks menores
    3. Cria embeddings com Ollama
    4. Armazena em um vectorstore (ChromaDB)
    5. Recupera documentos relevantes para cada pergunta
    6. Gera respostas baseadas nos documentos recuperados
    """)
    
    # Exemplos de perguntas
    st.markdown("### Exemplos de Perguntas")
    perguntas_exemplo = [
        "Quais são os tipos de avaliação previstos no regulamento?",
        "Como funciona a época especial de exames?",
        "Quais são as condições para obter o estatuto de estudante-atleta?",
        "Qual o prazo para revisão de provas?"
    ]
    
    for pergunta in perguntas_exemplo:
        if st.button(pergunta, key=f"btn_{pergunta}", use_container_width=True):
            st.session_state.exemplo_pergunta = pergunta

# Área principal
st.markdown('<h2 class="sub-header">Pergunte sobre o Regulamento Pedagógico</h2>', unsafe_allow_html=True)

# Verificar se o sistema foi inicializado
if "retriever" not in st.session_state or "qa_chain" not in st.session_state:
    st.info("ℹ️ Por favor, inicialize o sistema RAG usando o botão na barra lateral.")

# Campo de entrada para a pergunta
query = st.text_input("Sua pergunta:", key="query", 
                      placeholder="Digite sua pergunta sobre o regulamento...",
                      value=st.session_state.get("exemplo_pergunta", ""))

# Limpar a pergunta de exemplo após usá-la
if "exemplo_pergunta" in st.session_state:
    del st.session_state.exemplo_pergunta

# Botão para enviar a pergunta
if st.button("Enviar Pergunta") or query:
    if "retriever" not in st.session_state or "qa_chain" not in st.session_state:
        st.warning("⚠️ Por favor, inicialize o sistema RAG primeiro usando o botão na barra lateral.")
    elif not query or len(query.strip()) < 3:
        st.warning("⚠️ Por favor, digite uma pergunta mais específica.")
    else:
        with st.spinner("Buscando resposta..."):
            try:
                # Registrar a pergunta no histórico
                st.session_state.chat_history.append({"role": "user", "content": query})
                
                # Verificar cache local primeiro
                query_norm = normalize_query(query)
                cached_result = st.session_state.query_cache.get(query_norm)
                
                if cached_result:
                    logger.info("Usando resposta em cache")
                    resultado = cached_result
                else:
                    # Usar a função de cache para obter a resposta
                    resultado = get_cached_response(query, st.session_state.retriever, st.session_state.qa_chain)
                    # Armazenar no cache local
                    st.session_state.query_cache.set(query_norm, resultado)
                
                resposta_texto = resultado["resposta"]
                documentos_fonte = resultado["documentos"]
                tempo = resultado["tempo"]
                
                # Registrar a resposta no histórico
                st.session_state.chat_history.append({"role": "assistant", "content": resposta_texto})
                
                # Exibir a resposta
                st.markdown(f'<div class="response-box">{resposta_texto}</div>', unsafe_allow_html=True)
                st.info(f"⏱️ Tempo de resposta: {tempo:.2f} segundos")
                
                # Exibir os documentos fonte
                with st.expander("📄 Ver documentos fonte", expanded=False):
                    for i, doc in enumerate(documentos_fonte, 1):
                        st.markdown(f'<div class="document-box"><strong>Documento {i}:</strong> {doc.page_content}</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Erro ao processar pergunta: {str(e)}")
                st.exception(e)

# Exibir histórico de chat
if st.session_state.chat_history:
    with st.expander("💬 Histórico de Perguntas e Respostas", expanded=False):
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"**Você:** {msg['content']}")
            else:
                st.markdown(f"**Assistente:** {msg['content']}")
            st.markdown("---")

# Rodapé
st.markdown("---")
st.markdown("Desenvolvido para o Trabalho Prático de Inteligência Artificial - IPVC ESTG")
