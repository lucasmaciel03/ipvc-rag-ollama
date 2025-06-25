#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interface Streamlit para o RAG com Ollama para o Regulamento Pedagógico da ESTG
"""

import streamlit as st
import os
import time
import sys
import shutil
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

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
        color: #2563EB;
        margin-bottom: 1rem;
    }
    .info-text {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #3B82F6;
    }
    .document-box {
        background-color: #F3F4F6;
        padding: 0.8rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        border-left: 3px solid #6B7280;
    }
    .response-box {
        background-color: #ECFDF5;
        padding: 1.2rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        border-left: 5px solid #10B981;
    }
</style>
""", unsafe_allow_html=True)

# Título da aplicação
st.markdown('<h1 class="main-header">UROBOT - Assistente do Regulamento Pedagógico ESTG</h1>', unsafe_allow_html=True)

# Inicializar variáveis de sessão
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Função para carregar o PDF e criar o vectorstore
@st.cache_resource(ttl=3600)  # Cache por 1 hora para melhor performance
def load_documents_and_create_vectorstore(recreate=False):
    with st.spinner("Carregando o Regulamento Pedagógico..."):
        # Obter o caminho absoluto para o diretório raiz do projeto
        root_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construir o caminho completo para o arquivo PDF
        pdf_path = os.path.join(root_dir, "resources", "ESTG_Regulamento-Frequencia-Avaliacao2023.pdf")
        st.info(f"Carregando PDF de: {pdf_path}")
        
        try:
            # Carregar o PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            st.success(f"PDF carregado com {len(documents)} páginas")
            
            # Dividir o documento em chunks otimizados
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,  # Chunks um pouco maiores para reduzir o número total de chunks
                chunk_overlap=80,  # Overlap menor para processamento mais rápido
                length_function=len
            )
            chunks = text_splitter.split_documents(documents)
            st.info(f"Documento dividido em {len(chunks)} chunks")
            
            # Configurar embeddings
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            
            # Usar o caminho absoluto para o diretório de persistência
            vector_store_path = os.path.join(root_dir, "vector_store")
            
            # Verificar se já existe um vectorstore persistido
            if recreate and os.path.exists(vector_store_path) and os.listdir(vector_store_path):
                st.warning("Removendo vectorstore existente...")
                shutil.rmtree(vector_store_path)
            
            # Criar ou carregar o vectorstore
            if not os.path.exists(vector_store_path) or not os.listdir(vector_store_path) or recreate:
                st.info("Criando novo vectorstore (pode demorar alguns minutos)...")
                vectorstore = Chroma.from_documents(
                    documents=chunks,  # Usar chunks em vez de documentos completos
                    embedding=embeddings,
                    persist_directory=vector_store_path
                )
                vectorstore.persist()
                st.success("Vectorstore criado e persistido com sucesso!")
            else:
                st.info("Usando vectorstore existente...")
                vectorstore = Chroma(persist_directory=vector_store_path, embedding_function=embeddings)
            
            return vectorstore
            
        except Exception as e:
            st.error(f"❌ Erro ao processar o documento: {str(e)}")
            st.exception(e)
            return None

# Cache para respostas anteriores para evitar reprocessamento
@st.cache_data(ttl=1800)  # Cache por 30 minutos
def get_cached_response(query, _retriever, _qa_chain):
    """Busca respostas no cache ou executa a consulta se não estiver em cache"""
    try:
        # Recuperar documentos relevantes
        start_time = time.time()
        docs = _retriever.get_relevant_documents(query)
        
        # Executar a pergunta com paralelização
        resposta = _qa_chain.invoke({"query": query})
        tempo = time.time() - start_time
        
        # Extrair a resposta e os documentos fonte
        if isinstance(resposta, dict) and "result" in resposta:
            resposta_texto = resposta["result"]
            documentos_fonte = resposta.get("source_documents", [])
        else:
            resposta_texto = str(resposta)
            documentos_fonte = docs
        
        return {
            "resposta": resposta_texto,
            "documentos": documentos_fonte,
            "tempo": tempo
        }
    except Exception as e:
        st.error(f"Erro ao processar pergunta: {str(e)}")
        return {
            "resposta": f"Erro ao processar pergunta: {str(e)}",
            "documentos": [],
            "tempo": 0
        }

# Função para configurar o modelo e o pipeline RAG
def setup_rag_pipeline(vectorstore):
    with st.spinner("Configurando o pipeline de RAG..."):
        try:
            # Configurar o retriever otimizado para velocidade
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 2  # Reduzir para 2 chunks mais relevantes para maior velocidade
                }
            )
            
            # Configurar o modelo Ollama com parâmetros otimizados para velocidade
            llm = OllamaLLM(
                model="llama3",
                temperature=0.1,
                stop=["\n\n"],
                top_p=0.9,
                num_ctx=2048,  # Contexto menor para respostas mais rápidas
                num_thread=4,  # Usar múltiplas threads
                num_gpu=1      # Usar GPU se disponível
            )
            
            # Criar um template de prompt personalizado para instruir o modelo a responder em português
            prompt_template = """
            Você é um assistente especializado no Regulamento Pedagógico da ESTG (Escola Superior de Tecnologia e Gestão).
            Responda à pergunta em PORTUGUÊS com base nas informações fornecidas abaixo.
            Se a informação não estiver presente nos documentos fornecidos, diga que não tem informações suficientes para responder.
            
            Contexto:
            {context}
            
            Pergunta: {question}
            
            Resposta em português:
            """
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Criar a cadeia de QA
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            return retriever, qa_chain
            
        except Exception as e:
            st.error(f"❌ Erro ao configurar o pipeline RAG: {str(e)}")
            st.exception(e)
            return None, None

# Sidebar para configurações
with st.sidebar:
    st.markdown('<h2 class="sub-header">Configurações</h2>', unsafe_allow_html=True)
    
    # Botão para recriar o vectorstore
    recreate_vectorstore = st.button("Recriar Vectorstore", help="Recria o vectorstore do zero. Isso pode levar alguns minutos.")
    
    # Botão para carregar o modelo e configurar o pipeline
    if st.button("Iniciar Sistema RAG", help="Carrega o modelo e configura o pipeline RAG"):
        vectorstore = load_documents_and_create_vectorstore(recreate=recreate_vectorstore)
        if vectorstore:
            st.session_state.retriever, st.session_state.qa_chain = setup_rag_pipeline(vectorstore)
            if st.session_state.qa_chain:
                st.success("Sistema RAG inicializado com sucesso!")
    
    # Informações sobre o projeto
    st.markdown("---")
    st.markdown('<h3 class="sub-header">Sobre o Projeto</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-text">
    Este projeto utiliza Retrieval-Augmented Generation (RAG) com o modelo Ollama para responder perguntas sobre o Regulamento Pedagógico da ESTG.
    
    Desenvolvido como parte do projeto acadêmico de Inteligência Artificial.
    </div>
    """, unsafe_allow_html=True)
    
    # Perguntas predefinidas
    st.markdown('<h3 class="sub-header">Perguntas Exemplo</h3>', unsafe_allow_html=True)
    if st.button("Como posso justificar as faltas?"):
        st.session_state.query = "Como posso justificar as faltas?"
    if st.button("O que é a avaliação contínua?"):
        st.session_state.query = "O que é a avaliação contínua?"
    if st.button("Quais são os tipos de avaliação?"):
        st.session_state.query = "Quais são os tipos de avaliação?"

# Interface principal para perguntas e respostas
st.markdown('<h2 class="sub-header">Pergunte sobre o Regulamento Pedagógico</h2>', unsafe_allow_html=True)

# Campo de entrada para a pergunta
query = st.text_input("Sua pergunta:", key="query", placeholder="Digite sua pergunta sobre o regulamento...")

# Botão para enviar a pergunta
if st.button("Enviar Pergunta") or query:
    if not st.session_state.qa_chain:
        st.warning("Por favor, inicialize o sistema RAG primeiro usando o botão na barra lateral.")
    elif not query:
        st.warning("Por favor, digite uma pergunta.")
    else:
        with st.spinner("Buscando resposta..."):
            try:
                # Registrar a pergunta no histórico
                st.session_state.chat_history.append({"role": "user", "content": query})
                
                # Usar a função de cache para obter a resposta
                resultado = get_cached_response(query, st.session_state.retriever, st.session_state.qa_chain)
                
                resposta_texto = resultado["resposta"]
                documentos_fonte = resultado["documentos"]
                tempo = resultado["tempo"]
                
                # Registrar a resposta no histórico
                st.session_state.chat_history.append({"role": "assistant", "content": resposta_texto})
                
                # Exibir a resposta
                st.markdown(f'<div class="response-box">{resposta_texto}</div>', unsafe_allow_html=True)
                st.info(f"Tempo de resposta: {tempo:.2f} segundos")
                
                # Exibir os documentos fonte
                with st.expander("Ver documentos fonte", expanded=False):
                    for i, doc in enumerate(documentos_fonte, 1):
                        st.markdown(f'<div class="document-box"><strong>Documento {i}:</strong> {doc.page_content}</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Erro ao processar pergunta: {str(e)}")
                st.exception(e)

# Exibir histórico de chat
if st.session_state.chat_history:
    with st.expander("Histórico de Perguntas e Respostas", expanded=False):
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"**Você:** {msg['content']}")
            else:
                st.markdown(f"**Assistente:** {msg['content']}")
            st.markdown("---")

# Rodapé
st.markdown("---")
st.markdown("Desenvolvido com ❤️ usando Streamlit, LangChain e Ollama")
