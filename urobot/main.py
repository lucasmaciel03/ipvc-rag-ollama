#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG com Ollama para o Regulamento Pedagógico da ESTG
"""

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import time
import sys
import os
import shutil

def main():
    print("=== RAG com Ollama para o Regulamento Pedagógico da ESTG ===\n")
    
    # === 1. Carregar o PDF ===
    # Obter o caminho absoluto para o diretório raiz do projeto
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Construir o caminho completo para o arquivo PDF
    pdf_path = os.path.join(root_dir, "resources", "ESTG_Regulamento-Frequencia-Avaliacao2023.pdf")
    print(f"Carregando PDF de: {pdf_path}")
    
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(f"PDF carregado com {len(documents)} páginas")

        # === 2. Dividir o texto em chunks menores ===
        print("\nDividindo o documento em chunks menores...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Chunks menores para melhor precisão
            chunk_overlap=100,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Documento dividido em {len(chunks)} chunks")
        
        # === 3. Converter em embeddings e criar vector store ===
        print("\nConvertendo documentos em embeddings...")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        # Usar o caminho absoluto para o diretório de persistência
        vector_store_path = os.path.join(root_dir, "vector_store")
        print(f"Salvando vectorstore em: {vector_store_path}")
        
        # Verificar se já existe um vectorstore persistido
        recriar_vectorstore = False
        if os.path.exists(vector_store_path) and os.listdir(vector_store_path):
            resposta = input("Vectorstore já existe. Deseja recriá-lo? (s/n): ")
            if resposta.lower() == 's':
                print("Removendo vectorstore existente...")
                shutil.rmtree(vector_store_path)
                recriar_vectorstore = True
            else:
                print("Usando vectorstore existente...")
                vectorstore = Chroma(persist_directory=vector_store_path, embedding_function=embeddings)
        else:
            recriar_vectorstore = True
        
        if recriar_vectorstore:
            print("Criando novo vectorstore (pode demorar alguns minutos)...")
            vectorstore = Chroma.from_documents(
                documents=chunks,  # Usar chunks em vez de documentos completos
                embedding=embeddings,
                persist_directory=vector_store_path
            )
            vectorstore.persist()
            print("Vectorstore criado e persistido com sucesso!")

        # === 4. Criar pipeline de RAG ===
        print("\nConfigurando o pipeline de RAG...")
        
        # Configurar o retriever para recuperar mais documentos relevantes
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # Recuperar os 3 chunks mais relevantes
        )
        
        print("Conectando ao modelo Ollama (llama3)...")
        llm = OllamaLLM(
            model="llama3",
            temperature=0.1,
            stop=["\n\n"],
            top_p=0.9
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
        
        print("Criando a cadeia de QA...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        print("Pipeline de RAG configurado com sucesso!")

        # === 5. Testar as perguntas específicas da Tarefa 6 ===
        print("\n===== Testando as perguntas da Tarefa 6 =====\n")
        
        # Perguntas específicas da Tarefa 6
        perguntas = [
            "Como posso justificar as faltas?",
            "O que é a avaliação contínua?"
        ]
        
        respostas = []
        
        for i, pergunta in enumerate(perguntas, 1):
            print(f"\n🔍 Pergunta {i}: {pergunta}")
            print("Gerando resposta... (pode demorar alguns segundos)")
            sys.stdout.flush()  # Forçar a saída imediata
            
            start_time = time.time()
            try:
                # Recuperar documentos relevantes
                docs = retriever.get_relevant_documents(pergunta)
                print(f"Recuperados {len(docs)} documentos relevantes")
                
                # Mostrar os documentos recuperados
                print("\nDocumentos fonte recuperados:")
                for j, doc in enumerate(docs, 1):
                    print(f"Documento {j}: {doc.page_content[:150]}...")
                
                # Executar a pergunta
                resposta = qa_chain.invoke({"query": pergunta})
                tempo = time.time() - start_time
                
                # Extrair a resposta e os documentos fonte
                if isinstance(resposta, dict) and "result" in resposta:
                    resposta_texto = resposta["result"]
                    documentos_fonte = resposta.get("source_documents", [])
                else:
                    resposta_texto = str(resposta)
                    documentos_fonte = []
                
                print(f"Tempo de resposta: {tempo:.2f} segundos")
                print(f"\n🧠 Resposta:\n{resposta_texto}")
                
                respostas.append((pergunta, resposta_texto, tempo, docs))
            except Exception as e:
                print(f"Erro ao processar pergunta: {str(e)}")
                import traceback
                print(traceback.format_exc())
        
        print("\n===== Testes concluídos =====\n")
        
        # Salvar as respostas em um arquivo para referência
        resultados_path = os.path.join(root_dir, "resultados_testes.txt")
        with open(resultados_path, "w", encoding="utf-8") as f:
            f.write("=== RESULTADOS DOS TESTES DA TAREFA 6 ===\n\n")
            
            for i, (pergunta, resposta, tempo, docs) in enumerate(respostas, 1):
                f.write(f"Pergunta {i}: {pergunta}\n")
                f.write(f"Tempo de resposta: {tempo:.2f} segundos\n")
                f.write(f"Resposta:\n{resposta}\n\n")
                
                f.write("Documentos fonte utilizados:\n")
                for j, doc in enumerate(docs, 1):
                    f.write(f"Documento {j}: {doc.page_content[:200]}...\n\n")
                
                f.write("---\n\n")
        
        print(f"Respostas salvas em: {resultados_path}")
    
        # === 6. Loop interativo para testes adicionais ===
        print("\nDeseja fazer mais perguntas? (Digite 'sair' para terminar)")
        while True:
            query = input("\n🔍 Pergunta adicional (ou escreve 'sair'): ")
            if query.lower() in ["sair", "exit", "quit"]:
                break
            
            print("Gerando resposta... (pode demorar alguns segundos)")
            sys.stdout.flush()  # Forçar a saída imediata
            
            start_time = time.time()
            try:
                # Recuperar documentos relevantes
                docs = retriever.get_relevant_documents(query)
                print(f"Recuperados {len(docs)} documentos relevantes")
                
                # Mostrar os documentos recuperados
                print("\nDocumentos fonte recuperados:")
                for j, doc in enumerate(docs[:2], 1):
                    print(f"Documento {j}: {doc.page_content[:100]}...")
                
                # Executar a pergunta
                resposta = qa_chain.invoke({"query": query})
                tempo = time.time() - start_time
                
                # Extrair a resposta
                if isinstance(resposta, dict) and "result" in resposta:
                    resposta_texto = resposta["result"]
                else:
                    resposta_texto = str(resposta)
                
                print(f"Tempo de resposta: {tempo:.2f} segundos")
                print(f"\n🧠 Resposta:\n{resposta_texto}")
                
            except Exception as e:
                print(f"Erro ao processar pergunta: {str(e)}")
                import traceback
                print(traceback.format_exc())
    
    except Exception as e:
        print(f"\n❌ Erro durante a execução: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
