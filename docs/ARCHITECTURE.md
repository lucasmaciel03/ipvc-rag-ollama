# Arquitetura do Sistema RAG

Este documento descreve em detalhes a arquitetura de Retrieval-Augmented Generation (RAG) implementada no sistema ESTG RegPedagogicoBot.

## Visão Geral da Arquitetura RAG

O sistema utiliza a arquitetura RAG (Retrieval-Augmented Generation) para melhorar a qualidade e precisão das respostas do modelo de linguagem Ollama. A arquitetura RAG combina:

1. Um componente de **recuperação** (retrieval) que busca informações relevantes em uma base de conhecimento
2. Um componente de **geração** (generation) que utiliza essas informações para produzir respostas contextualizadas

![Arquitetura RAG](https://miro.medium.com/v2/resize:fit:1400/1*CuVO0YrCKTOySKvyY0EBxw.png)

## Componentes Principais

### 1. Processamento de Documentos

O sistema processa o PDF do Regulamento Pedagógico da ESTG através dos seguintes passos:

```python
# Carregar o PDF
documents = load_pdf(PDF_PATH)

# Dividir em chunks
chunks = split_documents(documents)
```

#### Parâmetros de Divisão Otimizados

Os documentos são divididos em chunks de tamanho otimizado para equilibrar precisão e velocidade:

- **Tamanho do Chunk**: 800 caracteres
- **Sobreposição**: 80 caracteres

Estes parâmetros foram cuidadosamente ajustados após testes extensivos para garantir:
- Chunks grandes o suficiente para manter contexto semântico
- Pequenos o suficiente para recuperação precisa
- Sobreposição adequada para evitar perda de informação nas fronteiras

### 2. Sistema de Embeddings

O sistema utiliza o modelo `nomic-embed-text` do Ollama para gerar embeddings vetoriais para cada chunk de texto:

```python
# Criar embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")
```

Estes embeddings capturam a semântica do texto em vetores de alta dimensão, permitindo buscas por similaridade semântica.

### 3. Vectorstore com ChromaDB

Os embeddings são armazenados em um banco de dados vetorial persistente usando ChromaDB:

```python
# Criar/carregar vectorstore
vectorstore = Chroma.from_documents(
    chunks, 
    embedding=embeddings, 
    persist_directory=VECTOR_STORE_DIR
)
vectorstore.persist()
```

O ChromaDB oferece:
- Armazenamento eficiente de vetores
- Busca rápida por similaridade
- Persistência dos dados entre sessões

### 4. Retriever Otimizado

O sistema configura um retriever para buscar os documentos mais relevantes para cada consulta:

```python
# Configurar retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}  # Recupera apenas 2 documentos
)
```

#### Otimizações do Retriever:
- Busca por similaridade semântica (cosine similarity)
- Recuperação de apenas 2 documentos mais relevantes (otimizado para velocidade)
- Filtragem de documentos com baixa relevância

### 5. Modelo LLM Ollama

O sistema utiliza o modelo `llama3` do Ollama com parâmetros otimizados:

```python
# Configurar o modelo Ollama
llm = OllamaLLM(
    model="llama3",
    temperature=0.1,
    stop=["\n\n"],
    top_p=0.9,
    num_ctx=2048,
    num_thread=4,
    num_gpu=1
)
```

#### Parâmetros Otimizados:
- **Temperatura**: 0.1 (baixa aleatoriedade para respostas mais consistentes)
- **Top_p**: 0.9 (filtra tokens de baixa probabilidade)
- **Contexto**: 2048 tokens (equilibra velocidade e capacidade)
- **Threads**: 4 (paralelização para melhor desempenho)
- **GPU**: Utiliza aceleração por GPU quando disponível

### 6. Prompt Personalizado

O sistema utiliza um prompt personalizado em português para instruir o modelo:

```python
prompt_template = """
Você é um assistente especializado no Regulamento Pedagógico da ESTG (Escola Superior de Tecnologia e Gestão).
Responda à pergunta em PORTUGUÊS com base nas informações fornecidas abaixo.
Se a informação não estiver presente nos documentos fornecidos, diga que não tem informações suficientes para responder.

Contexto:
{context}

Pergunta: {question}

Resposta em português:
"""
```

Este prompt:
- Estabelece o papel do assistente
- Instrui o modelo a responder em português
- Limita as respostas às informações fornecidas
- Fornece uma estrutura clara para a resposta

### 7. Pipeline RAG Completo

O pipeline RAG é implementado usando a cadeia RetrievalQA do LangChain:

```python
# Criar a cadeia de QA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Método "stuff" para contextos menores
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)
```

Este pipeline:
1. Recebe uma pergunta do usuário
2. Recupera documentos relevantes usando o retriever
3. Combina os documentos em um único contexto
4. Envia o contexto e a pergunta para o modelo LLM
5. Retorna a resposta gerada e os documentos fonte

## Sistema de Cache Multi-camada

Para otimizar o desempenho, o sistema implementa um cache em múltiplas camadas:

### 1. Cache de Vectorstore (TTL: 1 hora)

```python
@st.cache_resource(ttl=3600)
def load_documents_and_create_vectorstore(recreate=False):
    # Implementação
```

### 2. Cache de Respostas (TTL: 30 minutos)

```python
@st.cache_data(ttl=1800)
def get_cached_response(_query, _retriever, _qa_chain):
    # Implementação
```

### 3. Cache Local em Session State

```python
# Verificar cache local
query_norm = normalize_query(query)
cached_result = st.session_state.query_cache.get(query_norm)
```

### 4. Normalização de Consultas

```python
def normalize_query(query: str) -> str:
    # Remover espaços extras e converter para minúsculas
    return query.lower().strip()
```

## Fluxo de Execução

1. **Inicialização**:
   - Carregamento do PDF
   - Divisão em chunks
   - Criação/carregamento do vectorstore
   - Configuração do retriever e do modelo LLM

2. **Processamento de Consulta**:
   - Verificação de cache local
   - Normalização da consulta
   - Recuperação de documentos relevantes
   - Geração da resposta pelo modelo LLM
   - Armazenamento em cache

3. **Apresentação**:
   - Exibição da resposta
   - Exibição dos documentos fonte
   - Registro no histórico de chat
   - Exibição do tempo de resposta

## Diagrama de Arquitetura

```
┌─────────────────┐     ┌───────────────┐     ┌──────────────────┐
│                 │     │               │     │                  │
│  PDF Document   ├────►│ Text Splitter ├────►│ Document Chunks  │
│                 │     │               │     │                  │
└─────────────────┘     └───────────────┘     └──────────┬───────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌───────────────┐     ┌──────────────────┐
│                 │     │               │     │                  │
│  User Query     ├────►│   Retriever   │◄────┤    Vectorstore   │
│                 │     │               │     │                  │
└─────────┬───────┘     └───────┬───────┘     └──────────▲───────┘
          │                     │                        │
          │                     ▼                        │
┌─────────▼───────┐     ┌───────────────┐     ┌──────────┴───────┐
│                 │     │               │     │                  │
│  LLM (Ollama)   │◄────┤ Context + Query├────┤    Embeddings    │
│                 │     │               │     │                  │
└─────────┬───────┘     └───────────────┘     └──────────────────┘
          │
          ▼
┌─────────────────┐
│                 │
│    Response     │
│                 │
└─────────────────┘
```

## Considerações de Desempenho

O sistema foi otimizado para equilibrar qualidade de resposta e velocidade:

- **Tempo de Resposta**: Tipicamente entre 2-5 segundos para consultas em cache, 5-15 segundos para novas consultas
- **Uso de Memória**: ~500MB para o vectorstore, ~2GB para o modelo Ollama
- **Precisão**: Alta fidelidade às informações contidas no Regulamento Pedagógico

## Possíveis Melhorias Futuras

1. **Implementação de Reranking**: Adicionar um passo de reranking após a recuperação inicial
2. **Modelos Mais Leves**: Experimentar com modelos menores para dispositivos com recursos limitados
3. **Processamento Assíncrono**: Implementar processamento assíncrono para melhorar a experiência do usuário
4. **Expansão de Consultas**: Adicionar expansão de consultas para melhorar a recuperação
5. **Feedback do Usuário**: Implementar mecanismo de feedback para melhorar continuamente as respostas
