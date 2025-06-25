# Retrieval-Augmented Generation (RAG) Explicado

Este documento explica o conceito de Retrieval-Augmented Generation (RAG) e como ele é implementado no projeto ESTG RegPedagogicoBot.

## O que é RAG?

Retrieval-Augmented Generation (RAG) é uma técnica que combina sistemas de recuperação de informação com modelos de linguagem generativa para produzir respostas mais precisas, atualizadas e fundamentadas em fontes específicas.

![Diagrama RAG](https://miro.medium.com/v2/resize:fit:1400/1*f7X4gAl9tKnI_uJvSJ9kpg.png)

### Problema que o RAG Resolve

Os modelos de linguagem grandes (LLMs) como o Ollama, GPT-4 e outros têm limitações importantes:

1. **Conhecimento Estático**: O conhecimento do modelo é limitado aos dados de treinamento
2. **Alucinações**: Tendência a gerar informações incorretas com confiança
3. **Falta de Citações**: Incapacidade de citar fontes específicas
4. **Conhecimento Desatualizado**: Não têm acesso a informações posteriores ao treinamento

O RAG resolve esses problemas ao permitir que o modelo acesse informações externas atualizadas e específicas antes de gerar respostas.

## Como Funciona o RAG?

O RAG opera em duas etapas principais:

### 1. Etapa de Recuperação (Retrieval)

Quando uma pergunta é feita:

1. **Indexação de Documentos**: Documentos são divididos em chunks e convertidos em embeddings vetoriais
2. **Busca Semântica**: A pergunta é convertida em um embedding e comparada com os documentos indexados
3. **Recuperação Relevante**: Os documentos mais similares semanticamente são recuperados

### 2. Etapa de Geração (Generation)

Após a recuperação:

1. **Contextualização**: Os documentos recuperados são fornecidos como contexto ao LLM
2. **Geração Informada**: O LLM gera uma resposta baseada na pergunta e no contexto fornecido
3. **Citação de Fontes**: A resposta pode incluir referências aos documentos originais

## Implementação no ESTG RegPedagogicoBot

No nosso projeto, implementamos o RAG da seguinte forma:

### Processamento de Documentos

```python
# Carregar o PDF do Regulamento Pedagógico
loader = PyPDFLoader("resources/ESTG_Regulamento-Frequencia-Avaliacao2023.pdf")
documents = loader.load()

# Dividir em chunks para processamento eficiente
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=80,
    length_function=len
)
chunks = text_splitter.split_documents(documents)
```

### Criação de Embeddings e Vectorstore

```python
# Gerar embeddings usando Ollama
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Armazenar em um vectorstore ChromaDB
vectorstore = Chroma.from_documents(
    chunks, 
    embedding=embeddings, 
    persist_directory="vector_store"
)
vectorstore.persist()
```

### Configuração do Retriever

```python
# Configurar o retriever para buscar documentos relevantes
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}  # Recupera os 2 documentos mais relevantes
)
```

### Configuração do Modelo e Prompt

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

# Criar prompt personalizado em português
prompt_template = """
Você é um assistente especializado no Regulamento Pedagógico da ESTG.
Responda à pergunta em PORTUGUÊS com base nas informações fornecidas abaixo.
Se a informação não estiver presente nos documentos fornecidos, diga que não tem informações suficientes para responder.

Contexto:
{context}

Pergunta: {question}

Resposta em português:
"""
```

### Pipeline RAG Completo

```python
# Criar a cadeia RAG
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# Processar uma pergunta
result = qa_chain.invoke({"query": pergunta})
```

## Vantagens do RAG no Nosso Projeto

A implementação do RAG no ESTG RegPedagogicoBot oferece várias vantagens:

1. **Respostas Precisas**: As respostas são baseadas diretamente no Regulamento Pedagógico da ESTG
2. **Transparência**: O sistema mostra os trechos específicos do regulamento usados para gerar a resposta
3. **Conhecimento Especializado**: O modelo pode responder perguntas específicas sobre o regulamento, mesmo que não tenha sido treinado com esses dados
4. **Redução de Alucinações**: Ao basear respostas em documentos recuperados, reduzimos significativamente as alucinações do modelo
5. **Adaptabilidade**: O sistema pode ser facilmente adaptado para outros documentos ou regulamentos

## Otimizações no RAG

Nosso sistema implementa várias otimizações para melhorar o desempenho do RAG:

### 1. Tamanho de Chunk Otimizado

Após testes extensivos, determinamos que chunks de 800 caracteres com 80 de sobreposição oferecem o melhor equilíbrio entre precisão e desempenho.

### 2. Número de Documentos Recuperados

Limitamos a recuperação para apenas 2 documentos mais relevantes, o que:
- Reduz o ruído no contexto
- Acelera o processamento
- Mantém o contexto conciso e relevante

### 3. Prompt Especializado

Nosso prompt foi cuidadosamente projetado para:
- Instruir o modelo a responder em português
- Limitar as respostas às informações fornecidas
- Indicar quando não há informações suficientes

### 4. Sistema de Cache Multi-camada

Implementamos um sistema de cache em múltiplas camadas para melhorar o desempenho:
- Cache do vectorstore
- Cache de respostas
- Cache local em session_state
- Normalização de consultas

## Comparação com Outras Abordagens

| Abordagem | Vantagens | Desvantagens |
|-----------|-----------|--------------|
| **LLM Puro** | Simples de implementar | Conhecimento limitado, alucinações |
| **Fine-tuning** | Conhecimento incorporado no modelo | Caro, difícil de atualizar |
| **RAG** | Flexível, atualizável, preciso | Requer infraestrutura adicional |
| **RAG + Cache** (nossa abordagem) | Rápido, preciso, eficiente | Complexidade adicional |

## Desafios e Soluções

### Desafio 1: Tamanho do Contexto

**Problema**: Contexto muito grande pode exceder os limites do modelo e tornar o processamento lento.

**Solução**: Otimizamos o tamanho dos chunks e limitamos o número de documentos recuperados.

### Desafio 2: Qualidade dos Embeddings

**Problema**: Embeddings de baixa qualidade podem levar a recuperação irrelevante.

**Solução**: Utilizamos o modelo `nomic-embed-text` do Ollama, que oferece embeddings de alta qualidade.

### Desafio 3: Tempo de Resposta

**Problema**: O RAG pode ser mais lento que um LLM puro devido às etapas adicionais.

**Solução**: Implementamos um sistema de cache multi-camada e otimizamos os parâmetros do modelo.

## Conclusão

O Retrieval-Augmented Generation (RAG) é uma técnica poderosa que melhora significativamente a qualidade, precisão e confiabilidade das respostas geradas por modelos de linguagem. No ESTG RegPedagogicoBot, o RAG permite que o sistema forneça respostas precisas e fundamentadas sobre o Regulamento Pedagógico da ESTG, com transparência sobre as fontes utilizadas.

## Recursos Adicionais

- [LangChain RAG Documentation](https://python.langchain.com/docs/use_cases/question_answering/)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [RAG Paper (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
