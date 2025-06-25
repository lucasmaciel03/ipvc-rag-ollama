# Guia de Uso e Otimização

Este documento fornece instruções detalhadas sobre como utilizar o sistema ESTG RegPedagogicoBot e como otimizar seu desempenho.

## Índice

- [Iniciando a Aplicação](#iniciando-a-aplicação)
- [Usando a Interface](#usando-a-interface)
- [Exemplos de Perguntas](#exemplos-de-perguntas)
- [Otimizações de Desempenho](#otimizações-de-desempenho)
- [Solução de Problemas](#solução-de-problemas)
- [Perguntas Frequentes](#perguntas-frequentes)

## Iniciando a Aplicação

### Pré-requisitos

Antes de iniciar a aplicação, certifique-se de que:

1. O ambiente virtual está ativado:
   ```bash
   source streamlit_env/bin/activate  # No Windows: streamlit_env\Scripts\activate
   ```

2. O servidor Ollama está em execução:
   ```bash
   ollama serve
   ```

### Iniciar o Streamlit

Para iniciar a aplicação, execute:

```bash
# Versão original
streamlit run app.py

# OU versão refatorada (recomendada)
streamlit run app_refactored.py
```

A interface web estará disponível em `http://localhost:8501`

## Usando a Interface

### 1. Inicialização do Sistema RAG

Ao abrir a aplicação pela primeira vez:

1. Clique no botão **Iniciar Sistema RAG** na barra lateral
2. Aguarde o carregamento do PDF e a criação do vectorstore
   - Este processo pode levar alguns segundos na primeira execução
   - O sistema exibirá mensagens de progresso durante o carregamento

### 2. Fazendo Perguntas

Para fazer perguntas sobre o Regulamento Pedagógico da ESTG:

1. Digite sua pergunta na caixa de texto
2. Clique no botão **Enviar Pergunta** ou pressione Enter
3. Aguarde enquanto o sistema processa a pergunta
   - O tempo de resposta varia de 2-15 segundos dependendo da complexidade
   - Perguntas repetidas são respondidas mais rapidamente devido ao cache

### 3. Visualizando Respostas

Após o processamento, você verá:

- A resposta gerada pelo modelo
- O tempo de resposta em segundos
- Um expansor **Ver documentos fonte** que mostra os trechos do regulamento utilizados
- O histórico de perguntas e respostas anteriores (expansível)

### 4. Utilizando Exemplos Pré-definidos

Para facilitar o uso, a barra lateral contém exemplos de perguntas:

1. Clique em qualquer exemplo para preencher automaticamente a caixa de pergunta
2. A pergunta será preenchida e você pode enviar normalmente

### 5. Recriando o Vectorstore

Se necessário, você pode recriar o vectorstore:

1. Marque a opção **Recriar Vectorstore** na barra lateral
2. Clique em **Iniciar Sistema RAG**
3. Aguarde a recriação do vectorstore
   - Útil se o PDF do regulamento foi atualizado
   - Ou se o vectorstore estiver corrompido

## Exemplos de Perguntas

Aqui estão alguns exemplos de perguntas eficazes para testar o sistema:

### Perguntas Básicas

- "O que é o Regulamento Pedagógico da ESTG?"
- "Quais são os tipos de avaliação previstos no regulamento?"
- "Quando ocorrem as épocas de exame?"

### Perguntas Específicas

- "Quais são os requisitos para obter o estatuto de estudante-atleta?"
- "Como funciona a época especial de exames?"
- "Qual o prazo para revisão de provas?"
- "Quais são as condições para aprovação em uma unidade curricular?"

### Perguntas Complexas

- "Explique o processo de avaliação contínua e suas diferenças em relação à avaliação final."
- "Quais são os direitos dos estudantes com necessidades educativas especiais?"
- "Como funciona o processo de melhoria de nota e quais são as restrições?"

## Otimizações de Desempenho

O sistema já implementa várias otimizações, mas você pode melhorar ainda mais o desempenho:

### Ajustes de Configuração

Edite o arquivo `src/config/settings.py` para ajustar:

```python
# Configurações do modelo Ollama
OLLAMA_TEMPERATURE = 0.1        # Reduzir para respostas mais determinísticas
OLLAMA_NUM_CTX = 2048           # Reduzir para processamento mais rápido
OLLAMA_NUM_THREAD = 4           # Aumentar se tiver mais núcleos disponíveis
OLLAMA_NUM_GPU = 1              # Definir como 0 se não tiver GPU

# Configurações de processamento de documentos
CHUNK_SIZE = 800                # Ajustar conforme necessidade
CHUNK_OVERLAP = 80              # Reduzir para menos chunks

# Configurações do retriever
RETRIEVER_K = 2                 # Aumentar para mais contexto, reduzir para mais velocidade

# Cache
CACHE_TTL_VECTORSTORE = 3600    # Ajustar conforme necessidade
CACHE_TTL_RESPONSES = 1800      # Ajustar conforme necessidade
```

### Otimizações Extremas

Para casos onde a velocidade é crítica, você pode aplicar otimizações extremas:

```python
# Configurações extremamente otimizadas para velocidade
OLLAMA_TEMPERATURE = 0.0        # Sem aleatoriedade
OLLAMA_TOP_P = 0.5              # Reduzir para aumentar velocidade
OLLAMA_NUM_CTX = 1024           # Contexto muito menor
OLLAMA_NUM_THREAD = 8           # Maximizar threads
OLLAMA_TIMEOUT = 10             # Timeout para evitar esperas longas

# Processamento de documentos otimizado para velocidade
CHUNK_SIZE = 1000               # Chunks maiores = menos chunks
CHUNK_OVERLAP = 50              # Overlap mínimo

# Retriever otimizado
RETRIEVER_K = 1                 # Apenas o documento mais relevante
```

### Prompt Simplificado

Para respostas mais rápidas, você pode simplificar o prompt:

```python
# Prompt extremamente simplificado
prompt_template = """
Responda em PORTUGUÊS de forma BREVE e DIRETA usando APENAS estas informações:
{context}

Pergunta: {question}
Resposta concisa:
"""
```

## Solução de Problemas

### Problemas Comuns e Soluções

| Problema | Possível Causa | Solução |
|----------|----------------|---------|
| Erro de conexão com Ollama | Servidor Ollama não está rodando | Execute `ollama serve` em um terminal separado |
| Respostas muito lentas | Configurações não otimizadas | Ajuste os parâmetros conforme sugerido acima |
| Erro ao carregar o PDF | Arquivo não encontrado ou corrupto | Verifique se o PDF está em `resources/` |
| Erro "UnhashableParamError" | Parâmetros não hashable no cache | Prefixe parâmetros não hashable com underscore |
| Respostas imprecisas | Chunks muito grandes ou poucos documentos | Reduza CHUNK_SIZE e aumente RETRIEVER_K |
| Vectorstore corrompido | Problemas durante a criação | Marque "Recriar Vectorstore" e reinicie |

### Logs e Depuração

Para depurar problemas, verifique os logs do Streamlit:

```bash
# Executar com logs detalhados
streamlit run app_refactored.py --logger.level=debug
```

## Perguntas Frequentes

### O sistema funciona offline?

Sim, o sistema utiliza o Ollama localmente e não requer conexão com a internet após a instalação inicial.

### Posso usar outros modelos além do llama3?

Sim, você pode alterar o modelo no arquivo de configurações. Certifique-se de baixá-lo primeiro com `ollama pull nome-do-modelo`.

### Como adicionar mais documentos ao sistema?

Para adicionar mais documentos:
1. Adicione os PDFs à pasta `resources/`
2. Atualize o caminho no arquivo de configurações
3. Recrie o vectorstore

### O sistema suporta outros idiomas além do português?

O sistema está configurado para português, mas você pode adaptar o prompt para outros idiomas.

### Como melhorar a qualidade das respostas?

Para melhorar a qualidade:
1. Aumente o número de documentos recuperados (RETRIEVER_K)
2. Reduza o tamanho dos chunks para maior granularidade
3. Ajuste o prompt para instruções mais específicas
4. Use um modelo Ollama maior ou mais capaz

### Quanto tempo as respostas ficam em cache?

- Cache do vectorstore: 1 hora (configurável)
- Cache de respostas: 30 minutos (configurável)
- Cache local: Dura até o final da sessão Streamlit

---

Para mais informações, consulte a [documentação de arquitetura](ARCHITECTURE.md) ou o [guia de instalação](INSTALLATION.md).
