<div align="center">
  <a href="https://github.com/lucasmaciel03/ipvc-rag-ollama">
    <img src="https://ollama.com/public/ollama.png" alt="Logo" width="auto" height="auto">
  </a>

  <h1 align="center">ESTG RegPedagogicoBot</h1>

  <p align="center">
    Sistema RAG com Ollama LLM para responder perguntas sobre o Regulamento Pedagógico da ESTG
    <br />
    <a href="docs/ARCHITECTURE.md"><strong>Explorar a documentação »</strong></a>
    <br />
    <br />
    <a href="#demonstração">Ver Demonstração</a>
    ·
    <a href="https://github.com/lucasmaciel03/ipvc-rag-ollama/issues">Reportar Bug</a>
    ·
    <a href="https://github.com/lucasmaciel03/ipvc-rag-ollama/issues">Solicitar Funcionalidade</a>
  </p>
</div>

<!-- ÍNDICE -->
<details>
  <summary>Índice</summary>
  <ol>
    <li>
      <a href="#sobre-o-projeto">Sobre o Projeto</a>
      <ul>
        <li><a href="#tecnologias-utilizadas">Tecnologias Utilizadas</a></li>
      </ul>
    </li>
    <li>
      <a href="#começando">Começando</a>
      <ul>
        <li><a href="#pré-requisitos">Pré-requisitos</a></li>
        <li><a href="#instalação">Instalação</a></li>
      </ul>
    </li>
    <li><a href="#uso">Uso</a></li>
    <li><a href="#arquitetura-rag">Arquitetura RAG</a></li>
    <li><a href="#otimizações">Otimizações</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#documentação">Documentação</a></li>
    <li><a href="#contribuição">Contribuição</a></li>
    <li><a href="#licença">Licença</a></li>
    <li><a href="#contato">Contato</a></li>
    <li><a href="#agradecimentos">Agradecimentos</a></li>
  </ol>
</details>

<!-- SOBRE O PROJETO -->
## Sobre o Projeto

Este projeto implementa um sistema de Retrieval-Augmented Generation (RAG) utilizando o modelo LLM local **Ollama** para responder a perguntas sobre o **Regulamento Pedagógico da ESTG**. O sistema foi desenvolvido como parte do Trabalho Prático 3 da disciplina de Inteligência Artificial do curso de Engenharia Informática do IPVC.

O projeto adapta a plataforma UROBOT para utilizar o modelo LLM local Ollama em vez do ChatGPT, aplicando a técnica de Retrieval-Augmented Generation (RAG) para melhorar a precisão das respostas sobre o Regulamento Pedagógico da ESTG. A interface web foi desenvolvida com Streamlit, proporcionando uma experiência interativa e amigável.

Principais características:
* Interface web interativa desenvolvida com Streamlit
* Processamento de PDF e divisão em chunks otimizados
* Embeddings e vectorstore local com ChromaDB
* Pipeline RAG completo com LangChain
* Sistema de cache multi-camada para performance otimizada
* Documentação detalhada para uso e desenvolvimento

<p align="right">(<a href="#top">voltar ao topo</a>)</p>

### Tecnologias Utilizadas

O projeto foi desenvolvido utilizando as seguintes tecnologias:

* ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
* ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
* ![LangChain](https://img.shields.io/badge/LangChain-121212?style=for-the-badge&logo=chainlink&logoColor=white)
* ![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=llama&logoColor=white)
* ![ChromaDB](https://img.shields.io/badge/ChromaDB-4285F4?style=for-the-badge&logo=google-chrome&logoColor=white)

<p align="right">(<a href="#top">voltar ao topo</a>)</p>

<!-- COMEÇANDO -->
## Começando

Para obter uma cópia local em funcionamento, siga estas etapas simples.

### Pré-requisitos

* Python 3.10+
* Ollama (servidor local de LLM)
* Dependências Python listadas em `requirements.txt`:
  ```
  langchain>=0.1.0
  langchain-community>=0.1.0
  langchain-ollama>=0.1.0
  chromadb>=0.4.22
  pypdf>=3.17.0
  streamlit>=1.30.0
  pydantic>=2.5.0
  python-dotenv>=1.0.0
  tqdm>=4.66.0
  ```

### Instalação

1. Clone o repositório
   ```sh
   git clone https://github.com/lucasmaciel03/ipvc-rag-ollama.git
   cd ipvc-rag-ollama
   ```
2. Crie e ative um ambiente virtual
   ```sh
   python -m venv streamlit_env
   source streamlit_env/bin/activate  # No Windows: streamlit_env\Scripts\activate
   ```
3. Instale as dependências
   ```sh
   pip install -r requirements.txt
   ```
4. Instale e inicie o Ollama
   ```sh
   # Instale o Ollama seguindo as instruções em ollama.com/download
   ollama serve
   ```
5. Em outro terminal, baixe os modelos necessários
   ```sh
   ollama pull llama3
   ollama pull nomic-embed-text
   ```

Para instruções de instalação mais detalhadas, consulte o [Guia de Instalação](docs/INSTALLATION.md).

<p align="right">(<a href="#top">voltar ao topo</a>)</p>

<!-- USO -->
## Uso

### Iniciar a Aplicação

```bash
# Ativar o ambiente virtual (se ainda não estiver ativado)
source streamlit_env/bin/activate

# Executar a aplicação Streamlit
streamlit run app_refactored.py
```

A interface web estará disponível em `http://localhost:8501`

### Usando a Interface

1. Clique em **Iniciar Sistema RAG** na barra lateral
2. Aguarde o carregamento do PDF e a criação do vectorstore
3. Digite sua pergunta sobre o Regulamento Pedagógico da ESTG
4. Clique em **Enviar Pergunta** ou pressione Enter
5. Veja a resposta e os documentos fonte utilizados

### Exemplos de Perguntas

- "Quais são os tipos de avaliação previstos no regulamento?"
- "Como funciona a época especial de exames?"
- "Quais são as condições para obter o estatuto de estudante-atleta?"
- "Qual o prazo para revisão de provas?"

Para instruções de uso mais detalhadas e dicas de otimização, consulte o [Guia de Uso](docs/USAGE_GUIDE.md).

<p align="right">(<a href="#top">voltar ao topo</a>)</p>

<!-- ARQUITETURA RAG -->
## Arquitetura RAG

O sistema utiliza a arquitetura de Retrieval-Augmented Generation (RAG) para melhorar a qualidade das respostas do modelo LLM:

1. **Processamento de Documentos**:
   - Carregamento do PDF do Regulamento Pedagógico
   - Divisão em chunks de texto (800 caracteres com 80 de sobreposição)

2. **Criação de Embeddings**:
   - Geração de embeddings usando o modelo `nomic-embed-text` do Ollama
   - Armazenamento em um vectorstore ChromaDB persistente

3. **Pipeline de Recuperação**:
   - Busca semântica para encontrar os 2 chunks mais relevantes para a pergunta
   - Recuperação dos documentos para fornecer contexto ao LLM

4. **Geração de Respostas**:
   - Modelo Ollama `llama3` recebe a pergunta e o contexto recuperado
   - Prompt personalizado em português para instruir o modelo

Para uma explicação detalhada do conceito RAG e sua implementação, consulte [RAG Explicado](docs/RAG_EXPLAINED.md) e [Arquitetura do Sistema](docs/ARCHITECTURE.md).

<p align="right">(<a href="#top">voltar ao topo</a>)</p>

<!-- OTIMIZAÇÕES -->
## Otimizações

O sistema implementa várias otimizações para melhorar o desempenho:

### Otimizações de Performance

- **Sistema de Cache Multi-camada**:
  - Cache do vectorstore (TTL: 1 hora)
  - Cache de respostas (TTL: 30 minutos)
  - Cache local em session_state
  - Normalização de consultas para melhorar hits de cache

- **Configuração do Modelo**:
  - Temperatura baixa (0.1) para respostas mais consistentes
  - Contexto reduzido para processamento mais rápido
  - Uso de múltiplas threads e GPU quando disponível

- **Processamento de Documentos**:
  - Tamanho de chunk otimizado para equilibrar precisão e velocidade
  - Recuperação de apenas 2 documentos mais relevantes

### Melhorias de UX

- **Interface Responsiva**:
  - Feedback visual durante o processamento
  - Estilização CSS para melhor legibilidade

- **Funcionalidades Úteis**:
  - Histórico de chat
  - Exemplos de perguntas pré-definidas
  - Exibição de documentos fonte
  - Tempo de resposta

<p align="right">(<a href="#top">voltar ao topo</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [x] Interface web interativa com Streamlit
- [x] Implementação do pipeline RAG completo
- [x] Sistema de cache multi-camada
- [x] Documentação detalhada
- [ ] Implementação de testes automatizados
- [ ] Suporte a múltiplos documentos
- [ ] Interface de administração para gerenciar documentos
- [ ] Avaliação de qualidade das respostas

Veja os [issues abertos](https://github.com/lucasmaciel03/ipvc-rag-ollama/issues) para uma lista completa de funcionalidades propostas (e problemas conhecidos).

<p align="right">(<a href="#top">voltar ao topo</a>)</p>

<!-- DOCUMENTAÇÃO -->
## Documentação

O projeto inclui documentação detalhada para ajudar usuários e desenvolvedores:

1. **[Guia de Instalação](docs/INSTALLATION.md)**:
   - Instruções passo a passo para configurar o ambiente
   - Solução de problemas comuns de instalação
   - Configurações opcionais para melhor desempenho

2. **[Guia de Uso](docs/USAGE_GUIDE.md)**:
   - Como utilizar a interface Streamlit
   - Exemplos detalhados de perguntas eficazes
   - Dicas para otimizar o desempenho do sistema
   - Perguntas frequentes (FAQ)

3. **[Arquitetura do Sistema](docs/ARCHITECTURE.md)**:
   - Explicação detalhada dos componentes do sistema
   - Diagrama de arquitetura do pipeline RAG
   - Descrição do fluxo de dados e processamento
   - Considerações de desempenho e melhorias futuras

4. **[RAG Explicado](docs/RAG_EXPLAINED.md)**:
   - Conceito de Retrieval-Augmented Generation
   - Comparação com outras abordagens de IA
   - Implementação específica no projeto
   - Desafios e soluções

<p align="right">(<a href="#top">voltar ao topo</a>)</p>

<!-- CONTRIBUIÇÃO -->
## Contribuição

Contribuições são o que tornam a comunidade open source um lugar incrível para aprender, inspirar e criar. Qualquer contribuição que você fizer será **muito apreciada**.

Se você tiver uma sugestão para melhorar isso, por favor faça um fork do repositório e crie um pull request. Você também pode simplesmente abrir um issue com a tag "melhoria".
Não se esqueça de dar uma estrela ao projeto! Obrigado novamente!

1. Faça um Fork do Projeto
2. Crie sua Branch de Funcionalidade (`git checkout -b feature/FuncionalidadeIncrivel`)
3. Faça Commit de suas Alterações (`git commit -m 'Adiciona alguma FuncionalidadeIncrivel'`)
4. Faça Push para a Branch (`git push origin feature/FuncionalidadeIncrivel`)
5. Abra um Pull Request

<p align="right">(<a href="#top">voltar ao topo</a>)</p>

<!-- LICENÇA -->
## Licença

Distribuído sob a Licença MIT. Veja `LICENSE` para mais informações.

<p align="right">(<a href="#top">voltar ao topo</a>)</p>

<!-- CONTATO -->
## Contato

Lucas Maciel - [@lucasmaciel03](https://github.com/lucasmaciel03) - lucasmaciel@ipvc.pt

Link do Projeto: [https://github.com/lucasmaciel03/ipvc-rag-ollama](https://github.com/lucasmaciel03/ipvc-rag-ollama)

<p align="right">(<a href="#top">voltar ao topo</a>)</p>

<!-- AGRADECIMENTOS -->
## Agradecimentos

* [Ollama](https://ollama.com/) - Por fornecer uma forma fácil de executar LLMs localmente
* [LangChain](https://www.langchain.com/) - Pelo framework que facilita a criação de aplicações com LLMs
* [Streamlit](https://streamlit.io/) - Pela ferramenta que permite criar interfaces web rapidamente
* [IPVC ESTG](https://estg.ipvc.pt/) - Pela oportunidade de desenvolver este projeto
* [Best-README-Template](https://github.com/othneildrew/Best-README-Template) - Pelo template utilizado neste README

<p align="right">(<a href="#top">voltar ao topo</a>)</p>

<!-- ESTRUTURA DO PROJETO -->
## Estrutura do Projeto

```
.
├── app.py                  # Aplicação Streamlit principal
├── app_refactored.py       # Versão refatorada da aplicação
├── requirements.txt        # Dependências do projeto
├── resources/              # PDFs e outros recursos
│   └── ESTG_Regulamento-Frequencia-Avaliacao2023.pdf
├── src/                    # Código fonte modularizado
│   ├── __init__.py
│   ├── config/             # Configurações do sistema
│   ├── data/               # Processamento de documentos
│   ├── models/             # Modelos e embeddings
│   └── utils/              # Utilitários e helpers
├── docs/                   # Documentação detalhada
│   ├── ARCHITECTURE.md     # Arquitetura do sistema RAG
│   ├── INSTALLATION.md     # Guia de instalação detalhado
│   ├── USAGE_GUIDE.md      # Guia de uso e otimização
│   └── RAG_EXPLAINED.md    # Explicação do conceito RAG
├── urobot/                 # Código original do UROBOT
└── vector_store/          # Armazenamento de embeddings (ChromaDB)
```

<p align="right">(<a href="#top">voltar ao topo</a>)</p>
