# Guia de Instalação

Este documento fornece instruções detalhadas para instalar e configurar o sistema ESTG RegPedagogicoBot.

## Requisitos de Sistema

- Sistema Operacional: Windows, macOS ou Linux
- Python 3.10 ou superior
- Mínimo de 4GB de RAM (8GB recomendado)
- Espaço em disco: pelo menos 2GB para modelos e dependências

## 1. Preparação do Ambiente

### Clonar o Repositório

```bash
# Clonar o repositório
git clone https://github.com/lucasmaciel03/ipvc-rag-ollama.git
cd ipvc-rag-ollama
```

### Configurar o Ambiente Virtual

```bash
# No macOS/Linux
python -m venv streamlit_env
source streamlit_env/bin/activate

# No Windows
python -m venv streamlit_env
streamlit_env\Scripts\activate
```

### Instalar Dependências

```bash
pip install -r requirements.txt
```

## 2. Instalar e Configurar o Ollama

### Instalar o Ollama

1. Acesse [ollama.com/download](https://ollama.com/download)
2. Baixe e instale a versão adequada para seu sistema operacional
3. Siga as instruções de instalação específicas para sua plataforma

### Iniciar o Servidor Ollama

```bash
# Iniciar o servidor Ollama
ollama serve
```

O servidor Ollama deve estar em execução sempre que você utilizar o sistema.

### Baixar os Modelos Necessários

Em um novo terminal (mantendo o servidor Ollama em execução):

```bash
# Baixar o modelo principal para geração de texto
ollama pull llama3

# Baixar o modelo para embeddings
ollama pull nomic-embed-text
```

## 3. Verificar a Instalação

Para verificar se tudo está funcionando corretamente:

```bash
# Testar a conexão com o Ollama
curl http://localhost:11434/api/generate -d '{
  "model": "llama3",
  "prompt": "Hello, world!"
}'
```

Se você receber uma resposta do servidor Ollama, a instalação foi bem-sucedida.

## 4. Configurações Adicionais (Opcional)

### Configurar GPU (se disponível)

O Ollama utilizará automaticamente a GPU se estiver disponível. Para verificar:

```bash
# Verificar se a GPU está sendo utilizada
ollama list
```

### Ajustar Parâmetros de Desempenho

Se necessário, você pode ajustar os parâmetros do modelo no arquivo `src/config/settings.py`:

```python
# Configurações do modelo Ollama
OLLAMA_MODEL = "llama3"
OLLAMA_EMBEDDINGS_MODEL = "nomic-embed-text"
OLLAMA_TEMPERATURE = 0.1
OLLAMA_TOP_P = 0.9
OLLAMA_NUM_CTX = 2048
OLLAMA_NUM_THREAD = 4
OLLAMA_NUM_GPU = 1
```

## Solução de Problemas

### Erro de Conexão com o Ollama

Se você encontrar erros de conexão:

1. Verifique se o servidor Ollama está em execução
2. Confirme que a porta 11434 está disponível
3. Reinicie o servidor Ollama

### Erro ao Carregar o PDF

Se houver problemas ao carregar o PDF:

1. Verifique se o arquivo está presente em `resources/`
2. Confirme que o pacote `pypdf` está instalado
3. Tente reinstalar com `pip install pypdf --upgrade`

### Problemas de Memória

Se o sistema estiver consumindo muita memória:

1. Reduza o tamanho do contexto (`OLLAMA_NUM_CTX`) no arquivo de configurações
2. Diminua o número de threads (`OLLAMA_NUM_THREAD`)
3. Feche outros aplicativos que consomem muita memória

---

Para mais informações, consulte a [documentação oficial do Ollama](https://github.com/ollama/ollama) ou abra uma issue no repositório do projeto.
