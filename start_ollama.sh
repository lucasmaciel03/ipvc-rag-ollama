#!/bin/bash

if ! pgrep -x "ollama" > /dev/null
then
    echo "A iniciar o servidor Ollama..."
    ollama serve &
    sleep 3
fi

ollama run llama3
