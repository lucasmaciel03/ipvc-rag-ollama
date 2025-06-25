#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utilitários para cache e otimização de performance
"""

import time
import logging
from typing import Dict, Any, Callable

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleCache:
    """
    Implementação simples de cache com TTL (Time To Live)
    """
    
    def __init__(self, ttl: int = 1800):
        """
        Inicializa o cache
        
        Args:
            ttl: Tempo de vida das entradas em segundos (padrão: 30 minutos)
        """
        self.cache = {}
        self.timestamps = {}
        self.ttl = ttl
        logger.info(f"Cache inicializado com TTL de {ttl} segundos")
    
    def get(self, key: str) -> Any:
        """
        Recupera um valor do cache se existir e não estiver expirado
        
        Args:
            key: Chave para buscar no cache
            
        Returns:
            Valor armazenado ou None se não existir ou estiver expirado
        """
        if key in self.cache:
            # Verificar se o item expirou
            if time.time() - self.timestamps[key] > self.ttl:
                logger.debug(f"Item expirado no cache: {key}")
                self._remove(key)
                return None
            
            logger.debug(f"Cache hit: {key}")
            return self.cache[key]
        
        logger.debug(f"Cache miss: {key}")
        return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Armazena um valor no cache
        
        Args:
            key: Chave para armazenar
            value: Valor a ser armazenado
        """
        self.cache[key] = value
        self.timestamps[key] = time.time()
        logger.debug(f"Item adicionado ao cache: {key}")
    
    def _remove(self, key: str) -> None:
        """
        Remove um item do cache
        
        Args:
            key: Chave a ser removida
        """
        if key in self.cache:
            del self.cache[key]
            del self.timestamps[key]
            logger.debug(f"Item removido do cache: {key}")
    
    def clear(self) -> None:
        """
        Limpa todo o cache
        """
        self.cache = {}
        self.timestamps = {}
        logger.info("Cache limpo")

def normalize_query(query: str) -> str:
    """
    Normaliza uma consulta para melhorar hits de cache
    
    Args:
        query: Consulta original
        
    Returns:
        Consulta normalizada
    """
    # Remover espaços extras e converter para minúsculas
    return query.lower().strip()

def timed_execution(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """
    Executa uma função e mede o tempo de execução
    
    Args:
        func: Função a ser executada
        *args, **kwargs: Argumentos para a função
        
    Returns:
        Dicionário com o resultado e o tempo de execução
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    execution_time = time.time() - start_time
    
    return {
        "result": result,
        "execution_time": execution_time
    }
