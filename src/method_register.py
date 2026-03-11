"""
Method Registry - Register and retrieve different memory methods
"""

from typing import Dict, Type, Any, Optional

from src.method.base_method import BaseMethod
from src.method.bm25 import BM25Method
from src.method.embedding_mem import EmbeddingMethod
from src.method.longcontext import LongContextMethod
from src.method.ama_agent import AMAAgentMethod

import inspect


# Registry of available methods
_METHOD_REGISTRY: Dict[str, Type[BaseMethod]] = {
    "bm25": BM25Method,
    "embedding": EmbeddingMethod,
    "longcontext": LongContextMethod,
    "ama_agent": AMAAgentMethod,
}


def register_method(name: str, method_class: Type[BaseMethod]) -> None:
    """
    Register a new method.

    Args:
        name: Name of the method
        method_class: Class implementing BaseMethod interface
    """
    if not issubclass(method_class, BaseMethod):
        raise ValueError(f"Method class must inherit from BaseMethod, got {method_class}")

    _METHOD_REGISTRY[name] = method_class
    print(f"✅ Registered method: {name}")


def get_method(name: str, **kwargs) -> BaseMethod:
    """
    Get a method instance by name.

    Args:
        name: Name of the method
        **kwargs: Additional arguments to pass to the method constructor

    Returns:
        Instance of the requested method

    Raises:
        ValueError: If method name is not registered
    """
    if name not in _METHOD_REGISTRY:
        available = ", ".join(_METHOD_REGISTRY.keys())
        raise ValueError(f"Method '{name}' not found. Available methods: {available}")

    method_class = _METHOD_REGISTRY[name]

    # Filter kwargs based on method's __init__ signature
    
    init_params = inspect.signature(method_class.__init__).parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in init_params}

    return method_class(**filtered_kwargs)


def list_methods() -> list:
    """
    List all registered methods.

    Returns:
        List of method names
    """
    return list(_METHOD_REGISTRY.keys())


def get_method_class(name: str) -> Type[BaseMethod]:
    """
    Get the method class by name without instantiating.

    Args:
        name: Name of the method

    Returns:
        Method class

    Raises:
        ValueError: If method name is not registered
    """
    if name not in _METHOD_REGISTRY:
        available = ", ".join(_METHOD_REGISTRY.keys())
        raise ValueError(f"Method '{name}' not found. Available methods: {available}")

    return _METHOD_REGISTRY[name]
