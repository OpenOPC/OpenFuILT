##
# @file   semantic.py
# @author Shuo Yin
#
from typing import List
import torch
from collections.abc import Callable

from typing import TypeVar, Generic
T = TypeVar('T')

# =================JIT=================#

class JITFunction(Generic[T]):
    def __init__(self,
                 fn : T,
                initializer : Callable,
                binarizer: Callable,
                **kwargs
                 ) -> None:
        
        self.initializer = initializer
        self.binarizer = binarizer
        self.fn = fn
        self.kwargs = kwargs
        
    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

def opc(
        initializer: Callable = None,
        binarizer: Callable = None,
        **kwargs
        ):
    def decorator(fn):
        assert callable(fn)
        return JITFunction(
            fn,
            initializer,
            binarizer,
            kwargs=kwargs
        )
    return decorator
    
#=================OPTIMIZER=================#

def SGD():
    return torch.optim.SGD

def Adam():
    return torch.optim.Adam