from numbers import Real
from abc import ABC, abstractmethod
from typing import Callable, Sequence

from ..test_functions import OptTestFunction


class OptimizersBase(ABC):
    def __init__(self, func: Callable, ndim: int = None, lb: Sequence[Real] = None, ub: Sequence[Real] = None):
        if ndim is None and isinstance(func, OptTestFunction):
            ndim = func.ndim
        if lb is None and isinstance(func, OptTestFunction):
            lb = func.lb
        if ub is None and isinstance(func, OptTestFunction):
            ub = func.ub

        self.ndim = ndim
        self.lb = lb
        self.ub = ub

    @abstractmethod
    def fit(self):
        pass
