import warnings
import functools
from numbers import Real
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Literal, Iterator, Tuple, Optional, Sequence

import numpy as np

from ..test_functions import OptTestFunction

FUNC_WRAPPER_MODES = ('common', 'vectorization', 'cached', 'multithreading', 'multiprocessing')
FUNC_WRAPPER_MODES_TYPE = Literal['common', 'vectorization', 'cached', 'multithreading', 'multiprocessing']


class OptimizerBase(ABC):
    def __init__(self, func: Callable, ndim: int = None, lb: Sequence[Real] = None, ub: Sequence[Real] = None,
                 func_wrapper_mode: FUNC_WRAPPER_MODES_TYPE = 'common',
                 func_wrapper_options: Optional[Dict[str, Any]] = None,
                 log_iter_history: bool = True, log_func_call_history: bool = True):
        if ndim is None and isinstance(func, OptTestFunction):
            ndim = func.ndim
        if lb is None and isinstance(func, OptTestFunction):
            lb = func.lb
        if ub is None and isinstance(func, OptTestFunction):
            ub = func.ub

        self.ndim = ndim
        self.lb = np.array(lb)
        self.ub = np.array(ub)

        if not (self.ndim == len(self.lb) == len(self.ub)):
            raise ValueError('length of lb and ub must equal to ndim')
        if not np.all(self.ub > self.lb):
            raise ValueError('ub must be greater than lb')

        self.best_value: Optional[Real] = None
        self.best_position: Optional[Sequence[Real, ...]] = None

        self.log_func_call_history = log_func_call_history
        self.func_call_history: List[Sequence[Real, Sequence[Real, ...]]] = list()
        self.log_iter_history = log_iter_history
        self.iter_history: List[Sequence[Real, Sequence[Real, ...]]] = list()

        self.func = self._func_wrapper(func, mode=func_wrapper_mode, options=func_wrapper_options)

    def _func_wrapper(self, func: Callable, mode: FUNC_WRAPPER_MODES_TYPE, options: Optional[Dict[str, Any]] = None):
        if mode not in FUNC_WRAPPER_MODES:
            raise ValueError(f'mode must in {FUNC_WRAPPER_MODES!r}')

        if options is None:
            options = {}

        if mode == 'vectorization':
            def wrapper(xx):
                return func(xx.T)

            return wrapper
        elif mode == 'multiprocessing':
            if self.log_func_call_history:
                warnings.warn('in multiprocessing mode, log_func_call_history will not work')
            executor = ProcessPoolExecutor(**options)
            return lambda x: np.array(list(executor.map(func, x)))

        if mode == 'cached':
            @functools.wraps(func)
            @functools.lru_cache(**options)
            def func_cached(xx):
                return func(xx)

        @functools.wraps(func)
        def wrapper(xx):
            if self.log_func_call_history and self.best_value is not None:
                self.func_call_history.append((self.best_value, self.best_position))
            if mode == 'cached':
                return func_cached(tuple(xx))
            return func(xx)

        if mode == 'multithreading':
            executor = ThreadPoolExecutor(**options)
            return lambda x: np.array(list(executor.map(wrapper, x)))
        else:
            return lambda x: np.array(list(map(wrapper, x)))

    def update_best(self, best_value: Real, best_position: Sequence[Real]):
        if self.best_value is None or self.best_value > best_value:
            self.best_value = best_value
            self.best_position = best_position

    def log_history(self):
        if self.log_iter_history:
            self.iter_history.append((self.best_value, self.best_position))

    def run(self, *args, **kwargs) -> Tuple[Real, Sequence[Real]]:
        for _ in self.run_iter(*args, **kwargs):
            pass
        return self.best_value, self.best_position

    @abstractmethod
    def run_iter(self, *args, **kwargs) -> Iterator[Tuple[Real, Sequence[Real]]]:
        pass
