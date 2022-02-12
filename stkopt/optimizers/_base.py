import warnings
import functools
from numbers import Real
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

from ..test_functions import OptTestFunction

T_WrapperMode = Literal['common', 'vectorization', 'cached', 'multithreading', 'multiprocessing']
T_SequenceReal = Union[Sequence[Real], np.ndarray]


class OptimizerBase(ABC):
    def __init__(self, func: Callable, ndim: int = None, lb: T_SequenceReal = None, ub: T_SequenceReal = None,
                 func_wrapper_mode: T_WrapperMode = 'common', func_wrapper_options: Optional[Dict[str, Any]] = None,
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
        self.best_position: Optional[T_SequenceReal] = None

        self.log_func_call_history = log_func_call_history
        self.func_call_history: List[Tuple[Real, T_SequenceReal]] = list()
        self.log_iter_history = log_iter_history
        self.iter_history: List[Tuple[Real, T_SequenceReal]] = list()

        self.func = self._func_wrapper(func, mode=func_wrapper_mode, options=func_wrapper_options)

    def _func_wrapper(self, func: Callable, mode: T_WrapperMode, options: Optional[Dict[str, Any]] = None):
        func_wrapper_modes = ('common', 'vectorization', 'cached', 'multithreading', 'multiprocessing')
        if mode not in func_wrapper_modes:
            raise ValueError(f'mode must in {func_wrapper_modes!r}')

        if options is None:
            options = {}

        if mode == 'vectorization':
            @functools.wraps(func)
            def wrapper(xx):
                if self.log_func_call_history and self.best_value is not None:
                    self.func_call_history.append((self.best_value, self.best_position))
                return func(xx.T)

            return wrapper
        elif mode == 'cached':
            @functools.wraps(func)
            @functools.lru_cache(**options)
            def wrapper(xx):
                if self.log_func_call_history and self.best_value is not None:
                    self.func_call_history.append((self.best_value, self.best_position))
                return func(xx)

            return lambda x: np.array(list(map(lambda xx: wrapper(tuple(xx)), x)))
        elif mode == 'multiprocessing':
            if self.log_func_call_history:
                warnings.warn('in multiprocessing mode, log_func_call_history will not work')
            executor = ProcessPoolExecutor(**options)
            return lambda x: np.array(list(executor.map(func, x)))
        else:
            @functools.wraps(func)
            def wrapper(xx):
                if self.log_func_call_history and self.best_value is not None:
                    self.func_call_history.append((self.best_value, self.best_position))
                return func(xx)

            if mode == 'multithreading':
                executor = ThreadPoolExecutor(**options)
                return lambda x: np.array(list(executor.map(wrapper, x)))
            else:
                return lambda x: np.array(list(map(wrapper, x)))

    def update_best(self, best_value: Real, best_position: T_SequenceReal):
        self.best_value = best_value
        self.best_position = best_position

    def log_history(self):
        if self.log_iter_history:
            self.iter_history.append((self.best_value, self.best_position))

    def run(self, *args, **kwargs) -> Tuple[Real, T_SequenceReal]:
        for _ in self.run_iter(*args, **kwargs):
            pass
        return self.best_value, self.best_position

    @abstractmethod
    def run_iter(self, *args, **kwargs) -> Iterator[Tuple[Real, T_SequenceReal]]:
        pass
