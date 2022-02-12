from numbers import Real
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Tuple

import numpy as np

from ._base import OptimizerBase, T_WrapperMode, T_SequenceReal


class PSO(OptimizerBase):
    def __init__(self, func: Callable, ndim: int = None, lb: T_SequenceReal = None, ub: T_SequenceReal = None,
                 func_wrapper_mode: T_WrapperMode = 'common', func_wrapper_options: Optional[Dict[str, Any]] = None,
                 log_iter_history: bool = True, log_func_call_history: bool = True,
                 pop_size: int = 40, max_iter: int = 150, w: Real = 0.8, c1: Real = 0.5, c2: Real = 0.5,
                 v_max: Optional[T_SequenceReal] = None):
        super().__init__(func, ndim, lb, ub, func_wrapper_mode, func_wrapper_options, log_iter_history,
                         log_func_call_history)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        if v_max is None:
            v_max = self.ub - self.lb
        elif isinstance(v_max, Sequence) or isinstance(v_max, np.ndarray):
            if len(v_max) != self.ndim:
                raise ValueError('length of v_max must equal to ndim')
        else:
            raise TypeError(f'v_max must be NoneType, Sequence[Real] or np.ndarray, not {type(v_max)}')
        self.v_max = v_max

        self.positions: Optional[np.ndarray] = None
        self.values: Optional[np.ndarray] = None
        self.velocities: Optional[np.ndarray] = None
        self.p_best_values: Optional[np.ndarray] = None
        self.p_best_positions: Optional[np.ndarray] = None

    def init_population(self):
        self.positions = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop_size, self.ndim))
        self.values = self.func(self.positions).reshape(-1, 1)
        self.velocities = np.random.uniform(low=-self.v_max, high=self.v_max, size=(self.pop_size, self.ndim))
        self.p_best_values = self.values.copy()
        self.p_best_positions = self.positions.copy()
        self.update_global_beat()

    def update_velocities(self):
        self.velocities = self.w * self.velocities + \
                          self.c1 * np.random.rand(self.pop_size, self.ndim) * (self.best_position - self.positions) + \
                          self.c2 * np.random.rand(self.pop_size, self.ndim) * (self.p_best_positions - self.positions)

    def update_positions(self):
        self.positions = np.clip(self.positions + self.velocities, self.lb, self.ub)

    def update_values(self):
        self.values = self.func(self.positions).reshape(-1, 1)

    def update_partial_best(self):
        need_update = self.p_best_values > self.values
        self.p_best_positions = np.where(need_update, self.positions, self.p_best_positions)
        self.p_best_values = np.where(need_update, self.values, self.p_best_values)

    def update_global_beat(self):
        index_min = self.p_best_values.argmin()
        best_value = self.p_best_values[index_min][0]
        if self.best_value is None or self.best_value > best_value:
            self.update_best(best_value=best_value, best_position=self.positions[index_min, :].copy())

    def run_iter(self) -> Iterator[Tuple[Real, Sequence[Real]]]:
        self.init_population()
        for _ in range(self.max_iter):
            self.update_velocities()
            self.update_positions()
            self.update_values()
            self.update_partial_best()
            self.update_global_beat()
            self.log_history()
            yield self.best_value, self.best_position
