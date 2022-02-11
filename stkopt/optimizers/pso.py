from numbers import Real
from typing import Any, Callable, Dict, Optional, Sequence, Iterator, Tuple

import numpy as np

from ._base import OptimizerBase, FUNC_WRAPPER_MODES_TYPE


class PSO(OptimizerBase):
    def __init__(self, func: Callable, ndim: int = None, lb: Sequence[Real] = None, ub: Sequence[Real] = None,
                 func_wrapper_mode: FUNC_WRAPPER_MODES_TYPE = 'common',
                 func_wrapper_options: Optional[Dict[str, Any]] = None,
                 log_iter_history: bool = True, log_func_call_history: bool = True,
                 pop_size: int = 40, max_iter: int = 150, w: Real = 0.8, c1: Real = 0.5, c2: Real = 0.5,
                 v_max: Optional[Sequence[Real]] = None):
        super().__init__(func, ndim, lb, ub, func_wrapper_mode, func_wrapper_options, log_iter_history,
                         log_func_call_history)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        if v_max is None:
            v_max = self.ub - self.lb
        elif isinstance(v_max, Sequence):
            if len(v_max) != self.ndim:
                raise ValueError('length of v_max must equal to ndim')
        else:
            raise TypeError(f'v_max must be NoneType or Sequence[Real], not {type(v_max)}')
        self.v_max = v_max

    def run_iter(self) -> Iterator[Tuple[Real, Sequence[Real]]]:
        # init population
        positions = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop_size, self.ndim))
        values = self.func(positions).reshape(-1, 1)
        velocities = np.random.uniform(low=-self.v_max, high=self.v_max, size=(self.pop_size, self.ndim))
        p_best_values = values.copy()
        p_best_positions = positions.copy()
        index_min = p_best_values.argmin()
        self.update_best(best_value=p_best_values[index_min][0], best_position=positions[index_min, :].copy())

        for _ in range(self.max_iter):
            # update velocities
            velocities = self.w * velocities + \
                         self.c1 * np.random.rand(self.pop_size, self.ndim) * (self.best_position - positions) + \
                         self.c2 * np.random.rand(self.pop_size, self.ndim) * (p_best_positions - positions)
            # update positions
            positions = np.clip(positions + velocities, self.lb, self.ub)
            # update values
            values = self.func(positions).reshape(-1, 1)
            # update partial best
            need_update = p_best_values > values
            p_best_positions = np.where(need_update, positions, p_best_positions)
            p_best_values = np.where(need_update, values, p_best_values)
            # update global best
            index_min = p_best_values.argmin()
            self.update_best(best_value=p_best_values[index_min][0], best_position=positions[index_min, :].copy())
            # log history
            self.log_history()
            yield self.best_value, self.best_position
