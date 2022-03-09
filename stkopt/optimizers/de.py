from numbers import Real
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Tuple

import numpy as np

from ._base import OptimizerBase, T_WrapperMode, T_SequenceReal


class DE(OptimizerBase):
    def __init__(self, func: Callable, ndim: int = None, lb: T_SequenceReal = None, ub: T_SequenceReal = None,
                 func_wrapper_mode: T_WrapperMode = 'common', func_wrapper_options: Optional[Dict[str, Any]] = None,
                 log_iter_history: bool = True, log_func_call_history: bool = True,
                 pop_size: int = 40, max_iter: int = 150, f: Real = 0.5, cr: Real = 0.3):
        super().__init__(func, ndim, lb, ub, func_wrapper_mode, func_wrapper_options, log_iter_history,
                         log_func_call_history)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.f = f
        self.cr = cr

        self.positions: Optional[np.ndarray] = None
        self.values: Optional[np.ndarray] = None
        self.previous_values: Optional[np.ndarray] = None

        self.v: Optional[np.ndarray] = None
        self.u: Optional[np.ndarray] = None
        self.fitnesses: Optional[np.ndarray] = None
        self.previous_fitnesses: Optional[np.ndarray] = None

    def init_population(self):
        self.positions = np.random.randint(low=self.lb, high=self.ub, size=(self.pop_size, self.ndim))
        self.u = self.positions
        self.update_values()
        self.ranking()

    def mutation(self):
        r1, r2, r3 = (np.random.randint(0, self.pop_size, size=self.pop_size) for _ in range(3))
        self.v = self.positions[r1, :] + self.f * (self.positions[r2, :] - self.positions[r3, :])
        self.v = np.where((self.lb < self.v) & (self.v < self.ub),
                          self.v,
                          np.random.uniform(low=self.lb, high=self.ub, size=(self.pop_size, self.ndim)))

    def crossover(self):
        self.u = np.where(np.random.rand(self.pop_size, self.ndim) < self.cr,
                          self.v,
                          self.positions)

    def update_values(self):
        self.previous_values = self.values
        self.values = self.func(self.u)

    def ranking(self):
        self.previous_fitnesses = self.fitnesses
        self.fitnesses = -self.values

    def selection(self):
        self.positions = np.where((self.fitnesses > self.previous_fitnesses).reshape(-1, 1),
                                  self.u,
                                  self.positions)

    def update_generation_best(self):
        generation_best_index = self.values.argmin()
        self.best_value = self.values[generation_best_index]
        self.best_position = self.positions[generation_best_index, :]

    def run_iter(self, *args, **kwargs) -> Iterator[Tuple[Real, Sequence[Real]]]:
        self.init_population()
        for i in range(self.max_iter):
            self.mutation()
            self.crossover()
            self.update_values()
            self.ranking()
            self.selection()
            self.update_generation_best()
            self.log_history()
            yield self.best_value, self.best_position
