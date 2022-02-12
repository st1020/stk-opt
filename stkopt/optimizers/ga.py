from numbers import Real
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Tuple, Union

import numpy as np

from ._base import OptimizerBase, T_WrapperMode, T_SequenceReal


class GA(OptimizerBase):
    def __init__(self, func: Callable, ndim: int = None, lb: T_SequenceReal = None, ub: T_SequenceReal = None,
                 func_wrapper_mode: T_WrapperMode = 'common', func_wrapper_options: Optional[Dict[str, Any]] = None,
                 log_iter_history: bool = True, log_func_call_history: bool = True,
                 pop_size: int = 40, max_iter: int = 150, pc: Real = 0.9, pm: Real = 0.001,
                 precision: Union[Real, Sequence[Real]] = 1e-7):
        super().__init__(func, ndim, lb, ub, func_wrapper_mode, func_wrapper_options, log_iter_history,
                         log_func_call_history)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.pc = pc
        self.pm = pm
        self.precision = np.array(precision) + np.zeros(self.ndim)

        self.individual_len: np.ndarray = np.ceil(np.log2((self.ub - self.lb) / self.precision + 1)).astype(int)
        self.chromosome_len: int = int(sum(self.individual_len))

        self.values: Optional[np.ndarray] = None
        self.positions: Optional[np.ndarray] = None
        self.chromosomes: Optional[np.ndarray] = None
        self.fitnesses: Optional[np.ndarray] = None

    def init_population(self):
        self.chromosomes = np.random.randint(low=0, high=2, size=(self.pop_size, self.chromosome_len))

    def decode_chromosome(self):
        gene_index = self.individual_len.cumsum()
        self.positions = np.zeros(shape=(self.pop_size, self.ndim))
        for i, j in enumerate(gene_index):
            if i == 0:
                gene = self.chromosomes[:, :gene_index[0]]
            else:
                gene = self.chromosomes[:, gene_index[i - 1]:gene_index[i]]
            self.positions[:, i] = self.decode_gray_code(gene)
        self.positions = self.lb + (self.ub - self.lb) * self.positions

    @staticmethod
    def decode_gray_code(gray_code):
        gray_code_len = gray_code.shape[1]
        binary_code = gray_code.cumsum(axis=1) % 2
        mask = np.logspace(start=1, stop=gray_code_len, base=0.5, num=gray_code_len)
        return (binary_code * mask).sum(axis=1) / mask.sum()

    def update_values(self):
        self.values = self.func(self.positions)

    def ranking(self):
        self.fitnesses = -self.values

    def selection_roulette(self):
        fitnesses = self.fitnesses
        fitnesses = fitnesses - np.min(fitnesses) + 1e-10
        select_prob = fitnesses / fitnesses.sum()
        select_index = np.random.choice(range(self.pop_size), size=self.pop_size, p=select_prob)
        self.chromosomes = self.chromosomes[select_index, :]

    def selection_tournament(self, tournament_size=3):
        aspirant_indexes = np.random.randint(self.pop_size, size=(self.pop_size, tournament_size))
        aspirant_values = self.fitnesses[aspirant_indexes]
        winner = aspirant_values.argmax(axis=1)
        select_index = [aspirant_indexes[i, j] for i, j in enumerate(winner)]
        self.chromosomes = self.chromosomes[select_index, :]

    selection = selection_tournament

    def crossover_1point(self):
        for i in range(0, self.pop_size, 2):
            if np.random.rand() < self.pc:
                n = np.random.randint(self.chromosome_len)
                seg1, seg2 = self.chromosomes[i, n:].copy(), self.chromosomes[i + 1, n:].copy()
                self.chromosomes[i, n:], self.chromosomes[i + 1, n:] = seg2, seg1

    def crossover_2point(self):
        for i in range(0, self.pop_size, 2):
            if np.random.rand() < self.pc:
                n1, n2 = np.random.randint(0, self.chromosome_len, 2)
                if n1 > n2:
                    n1, n2 = n2, n1
                seg1, seg2 = self.chromosomes[i, n1:n2].copy(), self.chromosomes[i + 1, n1:n2].copy()
                self.chromosomes[i, n1:n2], self.chromosomes[i + 1, n1:n2] = seg2, seg1

    crossover = crossover_2point

    def mutation(self):
        self.chromosomes ^= (np.random.rand(self.pop_size, self.chromosome_len) < self.pm)

    def update_generation_best(self):
        generation_best_index = self.fitnesses.argmax()
        self.best_value = self.values[generation_best_index]
        self.best_position = self.positions[generation_best_index, :]

    def run_iter(self, *args, **kwargs) -> Iterator[Tuple[Real, Sequence[Real]]]:
        self.init_population()
        for i in range(self.max_iter):
            self.decode_chromosome()
            self.update_values()
            self.ranking()
            self.selection()
            self.crossover()
            self.mutation()
            self.update_generation_best()
            self.log_history()
            yield self.best_value, self.best_position
