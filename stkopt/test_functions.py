from numbers import Real
from inspect import isclass
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union, Sequence, Optional, Type

import numpy as np


class OptTestFunction(ABC):
    name: str
    description: str
    reference: str

    _ndim: Optional[int] = None
    _lb: Union[Real, Sequence[Real]]
    _ub: Union[Real, Sequence[Real]]
    _global_minimum_value: Real
    _global_minimum_position: Sequence[Union[Real, Sequence[Real]]]

    def __init__(self, ndim: int = None):
        if self._ndim is None:
            if ndim is None:
                raise TypeError("missing 1 required positional argument: 'ndim'")
            self.ndim = ndim
        else:
            if ndim is not None and ndim != self._ndim:
                raise ValueError(f'ndim must be {self._ndim}')
            self.ndim = self._ndim

    @staticmethod
    @abstractmethod
    def test_func(xx, *args, **kwargs):
        pass

    def _check_value(self, value, name: str) -> Tuple[Real, ...]:
        if isinstance(value, Sequence):
            return tuple(value)
        elif isinstance(value, Real):
            return tuple([value] * self.ndim)
        else:
            raise TypeError(f'{name} must be list or Real, not {type(value)}')

    @property
    def lb(self) -> Tuple[Real, ...]:
        return self._check_value(self._lb, '_lb')

    @property
    def ub(self) -> Tuple[Real, ...]:
        return self._check_value(self._ub, '_ub')

    @property
    def global_minimum_value(self) -> Real:
        if isinstance(self._global_minimum_value, Real):
            return self._global_minimum_value
        else:
            raise TypeError(f'_global_minimum_value must be list or Real, not {type(self._global_minimum_value)}')

    @property
    def global_minimum_position(self) -> Tuple[Tuple[Real, ...], ...]:
        return tuple(map(lambda x: self._check_value(x, '_global_minimum_position'), self._global_minimum_position))

    def __call__(self, xx, *args, **kwargs):
        if self.ndim != len(xx):
            raise ValueError(f'error quantity of argument, ndim is {self.ndim}')
        return self.test_func(xx, *args, **kwargs)

    def get_func_detail(self):
        return self, self.ndim, self.lb, self.ub


class F1(OptTestFunction):
    name = 'Ackley Function'
    description = 'Many Local Minima'
    reference = 'https://www.sfu.ca/~ssurjano/ackley.html'

    _lb = -32.768
    _ub = 32.768
    _global_minimum_value = 0
    _global_minimum_position = [0]

    @staticmethod
    def test_func(xx, a=20, b=0.2, c=2 * np.pi):
        d = len(xx)
        sum1 = np.sum(xx ** 2, axis=0)
        sum2 = np.sum(np.cos(c * xx), axis=0)
        term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
        term2 = - np.exp(sum2 / d)
        return term1 + term2 + a + np.exp(1)


class F2(OptTestFunction):
    name = 'Bukin Function'
    description = 'Many Local Minima'
    reference = 'https://www.sfu.ca/~ssurjano/bukin6.html'

    _ndim = 2
    _lb = [-15, -3]
    _ub = [-5, 3]
    _global_minimum_value = 0
    _global_minimum_position = [(-10, 1)]

    @staticmethod
    def test_func(xx, *args, **kwargs):
        x1, x2 = xx
        term1 = 100 * np.sqrt(np.abs(x2 - 0.01 * np.power(x1, 2)))
        term2 = 0.01 * np.abs(x1 + 10)
        return term1 + term2


class F3(OptTestFunction):
    name = 'Cross-in-Tray Function'
    description = 'Many Local Minima'
    reference = 'https://www.sfu.ca/~ssurjano/crossit.html'

    _ndim = 2
    _lb = -10
    _ub = 10
    _global_minimum_value = -2.06261
    _global_minimum_position = [(1.3491, -1.3491), (1.3491, 1.3491), (-1.3491, 1.3491), (-1.3491, -1.3491)]

    @staticmethod
    def test_func(xx, *args, **kwargs):
        x1, x2 = xx
        fact1 = np.sin(x1) * np.sin(x2)
        fact2 = np.exp(np.abs(100 - np.sqrt(np.power(x1, 2) + np.power(x2, 2)) / np.pi))
        return -0.0001 * np.power(np.abs(fact1 * fact2) + 1, 0.1)


class F4(OptTestFunction):
    name = 'Drop-Wave Function'
    description = 'Many Local Minima'
    reference = 'https://www.sfu.ca/~ssurjano/drop.html'

    _ndim = 2
    _lb = -5.12
    _ub = 5.12
    _global_minimum_value = -1
    _global_minimum_position = [(0, 0)]

    @staticmethod
    def test_func(xx, *args, **kwargs):
        x1, x2 = xx
        frac1 = 1 + np.cos(12 * np.sqrt(np.power(x1, 2) + np.power(x2, 2)))
        frac2 = 0.5 * (np.power(x1, 2) + np.power(x2, 2)) + 2
        return -frac1 / frac2


class F5(OptTestFunction):
    name = 'Eggholder Function'
    description = 'Many Local Minima'
    reference = 'https://www.sfu.ca/~ssurjano/egg.html'

    _ndim = 2
    _lb = -512
    _ub = 512
    _global_minimum_value = -959.6407
    _global_minimum_position = [(512, 404.2319)]

    @staticmethod
    def test_func(xx, *args, **kwargs):
        x1, x2 = xx
        term1 = -(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + x1 / 2 + 47)))
        term2 = -x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))
        return term1 + term2


test_functions: Dict[str, Type[OptTestFunction]] = {}
model = None  # define 'model' first to avoid globals changing while 'for'
for model in globals().values():
    if isclass(model) and issubclass(model, OptTestFunction):
        test_functions[model.__name__] = model
