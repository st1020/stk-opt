from numbers import Real
from typing import Callable, Sequence

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .test_functions import OptTestFunction


def draw_test_func(func: Callable, ndim: int = None, lb: Sequence[Real] = None, ub: Sequence[Real] = None,
                   step: Real = None):
    """
    绘制测试函数图像，如果 func 类型是 OptTestFunction，那么可以无需指定 ndim, lb, ub。不指定 step 时将根据 lb 和 ub 自动设置。

    :param func: 测试函数
    :param ndim: 维度
    :param lb: 参数最大值列表
    :param ub: 参数最小值列表
    :param step: 绘图步长
    """
    if ndim is None and isinstance(func, OptTestFunction):
        ndim = func.ndim
    if lb is None and isinstance(func, OptTestFunction):
        lb = func.lb
    if ub is None and isinstance(func, OptTestFunction):
        ub = func.ub

    if ndim == 1:
        start = lb[0]
        stop = ub[0]
        if step is None:
            step = (stop - start) / 1000
        x = np.arange(start, stop, step)
        y = func(np.array([x]))
        plt.plot(x, y)
        plt.show()
    elif ndim == 2:
        start_x, start_y = lb[:2]
        stop_x, stop_y = ub[:2]
        if step is None:
            step = max(stop_x - start_x, stop_y - start_y) / 1000
        figure = plt.figure()
        axes = Axes3D(figure, auto_add_to_figure=False)
        figure.add_axes(axes)
        x = np.arange(start_x, stop_x, step)
        y = np.arange(start_y, stop_y, step)
        x, y = np.meshgrid(x, y)
        z = func(np.array([x, y]))
        axes.plot_surface(x, y, z, cmap='rainbow')
        plt.show()
    else:
        raise ValueError(f'ndim must be 1 or 2, but {ndim} was given')
