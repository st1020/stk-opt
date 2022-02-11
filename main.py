from stkopt.test_functions import F1, F2
from stkopt.optimizers import PSO
from stkopt.utils import draw_test_func, execution_time


@execution_time
def run(mode):
    pso = PSO(F2(2), func_wrapper_mode=mode)
    pso.run()


if __name__ == '__main__':
    draw_test_func(F1(2))
    for i in ('common', 'vectorization', 'cached', 'multithreading', 'multiprocessing'):
        run(i)
