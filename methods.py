from oracles import BaseSmoothOracle
from momentum import BaseMomentum
import numpy as np
from typing import Tuple, Dict, Any, Optional, List

MAX_ITER = 100 * 1000
TOLERANCE = 1e-7


class BaseOptimizationMethod:
    """
    Base class for implementation of optimizations.
    args for __init__
    :param oracle: instance of BaseSmoothOracle with implemented methods
    :param tolerance: epsilon for stop criteria, default 1e-4
    :param max_iter: maximumself.base_steps for optimization method, default 10000
    """

    def __call__(self, x_0, l_0, *args, **kwargs) -> Tuple[np.ndarray, float]:
        """
        Optimize function within oracle
        :param x_0: start point
        :param args:
        :param kwargs:
        :return: function optimum point
        """
        raise NotImplementedError("Call function is not implemented")

    def counters(self):
        """
        Get inner counters like number ofself.base_steps
        :return: dict with inner counters: number of iterations, list with objective values, list with gradient norms
        """
        raise NotImplementedError('Counters optimization is not implemented.')

    def reset(self):
        """
        Correct reset of counters, revert optimization method state into start
        :return:
        """
        raise NotImplementedError('Reset optimization is not implemented.')


class PureGradientMethod(BaseOptimizationMethod):
    """Simple gradient method with momentum

        Momentum can be either None, extrapolation, armijo
        """

    def __init__(
            self,
            oracle: BaseSmoothOracle,
            tolerance: float = TOLERANCE,
            max_iter: int = MAX_ITER,
            momentum: Optional[BaseMomentum] = None
    ) -> None:
        if max_iter <= 0:
            raise ValueError("max_iter must be greater than 0")
        self.oracle = oracle
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.momentum = momentum
        self.calls: Dict[str, List[np.float64]] = dict(values=[], gradients=[])
        self.iteration = 0

    def __call__(self, x_0, l_0, *args, **kwargs) -> Tuple[np.ndarray, float]:
        x_k, l_k = x_0, l_0
        f_value = self.oracle.func(x_k)
        f_grad = self.oracle.grad(x_k)
        f_prev = np.float64("-inf")
        x_prev = x_k
        while (self.iteration == 0 or np.linalg.norm(x_k - x_prev) > self.tolerance) and self.iteration < self.max_iter:
            if self.iteration > 1 and self.momentum is not None:
                tau = self.momentum(x_k, x_prev, f_grad=f_grad, f_value=f_value)
                x_k = x_k + tau * (x_k - x_prev)
            f_prev = f_value = self.oracle.func(x_k)
            f_grad = self.oracle.grad(x_k)

            i_k = 0
            while True:
                y_star = x_k - f_grad / (2 ** i_k * l_k)
                f_star_value = self.oracle.func(y_star)
                if f_value - f_star_value + np.dot(f_grad, y_star - x_k) + \
                        (2 ** i_k * l_k / 2) * np.sum((y_star - x_k) ** 2) > 0:
                    break
                i_k += 1

            f_value = f_star_value
            x_prev = x_k
            x_k = y_star
            l_k *= 2 ** (i_k - 1)

            self.iteration += 1
            self.calls["gradients"].append(np.linalg.norm(f_grad))
            self.calls["values"].append(f_value)
        return x_k, self.calls["values"][-1]

    def counters(self):
        return self.iteration, self.calls["values"], self.calls["gradients"]

    def reset(self) -> None:
        self.calls = dict(values=[], gradients=[], iterations=0)

    def __str__(self) -> str:
        return "Pure gradient method" + ((" with " + str(self.momentum)) if self.momentum else "")
