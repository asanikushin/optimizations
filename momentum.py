from oracles import BaseSmoothOracle
import numpy as np


class BaseMomentum:
    def __call__(self, x_k, x_prev, *args, **kwargs) -> float:
        """
        Find value for tau_k
        :param x_k: current vector-point
        :param x_prev: previous vector-point
        :return: float value for tau
        """
        raise NotImplementedError("Call function is not implemented")

    @staticmethod
    def value(oracle: BaseSmoothOracle, tau, x_k, x_prev):
        return oracle.func(x_prev + tau * (x_k - x_prev))

    @staticmethod
    def grad(oracle: BaseSmoothOracle, tau, x_k, x_prev):
        delta = x_k - x_prev
        return np.dot(oracle.grad(x_prev + tau * delta), delta)


class Extrapolation(BaseMomentum):
    def __init__(self, oracle: BaseSmoothOracle):
        self.oracle = oracle

    def __call__(self, x_k, x_prev, *args, **kwargs) -> float:
        def value(tau):
            return self.value(self.oracle, tau, x_k, x_prev)

        def grad(tau):
            return self.grad(self.oracle, tau, x_k, x_prev)

        tau_k = 1
        tau_prev = 0
        while grad(tau_k) and value(tau_k) < value(tau_prev):
            tau_prev = tau_k
            tau_k = 2 * tau_k
        return tau_prev

    def __str__(self):
        return "Extrapolation momentum"


class ArmijoRule(BaseMomentum):
    def __init__(self, oracle: BaseSmoothOracle, alpha=1 / 3, beta=2 / 3):
        self.oracle = oracle
        self.alpha = alpha
        self.beta = beta

    def __call__(self, x_k, x_prev, **kwargs) -> float:
        def grad(tau):
            return self.grad(self.oracle, tau, x_k, x_prev)

        f_value = kwargs["f_value"] if "f_value" in kwargs else self.oracle.func(x_prev)
        f_deriv = grad(0)
        if f_deriv >= 0:
            return 0.0

        def left_border(tau):
            return f_value + f_deriv * self.beta * tau

        def right_border(tau):
            return f_value + f_deriv * self.alpha * tau

        if left_border(1) <= f_value <= right_border(1):
            return 1.0

        t_0, t_1 = 1, 1
        while True:
            x_new = x_prev + t_0 * (x_k - x_prev)
            f_new = self.oracle.func(x_new)
            if t_0 >= t_1 and f_new < right_border(t_0):
                t_1 = t_0 * 2
            elif t_0 <= t_1 and f_new > left_border(t_0):
                t_1 = t_0 / 2
            else:
                break
            t_0, t_1 = t_1, t_0
        t_0, t_1 = min(t_0, t_1), max(t_0, t_1)
        if t_1 < 1e-3:
            return 0.0  # there is actually no difference
        while True:
            mid = (t_0 + t_1) / 2
            x_new = x_prev + mid * (x_k - x_prev)
            f_new = self.oracle.func(x_new)
            if f_new < left_border(mid):
                t_0 = mid
            elif f_new > right_border(mid):
                t_1 = mid
            else:
                return mid

    def __str__(self):
        return "Armijo Rule"
