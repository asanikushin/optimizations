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
        return oracle.func(x_k + tau * (x_k - x_prev))

    @staticmethod
    def grad(oracle: BaseSmoothOracle, tau, x_k, x_prev):
        delta = x_k - x_prev
        return np.dot(oracle.grad(x_k + tau * delta), delta)


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
        return tau_k

    def __str__(self):
        return "Extrapolation momentum"


class ArmijoRule(BaseMomentum):
    def __init__(self, oracle: BaseSmoothOracle, alpha=1 / 3, beta=2 / 3):
        self.oracle = oracle
        self.alpha = alpha
        self.beta = beta

    def __call__(self, x_k, x_prev, **kwargs) -> float:
        f_value = kwargs.get("f_value") if kwargs.get("f_value") is None else self.oracle.func(x_k)
        f_grad = kwargs.get("f_grad") if kwargs.get("f_grad") is None else self.oracle.grad(x_k)
        phi_deriv = np.sum(2 * f_grad * x_k * (x_k - x_prev))
        left, right = f_value + phi_deriv / 3, f_value + 2 * phi_deriv / 3
        t_0, t_1 = 1, 1
        while phi_deriv >= 0:
            x_new = x_k + t_0 * (x_k - x_prev)
            f_new = self.oracle.func(x_new)
            if f_new <= left:
                if t_0 < t_1:
                    break
                t_1 = t_0 * 2
            elif f_new >= right:
                if t_0 > t_1:
                    break
                t_1 = t_0 / 2
            else:
                break
            t_0, t_1 = t_1, t_0
        t_0, t_1 = min(t_0, t_1), max(t_0, t_1)
        while np.abs(t_0 - t_1) > 1e-5:
            mid = (t_0 + t_1) / 2
            x_new = x_k + mid * (x_k - x_prev)
            f_new = self.oracle.func(x_new)
            if f_new <= left:
                t_0 = mid
            elif f_new >= right:
                t_1 = mid
            else:
                return mid
        return t_0

    def __str__(self):
        return "Armijo Rule"
