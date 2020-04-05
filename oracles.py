from abc import abstractmethod
import numpy as np
from typing import Dict, Optional, Union


class BaseSmoothOracle:
    """
    Base class for implementation of oracles.
    """

    @abstractmethod
    def func(self, x: np.ndarray, index: int = -1):
        """
        Computes the value of function at point x.
        :param x: point for computation (single value or vector)
        :param index:
        :return: function value (single value or vector)
        """
        raise NotImplementedError('Func oracle is not implemented.')

    @abstractmethod
    def grad(self, x: np.ndarray, index: int = -1) -> np.ndarray:
        """
        Computes the grad (or Jacobian) at point x.
        :param x: point for computation (single value or vector)
        :param index:
        :return: gradient/Jacobian value (single value, vector, matrix)
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    @abstractmethod
    def counters(self):
        """
        Get inner counters like number of calls
        :return: dict with inner counters
        """
        raise NotImplementedError('Counters oracle is not implemented.')

    @abstractmethod
    def reset(self) -> None:
        """
        Correct reset of counters, revert oracle state into start
        :return:
        """
        raise NotImplementedError('Reset oracle is not implemented.')


class ChebyshevOscillator(BaseSmoothOracle):  # phi_i(x)
    def __init__(self) -> None:
        self.calls: Dict[str, int] = dict(value=0, gradient=0)

    def counters(self) -> Dict[str, int]:
        return self.calls

    def reset(self) -> None:
        self.calls = dict(value=0, gradient=0)

    def func(self, x: np.ndarray, index: int = -1) -> np.float64:
        self.calls["value"] += 1

        return 2 * (x[index] ** 2) - 1

    def grad(self, x: np.ndarray, index: int = -1) -> np.ndarray:
        self.calls["gradient"] += 1

        vector = np.zeros(x.shape)
        vector[index] = 4 * x[index]
        return vector

    def __str__(self) -> str:
        return "Chebyshev Oscillator"


class TrigonometricOscillator(BaseSmoothOracle):
    def __init__(self) -> None:
        self.calls = dict(value=0, gradient=0)

    def counters(self) -> Dict[str, int]:
        return self.calls

    def reset(self) -> None:
        self.calls = dict(value=0, gradient=0)

    def func(self, x: np.ndarray, index: Optional[int] = None) -> np.float64:
        self.calls["value"] += 1
        if index is None:
            index = -1
        return -np.cos(np.pi * x[index])

    def grad(self, x: np.ndarray, index: Optional[int] = None) -> np.ndarray:
        self.calls["gradient"] += 1
        if index is None:
            index = -1

        vector = np.zeros(x.shape)
        vector[index] = np.pi * np.sin(np.pi * x[index])
        return vector

    def __str__(self) -> str:
        return "Trigonometric Oscillator"


class BigFuncOracle(BaseSmoothOracle):  # F(x)
    def __init__(self, n, oscillator: BaseSmoothOracle) -> None:
        assert n > 0
        self.dim = n
        self.oscillator = oscillator
        self.calls = dict(value=0, gradient=0)

    def func(self, x: np.ndarray, index=None) -> np.ndarray:
        assert x.shape[0] == self.dim
        self.calls["value"] += 1

        vector = [x[0] - 1]
        for i in range(1, self.dim):
            vector.append(x[i] - self.oscillator.func(x, index=i - 1))
        assert len(vector) == self.dim
        return np.array(vector)

    def grad(self, x: np.ndarray, index=None) -> np.ndarray:
        assert x.shape[0] == self.dim
        self.calls["gradient"] += 1

        matrix = np.identity(self.dim)
        for i in range(1, self.dim):
            matrix[i] -= self.oscillator.grad(x, index=i - 1)
        return matrix

    def counters(self) -> Dict[str, int]:
        return {"oscillator oracle": self.oscillator.counters(), **self.calls}

    def reset(self) -> None:
        self.calls = dict(value=0, gradient=0)
        self.oscillator.reset()

    def __str__(self) -> str:
        return f"F with {self.oscillator}"


class LittleFuncOracle(BaseSmoothOracle):  # f_1^hat
    def __init__(self, n, big_func: BigFuncOracle) -> None:
        assert n > 0
        self.dim = n
        self.big_func = big_func
        self.calls = dict(value=0, gradient=0)

    def func(self, x: np.ndarray, index: int = -1):
        assert x.shape[0] == self.dim
        self.calls["value"] += 1
        return np.linalg.norm(self.big_func.func(x)) / np.sqrt(self.dim)

    def grad(self, x: np.ndarray, index: int = -1) -> np.ndarray:
        assert x.shape[0] == self.dim
        self.calls["gradient"] += 1
        value = self.big_func.func(x)
        jacobean = self.big_func.grad(x)

        return np.dot(value, jacobean) / np.linalg.norm(value) / np.sqrt(self.dim)

    def counters(self) -> Dict[str, Union[int, Dict[str, int]]]:
        return {
            "F oracle": self.big_func.counters(),
            **self.calls}

    def reset(self):
        self.calls = dict(value=0, gradient=0)
        self.big_func.reset()

    def __str__(self):
        return f"f_1^hat for {self.big_func}"
