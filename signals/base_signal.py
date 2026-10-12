from __future__ import annotations
import numpy as np
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import List, Tuple, Union


class BaseSignal:
    """
    This base class defines the lhs and rhs arithmetic operation calls (+, -, * and /) for different operand
    types.
    """

    def __add__(self, other: Union[BaseSignal, float, int]) -> BaseSignal:
        if isinstance(other, BaseSignal):
            return SumOfSignals(self, other)
        elif isinstance(other, (float, int)):
            return SumOfSignals(self, Const(value=other))
        else:
            raise TypeError(
                "Addition with Signal only support for [SignalOperations | float | int]. "
                f"Operand + was called for types: {type(self)} + {type(other)}"
            )

    def __radd__(self, other: Union[BaseSignal, float, int]) -> BaseSignal:
        return self + other

    def __mul__(self, other: Union[BaseSignal, float, int]) -> BaseSignal:
        if isinstance(other, BaseSignal):
            return ProductOfSignals(self, other)
        elif isinstance(other, (float, int)):
            return ProductOfSignals(self, Const(value=other))
        else:
            raise TypeError(
                f"Multiplication with Signal only support for [SignalOperations | float | int]. "
                f"Operand * was called for types: {type(self)} * {type(other)}"
            )

    def __rmul__(self, other: Union[BaseSignal, float, int]) -> BaseSignal:
        return self * other

    def __sub__(self, other: Union[BaseSignal, float, int]) -> BaseSignal:
        if isinstance(other, BaseSignal):
            return DifferenceOfSignals(self, other)
        elif isinstance(other, (float, int)):
            return DifferenceOfSignals(self, Const(value=other))
        else:
            raise TypeError(
                "Addition with Signal only support for [SignalOperations | float | int]. "
                f"Operand - was called for types: {type(self)} - {type(other)}"
            )

    def __rsub__(self, other: Union[BaseSignal, float, int]) -> BaseSignal:
        if isinstance(other, BaseSignal):
            return DifferenceOfSignals(other, self)
        elif isinstance(other, (float, int)):
            return DifferenceOfSignals(Const(value=other), self)
        else:
            raise TypeError(
                "Addition with Signal only support for [SignalOperations | float | int]. "
                f"Operand - was called for types: {type(other)} - {type(self)}"
            )

    def __truediv__(self, other: Union[BaseSignal, float, int]) -> BaseSignal:
        if isinstance(other, BaseSignal):
            return DivisionOfSignals(self, other)
        elif isinstance(other, (float, int)):
            return DivisionOfSignals(self, Const(value=other))
        else:
            raise TypeError(
                "Addition with Signal only support for [SignalOperations | float | int]. "
                f"Operand / was called for types: {type(self)} / {type(other)}"
            )

    def __rtruediv__(self, other: Union[BaseSignal, float, int]):
        if isinstance(other, BaseSignal):
            return DivisionOfSignals(other, self)
        elif isinstance(other, (float, int)):
            return DivisionOfSignals(Const(value=other), self)
        else:
            raise TypeError(
                "Addition with Signal only support for [SignalOperations | float | int]. "
                f"Operand / was called for types: {type(other)} / {type(self)}"
            )

    def __call__(self, t: float) -> float:
        raise NotImplementedError


@dataclass
class TwoSidedOperation(BaseSignal, ABC):
    """Container class to define an arithmetic operations left and right hand side."""

    lhs: Union[BaseSignal, float, int]
    rhs: Union[BaseSignal, float, int]


class SumOfSignals(TwoSidedOperation):
    def __call__(self, t: float) -> float:
        return self.lhs(t) + self.rhs(t)


class DifferenceOfSignals(TwoSidedOperation):
    def __call__(self, t: float) -> float:
        return self.lhs(t) - self.rhs(t)


class ProductOfSignals(TwoSidedOperation):
    def __call__(self, t: float) -> float:
        return self.lhs(t) * self.rhs(t)


class DivisionOfSignals(TwoSidedOperation):
    def __call__(self, t: float) -> float:
        return self.lhs(t) / self.rhs(t)


@dataclass
class Signal(BaseSignal, ABC):
    """
    Signal objects can be called at time t (float) to return their (float) value.
    The signals are defined on the domain t in [t_start, t_end] and evaluate to zeto outside those bounds.
    The default domain is [0, inf]
    """

    t_start: float = field(default=0.0, init=True)
    t_end: float = field(default=np.PINF, init=True)

    def __call__(self, t: float) -> float:
        if self.t_start <= t < self.t_end:
            return self._signal(t - self.t_start)
        else:
            return 0.0

    @abstractmethod
    def _signal(self, t: float) -> float:
        """Evaluate the signal at time-step t."""
        raise NotImplementedError

    def eval_on(
        self,
        t_array: Union[np.ndarray, List[Union[int, float]], Tuple[Union[int, float]]],
    ) -> np.ndarray:
        """Evaluate the signal on an array of timestamps."""
        return np.array([self.__call__(t_i) for t_i in t_array])


@dataclass
class Const(Signal):
    """Signal with a constant value defined on domain [-inf, +inf]"""

    value: float = 0.0
    t_start: float = np.NINF

    def _signal(self, t: float) -> float:
        return self.value
