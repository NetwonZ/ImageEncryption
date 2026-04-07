"""One-dimensional chaotic system based on a sinusoidal-cosine map.

Given formula:
f(x) = |sin((A + B*mu) * (4/(a-0.5) * cos(pi*x)))|
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Union

import numpy as np
import sympy as sp


Number = Union[int, float, np.number]
ArrayInput = Union[Number, np.ndarray]
MapExprBuilder = Callable[[sp.Symbol, dict[str, sp.Symbol]], sp.Expr]


class Chaotic1DSystem:
	"""A configurable 1D chaotic map system with automatic differentiation."""

	def __init__(
		self,
		params: dict[str, float],
		map_expr_builder: MapExprBuilder | None = None,
	) -> None:
		if not isinstance(params, dict) or len(params) == 0:
			raise ValueError("'params' must be a non-empty dictionary of parameter values.")

		self.params = {name: float(value) for name, value in params.items()}
		self._param_order = list(self.params.keys())
		self._map_expr_builder = map_expr_builder or self._default_map_expr
		self._uses_default_map = map_expr_builder is None

		self._compile_symbolic_system()

	@staticmethod
	def _default_map_expr(x: sp.Symbol, p: dict[str, sp.Symbol]) -> sp.Expr:
		required = {"A", "B", "mu", "a"}
		missing = required - set(p)
		if missing:
			raise ValueError(f"Default map requires params keys: {sorted(required)}. Missing: {sorted(missing)}")

		return sp.Abs(
			sp.sin(
				(p["A"] + p["B"] * p["mu"]) * ((sp.Integer(4) / (p["a"] - sp.Rational(1, 2))) * sp.cos(sp.pi * x))
			)
		)

	def _compile_symbolic_system(self) -> None:
		self._x_symbol = sp.Symbol("x", real=True)
		self._param_symbols = {name: sp.Symbol(name, real=True) for name in self._param_order}

		expr = self._map_expr_builder(self._x_symbol, self._param_symbols)
		if not isinstance(expr, sp.Expr):
			raise TypeError("'map_expr_builder' must return a SymPy expression.")

		expected_symbols = {self._x_symbol, *self._param_symbols.values()}
		extra_symbols = expr.free_symbols - expected_symbols
		if extra_symbols:
			raise ValueError(f"Expression has unknown symbols: {sorted(str(s) for s in extra_symbols)}")

		self._map_expr = expr
		self._diff_expr = sp.diff(self._map_expr, self._x_symbol)

		arg_symbols = [self._x_symbol] + [self._param_symbols[name] for name in self._param_order]
		self._map_callable = sp.lambdify(arg_symbols, self._map_expr, modules="numpy")
		self._diff_callable = sp.lambdify(arg_symbols, self._diff_expr, modules="numpy")

	def set_map_relation(self, map_expr_builder: MapExprBuilder, params: dict[str, float] | None = None) -> None:
		"""Replace f(x) relation and optionally refresh parameters."""
		if params is not None:
			if not isinstance(params, dict) or len(params) == 0:
				raise ValueError("'params' must be a non-empty dictionary of parameter values.")
			self.params = {name: float(value) for name, value in params.items()}
			self._param_order = list(self.params.keys())

		self._map_expr_builder = map_expr_builder
		self._uses_default_map = False
		self._compile_symbolic_system()

	def _evaluate_map(self, x: ArrayInput) -> np.ndarray:
		param_values = [self.params[name] for name in self._param_order]
		return np.asarray(self._map_callable(x, *param_values), dtype=float)

	def _evaluate_diff(self, x: Number) -> float:
		param_values = [self.params[name] for name in self._param_order]
		derivative = self._diff_callable(float(x), *param_values)
		return float(np.abs(derivative))

	def _validate_default_singularity(self) -> None:
		if self._uses_default_map and np.isclose(self.params.get("a", np.nan), 0.5):
			raise ValueError("Parameter 'a' cannot be 0.5 for the default map because of division by zero.")

	def map_value(self, x: ArrayInput) -> Union[float, np.ndarray]:
		"""Evaluate one iteration of the chaotic map.

		Args:
			x: Scalar or NumPy array input.

		Returns:
			Scalar float for scalar input, ndarray for array input.
		"""
		self._validate_default_singularity()

		x_array = np.asarray(x, dtype=float)
		mapped = self._evaluate_map(x_array)

		if np.isscalar(x) or x_array.ndim == 0:
			return float(mapped)
		return mapped

	def _abs_derivative(self, x: Number) -> float:
		"""Return |f'(x)| from symbolic auto-differentiation."""
		self._validate_default_singularity()
		return self._evaluate_diff(x)

	def generate_sequence(self, x0: Number, n: int, include_x0: bool = False) -> np.ndarray:
		"""Generate a chaotic sequence by iterative mapping.

		Args:
			x0: Initial scalar value.
			n: Number of iterations to generate.
			include_x0: If True, prepend x0 to the output sequence.

		Returns:
			NumPy array of generated values.
		"""
		if not isinstance(n, int):
			raise TypeError("'n' must be an integer.")
		if n <= 0:
			raise ValueError("'n' must be a positive integer.")

		x0_array = np.asarray(x0)
		if x0_array.ndim != 0:
			raise TypeError("'x0' must be a scalar value.")

		values = []
		current = float(x0)

		if include_x0:
			values.append(current)

		for _ in range(n):
			current = float(self.map_value(current))
			values.append(current)

		return np.asarray(values, dtype=float)

	def lyapunov_exponent(self, x0: Number, n: int, discard: int = 100, epsilon: float = 1e-12) -> float:
		"""Estimate Lyapunov exponent for the map orbit starting from x0.

		Lyapunov exponent is estimated by:
		lambda ~= (1/n) * sum(log(|f'(x_i)|)), i=1..n
		after discarding initial transient iterations.

		Args:
			x0: Initial scalar value.
			n: Number of post-transient iterations used for estimation.
			discard: Number of initial transient iterations to skip.
			epsilon: Lower bound for |f'(x)| before log, for numerical stability.

		Returns:
			Estimated Lyapunov exponent as float.
		"""
		if not isinstance(n, int):
			raise TypeError("'n' must be an integer.")
		if not isinstance(discard, int):
			raise TypeError("'discard' must be an integer.")
		if n <= 0:
			raise ValueError("'n' must be a positive integer.")
		if discard < 0:
			raise ValueError("'discard' must be a non-negative integer.")
		if epsilon <= 0:
			raise ValueError("'epsilon' must be positive.")

		x0_array = np.asarray(x0)
		if x0_array.ndim != 0:
			raise TypeError("'x0' must be a scalar value.")

		self._validate_default_singularity()

		current = float(x0)
		total_steps = discard + n
		log_sum = 0.0

		for step in range(total_steps):
			if step >= discard:
				abs_derivative = self._abs_derivative(current)
				log_sum += float(np.log(max(abs_derivative, epsilon)))
			current = float(self.map_value(current))

		return log_sum / float(n)

	def lyapunov_parameter_scan(
		self,
		parameter: str,
		values: np.ndarray,
		x0: Number,
		n: int,
		discard: int = 100,
		epsilon: float = 1e-12,
	) -> tuple[np.ndarray, np.ndarray]:
		"""Scan one parameter in params and return Lyapunov curve data.

		Args:
			parameter: Key in self.params to scan.
			values: 1D NumPy array of parameter values to scan.
			x0: Initial scalar value for orbit.
			n: Number of post-transient iterations per scan point.
			discard: Number of transient iterations to skip per point.
			epsilon: Lower bound for |f'(x)| before log.

		Returns:
			A tuple (param_values, lyapunov_values), both 1D arrays.
		"""
		if parameter not in self.params:
			raise ValueError(f"Unknown parameter '{parameter}'. Available keys: {self._param_order}")

		param_values = np.asarray(values, dtype=float)
		if param_values.ndim != 1:
			raise TypeError("'values' must be a 1D array.")
		if param_values.size == 0:
			raise ValueError("'values' must not be empty.")

		original_value = float(self.params[parameter])
		lyap_values = np.empty_like(param_values, dtype=float)

		try:
			for idx, param in enumerate(param_values):
				param_float = float(param)
				self.params[parameter] = param_float
				try:
					lyap_values[idx] = self.lyapunov_exponent(
						x0=x0,
						n=n,
						discard=discard,
						epsilon=epsilon,
					)
				except (ValueError, ZeroDivisionError, FloatingPointError, OverflowError):
					lyap_values[idx] = np.nan
		finally:
			self.params[parameter] = original_value

		#sym:lyapunov_parameter_scan
		try:
			import matplotlib.pyplot as plt
		except ImportError as exc:
			raise ImportError("matplotlib is required for plotting. Install it via: pip install matplotlib") from exc

		plt.figure(figsize=(8, 4.5))
		plt.plot(param_values, lyap_values, color="tab:blue", linewidth=1.2)
		plt.axhline(0.0, color="tab:red", linestyle="--", linewidth=1.0)
		plt.xlabel(f"Scanned parameter: {parameter}")
		plt.ylabel("Lyapunov exponent")
		plt.title(f"Lyapunov exponent curve vs {parameter}")
		plt.grid(True, alpha=0.3)
		plt.tight_layout()
		plt.show()

		return param_values, lyap_values


if __name__ == "__main__":


	def logistic_map(x: sp.Symbol, p: dict[str, sp.Symbol]) -> sp.Expr:
		return p["r"] * x * (1 - x)


	system_logistic = Chaotic1DSystem(params={"r": 3.6}, map_expr_builder=logistic_map)
	r_values = np.linspace(2.8, 4.0, 160)
	scan_r, scan_lyap_r = system_logistic.lyapunov_parameter_scan(
		parameter="r",
		values=r_values,
		x0=0.123456,
		n=3000,
		discard=1000,
	)
	print("Logistic scan [r, lyapunov] =")
	print(np.column_stack((scan_r, scan_lyap_r)))
	pass
