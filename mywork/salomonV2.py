from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path
import time
import numpy as np
import sympy as sp
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

try:
	from numba import njit, prange
except ImportError:  # pragma: no cover - fallback keeps the module usable without numba.
	njit = None
	prange = range


if njit is not None:
	@njit(cache=True)
	def _step_numba(
		x: np.ndarray,
		z: float,
		p_idx: np.ndarray,
		q_idx: np.ndarray,
		mu: float,
		lam: float,
		a: float,
		b: float,
		alpha: float,
		beta: float,
		is_mod: bool,
	) -> tuple[np.ndarray, float]:
		L = x.size
		fx = np.empty(L, dtype=np.float64)
		x_next = np.empty(L, dtype=np.float64)
		factor_x = 5.0 + 3.0 * mu
		factor_z = 5.0 + 3.0 * lam
		inner_factor = 15.0 * math.pi
		angle_factor = 2.0 * math.pi
		alpha_scale = 10.0 ** alpha
		beta_scale = 10.0 ** beta

		for i in range(L):
			xi = x[i]
			fx[i] = abs(math.sin(factor_x * (1.0 - (a * xi * math.sin(inner_factor * xi * (1.0 - xi))))))

		for i in range(L):
			left = fx[i - 1] if i > 0 else fx[L - 1]
			right = fx[i + 1] if i < L - 1 else fx[0]
			fp = fx[p_idx[i]]
			fq = fx[q_idx[i]]
			value = alpha_scale - math.cos(angle_factor * (left + fx[i] + right))
			value += beta_scale * math.sqrt(fp * fp + fq * fq)
			if is_mod:
				value = value % 1.0
			x_next[i] = value

		z_next = abs(math.sin(factor_z * (1.0 - (b * z * math.sin(inner_factor * z * (1.0 - z))))))
		if is_mod:
			z_next = z_next % 1.0
		return x_next, float(z_next)

	@njit(parallel=True, cache=True)
	def _generate_rdseq_numba(
		x0: np.ndarray,
		p_idx: np.ndarray,
		q_idx: np.ndarray,
		mu: float,
		a: float,
		alpha: float,
		beta: float,
		is_mod: bool,
		N: int,
	) -> np.ndarray:
		L = x0.size
		x = x0.copy()
		fx = np.empty(L, dtype=np.float64)
		x_next = np.empty(L, dtype=np.float64)
		x_values = np.empty((N, L), dtype=np.float64)
		factor = 5.0 + 3.0 * mu
		inner_factor = 15.0 * math.pi
		angle_factor = 2.0 * math.pi

		for t in range(N):
			for i in prange(L):
				xi = x[i]
				fx[i] = abs(math.sin(factor * (1.0 - (a * xi * math.sin(inner_factor * xi * (1.0 - xi))))))

			for i in prange(L):
				left = fx[i - 1] if i > 0 else fx[L - 1]
				right = fx[i + 1] if i < L - 1 else fx[0]
				fp = fx[p_idx[i]]
				fq = fx[q_idx[i]]
				value = (10.0 ** alpha) - math.cos(angle_factor * (left + fx[i] + right))
				value += (10.0 ** beta) * math.sqrt(fp * fp + fq * fq)
				if is_mod:
					value = value % 1.0
				x_next[i] = value
				x_values[t, i] = value

			tmp = x
			x = x_next
			x_next = tmp

		return x_values
else:
	_step_numba = None
	_generate_rdseq_numba = None

class SalomoncouplingCML:
	"""Salomon coupling CML with non-adjacent p/q indices.

	Core update rule:
		x_{n+1}(i) = 10^alpha - cos(2*pi*(f(x_{i-1}) + f(x_i) + f(x_{i+1})))
					 + 10^beta*sqrt(f(x_p)^2 + f(x_q)^2)
	f(x) = |sin((5 + 3 * mu) * (1 - (a * x * sin(15 * pi * x * (1 - x)))))|
	p/q index rule:
		p = ((1 + xi) * i) % L
		q = ((eta + xi*eta + 1) * i) % L
	"""

	def __init__(
		self,
		L: int,
		params: dict[str, float],
		initstate: dict[str, np.ndarray | float],
		is_mod: bool = True,
	) -> None:
		if int(L) <= 0:
			raise ValueError("L must be a positive integer.")

		required = {"mu", "lam", "a", "b", "alpha", "beta", "xi", "eta"}
		missing = required - set(params.keys())
		if missing:
			raise ValueError(f"Missing required params: {sorted(missing)}")

		if "x0" not in initstate or "z0" not in initstate:
			raise ValueError("initstate must contain keys 'x0' and 'z0'.")

		self.L = int(L)
		self.mu = float(params["mu"])
		self.lam = float(params["lam"])
		self.a = float(params["a"])
		self.b = float(params["b"])
		self.alpha = float(params["alpha"])
		self.beta = float(params["beta"])
		self.xi = int(params["xi"])
		self.eta = int(params["eta"])
		self.is_mod = bool(is_mod)

		self.x0 = np.asarray(initstate["x0"], dtype=float).copy()
		self.z0 = float(initstate["z0"])
		if self.x0.size != self.L:
			raise ValueError(f"x0 length must equal L={self.L}")

		self.original_params = {
			"mu": self.mu,
			"lam": self.lam,
			"a": self.a,
			"b": self.b,
			"alpha": self.alpha,
			"beta": self.beta,
			"xi": self.xi,
			"eta": self.eta,
		}
		self.last_scan_path: str | None = None
		self.last_ie_scan_path: str | None = None
		self.last_ami_scan_path: str | None = None

		self._sync_index_rule()
		self._build_symbolic_functions()

	@staticmethod
	def _salomon_f_expr(x: sp.Symbol, mu: sp.Symbol, a: sp.Symbol) -> sp.Expr:
		return sp.Abs(sp.sin((5 + 3 * mu) * (1 - (a * x * sp.sin(15 * sp.pi * x * (1 - x))))))

	@staticmethod
	def _salomon_g_expr(z: sp.Symbol, lam: sp.Symbol, b: sp.Symbol) -> sp.Expr:
		return sp.Abs(sp.sin((5 + 3 * lam) * (1 - (b * z * sp.sin(15 * sp.pi * z * (1 - z))))))

	def _build_symbolic_functions(self) -> None:
		x = sp.Symbol("x", real=True)
		z = sp.Symbol("z", real=True)
		mu = sp.Symbol("mu", real=True)
		lam = sp.Symbol("lam", real=True)
		a = sp.Symbol("a", real=True)
		b = sp.Symbol("b", real=True)

		f_expr = self._salomon_f_expr(x, mu, a)
		g_expr = self._salomon_g_expr(z, lam, b)
		f_diff_expr = sp.diff(f_expr, x)

		self._f = sp.lambdify((x, mu, a), f_expr, modules="numpy")
		self._g = sp.lambdify((z, lam, b), g_expr, modules="numpy")
		self._f_diff = sp.lambdify((x, mu, a), f_diff_expr, modules="numpy")

	def _build_neighbor_indices(self) -> None:
		i = np.arange(self.L, dtype=int)
		p = ((1 + self.xi) * i) % self.L
		q = ((self.eta + self.xi * self.eta + 1) * i) % self.L
		self._p_idx = p.astype(int)
		self._q_idx = q.astype(int)

	def _set_param_value(self, name: str, value: float) -> None:
		if not hasattr(self, name):
			raise ValueError(f"Unknown parameter: {name}")
		if name in ("xi", "eta"):
			setattr(self, name, int(value))
		else:
			setattr(self, name, float(value))

	def _sync_index_rule(self) -> None:
		if self.xi == 0:
			self.eta = self.L
		if self.eta == 0:
			self.xi = self.L
		self._build_neighbor_indices()

	def _reset_params(self) -> None:
		for key, value in self.original_params.items():
			self._set_param_value(key, value)
		self._sync_index_rule()

	@staticmethod
	def _timestamped_path(path: Path) -> Path:
		stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		return path.with_name(f"{path.stem}_{stamp}{path.suffix}")

	def f(self, x: np.ndarray | float) -> np.ndarray | float:
		return self._f(x, self.mu, self.a)

	def g(self, z: float) -> float:
		return float(self._g(z, self.lam, self.b))

	def _f_prime(self, x: np.ndarray) -> np.ndarray:
		return np.asarray(self._f_diff(x, self.mu, self.a), dtype=float)

	def step(self, x: np.ndarray, z: float) -> tuple[np.ndarray, float]:
		x = np.asarray(x, dtype=float)
		if x.size != self.L:
			raise ValueError(f"x length must equal L={self.L}")

		z = float(z)
		if _step_numba is not None:
			return _step_numba(
				x=x,
				z=z,
				p_idx=np.asarray(self._p_idx, dtype=np.int64),
				q_idx=np.asarray(self._q_idx, dtype=np.int64),
				mu=float(self.mu),
				lam=float(self.lam),
				a=float(self.a),
				b=float(self.b),
				alpha=float(self.alpha),
				beta=float(self.beta),
				is_mod=bool(self.is_mod),
			)

		fx = np.asarray(self.f(x), dtype=float)
		fx_left = np.roll(fx, 1)
		fx_right = np.roll(fx, -1)
		fx_p = fx[self._p_idx]
		fx_q = fx[self._q_idx]

		x_next = (10.0 ** self.alpha) - np.cos(2.0 * np.pi * (fx_left + fx + fx_right))
		x_next += (10.0 ** self.beta) * np.sqrt(fx_p * fx_p + fx_q * fx_q)

		if self.is_mod:
			x_next = np.mod(x_next, 1.0)

		z_next = np.mod(self.g(z), 1.0)
		return x_next, float(z_next)


	def iterate_states(
		self,
		x0: np.ndarray,
		z0: float,
		arrL: int,
	) -> np.ndarray:
		"""Iterate the map arrL times and return all generated x states.

		Each iteration produces one length-L state vector, so the returned array
		has shape (arrL, L).
		"""
		if not isinstance(arrL, int) or arrL <= 0:
			raise ValueError("arrL must be a positive integer")

		x = np.asarray(x0, dtype=float).copy()
		z = float(z0)
		if x.size != self.L:
			raise ValueError(f"x0 length must equal L={self.L}")

		states = np.empty((arrL, self.L), dtype=float)
		for i in range(arrL):
			x, z = self.step(x, z)
			states[i, :] = x

		return states

	#sym:generate_random_bits_file
	def generate_random_bits_file(
		self,
		n_bits: int,
		save_path: str = "mywork/output/salomonV2_random.bin",
		x0: np.ndarray | None = None,
		z0: float | None = None,
		warmup: int = 200,
		scale_factor: float = 10**10,
		bitorder: str = "little",
		mode: str = "overall",
		lattice_index: int = 94,
	) -> Path:
		"""Generate random bits via high-bit extraction and save to a binary file.

		Modes:
			- ``overall``: encode all ``L`` lattice values each step.
			- ``lattice``: encode only one lattice value per step.

		Extraction rule per state value x_i:
			byte_i = floor((x_i mod 1) * scale_factor) % 256
		Each byte_i is expanded to 8 bits via np.unpackbits using `bitorder`.

		Note:
			`threshold` is kept only for backward compatibility and is ignored.
		"""
		if not isinstance(n_bits, int) or n_bits <= 0:
			raise ValueError("n_bits must be a positive integer")
		if not isinstance(warmup, int) or warmup < 0:
			raise ValueError("warmup must be a non-negative integer")
		if not np.isfinite(scale_factor) or scale_factor <= 0:
			raise ValueError("scale_factor must be a positive finite number")
		if bitorder not in ("little", "big"):
			raise ValueError("bitorder must be 'little' or 'big'")

		mode_normalized = str(mode).strip().lower()
		if mode_normalized not in ("overall", "lattice"):
			raise ValueError("mode must be 'overall' or 'lattice'")
		if not isinstance(lattice_index, int):
			raise ValueError("lattice_index must be an integer")
		if not 0 <= lattice_index < self.L:
			raise ValueError(f"lattice_index must be in [0, {self.L - 1}]")

		path = Path(save_path)
		path.parent.mkdir(parents=True, exist_ok=True)

		if path.exists():
			print(f"[prng] File already exists: {path}")
			while True:
				choice = input("Choose action: [s]kip / [d]elete and regenerate: ").strip().lower()
				if choice in ("s", "skip"):
					print("[prng] Skip generation and keep existing file.")
					return path
				if choice in ("d", "delete"):
					path.unlink()
					print("[prng] Existing file deleted. Start generating.")
					break
				print("Invalid input. Please type 's' or 'd'.")

		x = np.asarray(self.x0 if x0 is None else x0, dtype=float).copy()
		z = float(self.z0 if z0 is None else z0)
		if x.size != self.L:
			raise ValueError(f"x0 length must equal L={self.L}")

		for _ in range(warmup):
			x, z = self.step(x, z)

		if mode_normalized == "overall":
			bits_per_step = self.L * 8
			steps = (n_bits + bits_per_step - 1) // bits_per_step
			bits = np.empty(steps * bits_per_step, dtype=np.uint8)
			pos = 0

			with Progress(
				TextColumn("[bold cyan]{task.description}"),
				BarColumn(),
				TaskProgressColumn(),
				TimeElapsedColumn(),
				TimeRemainingColumn(),
			) as progress:
				task = progress.add_task("Generating Salomon random bits (overall)", total=steps)
				for _ in range(steps):
					x, z = self.step(x, z)
					x_frac = np.mod(x, 1.0)
					x_scaled = np.floor(x_frac * scale_factor).astype(np.uint64)
					row_bytes = np.mod(x_scaled, 256).astype(np.uint8)
					row_bits = np.unpackbits(row_bytes, bitorder=bitorder)
					bits[pos:pos + row_bits.size] = row_bits
					pos += row_bits.size
					progress.update(task, advance=1)
		else:
			bits_per_step = 8
			steps = (n_bits + bits_per_step - 1) // bits_per_step
			bits = np.empty(steps * bits_per_step, dtype=np.uint8)
			pos = 0

			with Progress(
				TextColumn("[bold cyan]{task.description}"),
				BarColumn(),
				TaskProgressColumn(),
				TimeElapsedColumn(),
				TimeRemainingColumn(),
			) as progress:
				task = progress.add_task(
					f"Generating Salomon random bits (lattice {lattice_index + 1})",
					total=steps,
				)
				for _ in range(steps):
					x, z = self.step(x, z)
					x_value = float(np.mod(x[lattice_index], 1.0))
					x_scaled = int(np.floor(x_value * scale_factor)) % 256
					row_bits = np.unpackbits(np.array([x_scaled], dtype=np.uint8), bitorder=bitorder)
					bits[pos:pos + 8] = row_bits
					pos += 8
					progress.update(task, advance=1)

		bits = bits[:n_bits]
		one_count = int(np.sum(bits))
		zero_count = int(bits.size - one_count)

		pad = (-n_bits) % 8
		if pad:
			bits = np.pad(bits, (0, pad), mode="constant", constant_values=0)

		packed = np.packbits(bits, bitorder=bitorder)
		path.write_bytes(packed.tobytes())

		print(f"[prng] Mode: {mode_normalized}")
		if mode_normalized == "lattice":
			print(f"[prng] Lattice index: {lattice_index}")
		print(f"[prng] Generated {n_bits} bits -> {packed.size} bytes")
		print(f"[prng] Saved to: {path}")
		print(f"[prng] Ones count: {one_count}")
		print(f"[prng] Zeros count: {zero_count}")
		return path

	def _jacobian_x(self, x: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
		"""Jacobian of the V2 x-update w.r.t. x.

		Note: when is_mod=True, this Jacobian ignores the discontinuous modulo wrap,
		which is standard in Lyapunov estimation for modulo maps (except measure-zero points).
		"""
		x = np.asarray(x, dtype=float)
		if x.size != self.L:
			raise ValueError(f"x length must equal L={self.L}")

		fx = np.asarray(self.f(x), dtype=float)
		fp = self._f_prime(x)

		fx_left = np.roll(fx, 1)
		fx_right = np.roll(fx, -1)
		S = fx_left + fx + fx_right
		sin_term = 2.0 * np.pi * np.sin(2.0 * np.pi * S)

		fx_p = fx[self._p_idx]
		fx_q = fx[self._q_idx]
		fp_p = fp[self._p_idx]
		fp_q = fp[self._q_idx]
		R = np.sqrt(fx_p * fx_p + fx_q * fx_q)
		inv_R = 1.0 / np.maximum(R, epsilon)
		coupling_scale = 10.0 ** self.beta

		J = np.zeros((self.L, self.L), dtype=float)
		for i in range(self.L):
			left_idx = (i - 1) % self.L
			center_idx = i
			right_idx = (i + 1) % self.L

			J[i, left_idx] += sin_term[i] * fp[left_idx]
			J[i, center_idx] += sin_term[i] * fp[center_idx]
			J[i, right_idx] += sin_term[i] * fp[right_idx]

			p_idx = int(self._p_idx[i])
			q_idx = int(self._q_idx[i])
			J[i, p_idx] += coupling_scale * (fx_p[i] * fp_p[i]) * inv_R[i]
			J[i, q_idx] += coupling_scale * (fx_q[i] * fp_q[i]) * inv_R[i]

		return J

	#sym:lyapunov_spectrum
	def lyapunov_spectrum(
		self,
		x0: np.ndarray,
		z0: float,
		n: int,
		discard: int = 100,
		epsilon: float = 1e-12,
	) -> np.ndarray:
		"""Return full Lyapunov spectrum (length L) in descending order."""
		if not isinstance(n, int) or n <= 0:
			raise ValueError("n must be a positive integer")
		if not isinstance(discard, int) or discard < 0:
			raise ValueError("discard must be a non-negative integer")
		if epsilon <= 0.0:
			raise ValueError("epsilon must be positive")

		x = np.asarray(x0, dtype=float).copy()
		z = float(z0)
		if x.size != self.L:
			raise ValueError(f"x0 length must equal L={self.L}")

		Q = np.eye(self.L, dtype=float)
		log_sum = np.zeros(self.L, dtype=float)

		total_steps = discard + n
		for step_idx in range(total_steps):
			J = self._jacobian_x(x, epsilon=epsilon)
			Z = J @ Q
			Q, R = np.linalg.qr(Z)

			if step_idx >= discard:
				d = np.abs(np.diag(R))
				log_sum += np.log(d + epsilon)

			x, z = self.step(x, z)

		spectrum = log_sum / float(n)
		return np.sort(spectrum)[::-1]

	def lyap_scan(
		self,
		param1: str,
		values1: np.ndarray,
		param2: str,
		values2: np.ndarray,
		x0: np.ndarray,
		z0: float,
		n: int,
		discard: int = 100,
		epsilon: float = 1e-12,
		save_path: str = "mywork/output/salomon_lyapunov_scan.npz",
		timestamp_on_exists: bool = False,
	) -> np.ndarray:
		"""Scan two parameters and save full Lyapunov spectra for each grid point.

		Returns:
			spectra: shape = (len(values1), len(values2), L)
		"""
		if not hasattr(self, param1):
			raise ValueError(f"Unknown parameter: {param1}")
		if not hasattr(self, param2):
			raise ValueError(f"Unknown parameter: {param2}")

		path = Path(save_path)
		path.parent.mkdir(parents=True, exist_ok=True)

		if path.exists():
			if timestamp_on_exists:
				new_path = self._timestamped_path(path)
				print(f"[scan] File already exists: {path}")
				print(f"[scan] timestamp_on_exists=True, save to: {new_path}")
				path = new_path
				path.parent.mkdir(parents=True, exist_ok=True)
			else:
				print(f"[scan] File already exists: {path}")
				while True:
					choice = input("Choose action: [s]kip / [d]elete and rescan: ").strip().lower()
					if choice in ("s", "skip"):
						print("[scan] Skip current scan. Loading existing spectra from file.")
						with np.load(path) as existing:
							if "spectra" not in existing:
								raise KeyError(f"'spectra' not found in existing file: {path}")
							self.last_scan_path = str(path)
							return np.asarray(existing["spectra"], dtype=float)
					if choice in ("d", "delete"):
						path.unlink()
						print("[scan] Existing file deleted. Start new scan.")
						break
					print("Invalid input. Please type 's' or 'd'.")

		v1 = np.asarray(values1, dtype=float)
		v2 = np.asarray(values2, dtype=float)
		if v1.ndim != 1 or v2.ndim != 1:
			raise ValueError("values1 and values2 must be 1D arrays")
		if v1.size == 0 or v2.size == 0:
			raise ValueError("values1 and values2 must not be empty")

		x0_arr = np.asarray(x0, dtype=float)
		if x0_arr.size != self.L:
			raise ValueError(f"x0 length must equal L={self.L}")

		spectra = np.empty((v1.size, v2.size, self.L), dtype=float)
		ked = np.empty((v1.size, v2.size), dtype=float)
		keb = np.empty((v1.size, v2.size), dtype=float)

		total = int(v1.size * v2.size)
		try:
			with Progress(
				TextColumn("[bold cyan]{task.description}"),
				BarColumn(),
				TaskProgressColumn(),
				TimeElapsedColumn(),
				TimeRemainingColumn(),
			) as progress:
				task = progress.add_task(f"Scanning Lyapunov {param1} x {param2}", total=total)

				for i, p1 in enumerate(v1):
					for j, p2 in enumerate(v2):
						self._set_param_value(param1, float(p1))
						self._set_param_value(param2, float(p2))
						self._sync_index_rule()

						spectra[i, j, :] = self.lyapunov_spectrum(
							x0=x0_arr,
							z0=float(z0),
							n=n,
							discard=discard,
							epsilon=epsilon,
						)
						ked_, keb_ = self.ked_keb(spectra[i, j, :])
						ked[i, j] = ked_
						keb[i, j] = keb_

						progress.update(task, advance=1)
		finally:
			self._reset_params()

		np.savez_compressed(
			path,
			spectra=spectra,
			ked=ked,
			keb=keb,
			param1_name=param1,
			param2_name=param2,
			param1_values=v1,
			param2_values=v2,
		)
		self.last_scan_path = str(path)
		return spectra

	@staticmethod
	def ked_keb(spectrum: np.ndarray) -> tuple[float, float]:
		"""Compute KED and KEB from one Lyapunov spectrum vector."""
		lam = np.asarray(spectrum, dtype=float).reshape(-1)
		positive = lam[lam > 0.0]
		N = lam.size
		if N == 0:
			return 0.0, 0.0

		ked = float(np.sum(positive) / N)
		keb = float(positive.size / N)
		return ked, keb

	def average_information_entropy(
		self,
		states: np.ndarray,
		n_states: int = 10,
	) -> tuple[float, np.ndarray]:
		"""Compute per-lattice Shannon entropy H(j) and average entropy Hd.

		The state values are discretized into `n_states` bins on [0, 1):
			state = floor((x mod 1) * n_states)
		and then
			H(j) = -sum_i p_i log2(p_i),
			Hd = mean_j H(j).
		"""
		states_arr = np.asarray(states, dtype=float)
		if states_arr.ndim != 2 or states_arr.shape[1] != self.L:
			raise ValueError(f"states must have shape (T, L) with L={self.L}")
		if states_arr.shape[0] == 0:
			raise ValueError("states must contain at least one time step")
		if not isinstance(n_states, int) or n_states <= 1:
			raise ValueError("n_states must be an integer greater than 1")

		x_mod = np.mod(states_arr, 1.0)
		state_idx = np.floor(x_mod * float(n_states)).astype(np.int64)
		state_idx = np.clip(state_idx, 0, n_states - 1)

		h_each = np.empty(self.L, dtype=float)
		sample_count = float(state_idx.shape[0])
		for j in range(self.L):
			counts = np.bincount(state_idx[:, j], minlength=n_states).astype(float)
			p = counts / sample_count
			nz = p > 0.0
			h_each[j] = float(-np.sum(p[nz] * np.log2(p[nz])))

		hd = float(np.mean(h_each))
		return hd, h_each

	#sym:IE
	def IE(
		self,
		param_name: str = "lam",
		param_range: np.ndarray | None = None,
		x0: np.ndarray | None = None,
		z0: float | None = None,
		n: int = 1000,
		discard: int = 200,
		n_states: int = 10,
		save_path: str = "mywork/output/salomon_ie_scan_mu_fixed.npz",
		timestamp_on_exists: bool = False,
		plot: bool = True,
		save_fig_path: str | None = None,
	) -> tuple[np.ndarray, np.ndarray]:
		"""Scan one parameter and compute IE(i, p) and Hd(p).

		Paper-aligned definitions:
			H(j) = -sum_i p(v_i)log2(p(v_i))
			Hd = (1/L) * sum_j H(j)

		Returns:
			ie_grid: shape = (len(param_range), L)
			hd_vec: shape = (len(param_range),)
		"""
		if not hasattr(self, param_name):
			raise ValueError(f"Unknown parameter: {param_name}")
		if not isinstance(n, int) or n <= 0:
			raise ValueError("n must be a positive integer")
		if not isinstance(discard, int) or discard < 0:
			raise ValueError("discard must be a non-negative integer")
		if discard >= n:
			raise ValueError(f"discard ({discard}) must be less than n ({n})")
		if not isinstance(n_states, int) or n_states <= 1:
			raise ValueError("n_states must be an integer greater than 1")

		if param_range is None:
			param_range = np.linspace(0.0, 1.0, 41)

		v_param = np.asarray(param_range, dtype=float)
		if v_param.ndim != 1:
			raise ValueError("param_range must be a 1D array")
		if v_param.size == 0:
			raise ValueError("param_range must not be empty")

		x0_arr = np.asarray(self.x0 if x0 is None else x0, dtype=float).copy()
		z0_val = float(self.z0 if z0 is None else z0)
		if x0_arr.size != self.L:
			raise ValueError(f"x0 length must equal L={self.L}")

		path = Path(save_path)
		path.parent.mkdir(parents=True, exist_ok=True)

		if path.exists():
			if timestamp_on_exists:
				new_path = self._timestamped_path(path)
				print(f"[ie] File already exists: {path}")
				print(f"[ie] timestamp_on_exists=True, save to: {new_path}")
				path = new_path
				path.parent.mkdir(parents=True, exist_ok=True)
			else:
				print(f"[ie] File already exists: {path}")
				while True:
					choice = input("Choose action: [s]kip / [d]elete and rescan: ").strip().lower()
					if choice in ("s", "skip"):
						print("[ie] Skip current scan. Loading existing IE grid from file.")
						with np.load(path) as existing:
							if "ie_grid" not in existing or "hd_vec" not in existing:
								raise KeyError(f"'ie_grid' or 'hd_vec' not found in existing file: {path}")
							ie_existing = np.asarray(existing["ie_grid"], dtype=float)
							hd_existing = np.asarray(existing["hd_vec"], dtype=float)
						self.last_ie_scan_path = str(path)
						if plot:
							self.plot_IE_wireframe(data_path=str(path), save_fig_path=save_fig_path)
						return ie_existing, hd_existing
					if choice in ("d", "delete"):
						path.unlink()
						print("[ie] Existing file deleted. Start new scan.")
						break
					print("Invalid input. Please type 's' or 'd'.")

		ie_grid = np.empty((v_param.size, self.L), dtype=float)
		hd_vec = np.empty(v_param.size, dtype=float)

		try:
			with Progress(
				TextColumn("[bold cyan]{task.description}"),
				BarColumn(),
				TaskProgressColumn(),
				TimeElapsedColumn(),
				TimeRemainingColumn(),
			) as progress:
				task = progress.add_task(f"Scanning IE(i, {param_name})", total=int(v_param.size))

				for i, p in enumerate(v_param):
					self._set_param_value(param_name, float(p))
					self._sync_index_rule()

					x = x0_arr.copy()
					z = z0_val

					for _ in range(discard):
						x, z = self.step(x, z)

					states = np.empty((n, self.L), dtype=float)
					for t in range(n):
						x, z = self.step(x, z)
						states[t, :] = x

					hd, h_each = self.average_information_entropy(states=states, n_states=n_states)
					ie_grid[i, :] = h_each
					hd_vec[i] = hd

					progress.update(task, advance=1)
		finally:
			self._reset_params()

		np.savez_compressed(
			path,
			ie_grid=ie_grid,
			hd_vec=hd_vec,
			param_name=param_name,
			param_values=v_param,
			n=n,
			discard=discard,
			n_states=n_states,
			L=self.L,
			mu=float(self.original_params["mu"]),
			lam=float(self.original_params["lam"]),
			a=float(self.original_params["a"]),
			b=float(self.original_params["b"]),
			xi=int(self.original_params["xi"]),
			eta=int(self.original_params["eta"]),
		)
		self.last_ie_scan_path = str(path)
		if plot:
			self.plot_IE_wireframe(data_path=str(path), save_fig_path=save_fig_path)
		return ie_grid, hd_vec

	@staticmethod
	def _entropy_from_counts(counts: np.ndarray) -> float:
		"""Shannon entropy from histogram counts in bits."""
		counts_arr = np.asarray(counts, dtype=float)
		total = float(np.sum(counts_arr))
		if total <= 0.0:
			return 0.0
		p = counts_arr / total
		nz = p > 0.0
		return float(-np.sum(p[nz] * np.log2(p[nz])))

	def _mutual_information_discrete_pair(
		self,
		x_i: np.ndarray,
		x_j: np.ndarray,
		n_states: int,
	) -> float:
		"""Mutual information I(S(i), S(j)) for two discretized lattice sequences."""
		xi = np.asarray(x_i, dtype=np.int64).reshape(-1)
		xj = np.asarray(x_j, dtype=np.int64).reshape(-1)
		if xi.size != xj.size:
			raise ValueError("x_i and x_j must have the same length")
		if xi.size == 0:
			return 0.0

		counts_i = np.bincount(xi, minlength=n_states)
		counts_j = np.bincount(xj, minlength=n_states)
		joint_idx = xi * n_states + xj
		joint_counts = np.bincount(joint_idx, minlength=n_states * n_states)

		h_i = self._entropy_from_counts(counts_i)
		h_j = self._entropy_from_counts(counts_j)
		h_ij = self._entropy_from_counts(joint_counts)

		mi = h_i + h_j - h_ij
		if mi < 0.0 and abs(mi) < 1e-12:
			mi = 0.0
		return float(max(mi, 0.0))

	def average_mutual_information(
		self,
		states: np.ndarray,
		n_states: int = 10,
	) -> tuple[float, np.ndarray]:
		"""Compute average mutual information Ld across all lattice pairs.

		Paper definition:
			I(S(i), S(j)) = F(S(i)) - F(S(i)|S(j))
			Ld = sum_{i=1..L} sum_{j=i+1..L} I(S(i), S(j)) / (L * (L - 1))
		"""
		states_arr = np.asarray(states, dtype=float)
		if states_arr.ndim != 2 or states_arr.shape[1] != self.L:
			raise ValueError(f"states must have shape (T, L) with L={self.L}")
		if states_arr.shape[0] == 0:
			raise ValueError("states must contain at least one time step")
		if not isinstance(n_states, int) or n_states <= 1:
			raise ValueError("n_states must be an integer greater than 1")

		x_mod = np.mod(states_arr, 1.0)
		state_idx = np.floor(x_mod * float(n_states)).astype(np.int64)
		state_idx = np.clip(state_idx, 0, n_states - 1)

		mi_upper = np.zeros((self.L, self.L), dtype=float)
		sum_mi = 0.0
		for i in range(self.L):
			x_i = state_idx[:, i]
			for j in range(i + 1, self.L):
				mi_ij = self._mutual_information_discrete_pair(x_i, state_idx[:, j], n_states=n_states)
				mi_upper[i, j] = mi_ij
				sum_mi += mi_ij

		den = float(self.L * (self.L - 1))
		ld = float(sum_mi / den) if den > 0.0 else 0.0
		return ld, mi_upper

	def AMI_scan(
		self,
		param1: str = "mu",
		values1: np.ndarray | None = None,
		param2: str = "lam",
		values2: np.ndarray | None = None,
		x0: np.ndarray | None = None,
		z0: float | None = None,
		n: int = 1000,
		discard: int = 200,
		n_states: int = 10,
		save_path: str = "mywork/output/salomon_ami_scan.npz",
		timestamp_on_exists: bool = False,
		plot: bool = True,
		save_fig_path: str | None = None,
	) -> np.ndarray:
		"""Scan two parameters, compute average mutual information Ld, and save it.

		The saved surface follows the paper-style definition used by
		``average_mutual_information``: each grid point stores the average pairwise
		mutual information among lattice sequences after the transient is discarded.
		"""
		if not hasattr(self, param1):
			raise ValueError(f"Unknown parameter: {param1}")
		if not hasattr(self, param2):
			raise ValueError(f"Unknown parameter: {param2}")
		if param1 == param2:
			raise ValueError("param1 and param2 must be different parameters")
		if not isinstance(n, int) or n <= 0:
			raise ValueError("n must be a positive integer")
		if not isinstance(discard, int) or discard < 0:
			raise ValueError("discard must be a non-negative integer")
		if discard >= n:
			raise ValueError(f"discard ({discard}) must be less than n ({n})")
		if not isinstance(n_states, int) or n_states <= 1:
			raise ValueError("n_states must be an integer greater than 1")

		if values1 is None:
			values1 = np.linspace(3.6, 4.0, 41)
		if values2 is None:
			values2 = np.linspace(0.0, 1.0, 41)

		v1 = np.asarray(values1, dtype=float)
		v2 = np.asarray(values2, dtype=float)
		if v1.ndim != 1 or v2.ndim != 1:
			raise ValueError("values1 and values2 must be 1D arrays")
		if v1.size == 0 or v2.size == 0:
			raise ValueError("values1 and values2 must not be empty")

		for name, values in ((param1, v1), (param2, v2)):
			if name in ("xi", "eta") and not np.allclose(values, np.rint(values)):
				raise ValueError(f"{name} scan values must be integers")

		x0_arr = np.asarray(self.x0 if x0 is None else x0, dtype=float).copy()
		z0_val = float(self.z0 if z0 is None else z0)
		if x0_arr.size != self.L:
			raise ValueError(f"x0 length must equal L={self.L}")

		path = Path(save_path)
		path.parent.mkdir(parents=True, exist_ok=True)

		if path.exists():
			if timestamp_on_exists:
				new_path = self._timestamped_path(path)
				print(f"[ami] File already exists: {path}")
				print(f"[ami] timestamp_on_exists=True, save to: {new_path}")
				path = new_path
				path.parent.mkdir(parents=True, exist_ok=True)
			else:
				print(f"[ami] File already exists: {path}")
				while True:
					choice = input("Choose action: [s]kip / [d]elete and rescan: ").strip().lower()
					if choice in ("s", "skip"):
						print("[ami] Skip current scan. Loading existing Ld grid from file.")
						with np.load(path) as existing:
							if "ld_grid" not in existing:
								raise KeyError(f"'ld_grid' not found in existing file: {path}")
							ld_existing = np.asarray(existing["ld_grid"], dtype=float)
						self.last_ami_scan_path = str(path)
						if plot:
							self.plot_avg_AMI_wireframe(data_path=str(path), save_fig_path=save_fig_path)
						return ld_existing
					if choice in ("d", "delete"):
						path.unlink()
						print("[ami] Existing file deleted. Start new scan.")
						break
					print("Invalid input. Please type 's' or 'd'.")

		ld_grid = np.empty((v1.size, v2.size), dtype=float)
		total = int(v1.size * v2.size)

		try:
			with Progress(
				TextColumn("[bold cyan]{task.description}"),
				BarColumn(),
				TaskProgressColumn(),
				TimeElapsedColumn(),
				TimeRemainingColumn(),
			) as progress:
				task = progress.add_task(f"Scanning AMI {param1} x {param2}", total=total)

				for i, p1 in enumerate(v1):
					for j, p2 in enumerate(v2):
						self._set_param_value(param1, float(p1))
						self._set_param_value(param2, float(p2))
						self._sync_index_rule()

						x = x0_arr.copy()
						z = z0_val
						for _ in range(discard):
							x, z = self.step(x, z)

						states = self.iterate_states(x0=x, z0=z, arrL=n)
						ld, _ = self.average_mutual_information(states=states, n_states=n_states)
						ld_grid[i, j] = ld

						progress.update(task, advance=1)
		finally:
			self._reset_params()

		np.savez_compressed(
			path,
			ld_grid=ld_grid,
			param1_name=param1,
			param2_name=param2,
			param1_values=v1,
			param2_values=v2,
			n=n,
			discard=discard,
			n_states=n_states,
			L=self.L,
			mu=self.mu,
			lam=self.lam,
			a=self.a,
			b=self.b,
			xi=self.xi,
			eta=self.eta,
			metric_name="average_mutual_information",
			metric_symbol="Ld",
		)
		self.last_ami_scan_path = str(path)
		if plot:
			self.plot_avg_AMI_wireframe(data_path=str(path), save_fig_path=save_fig_path)
		return ld_grid


	#sym:ie_scan_wireframe
	def avg_IE(
		self,
		param1: str = "mu",
		values1: np.ndarray | None = None,
		param2: str = "lam",
		values2: np.ndarray | None = None,
		x0: np.ndarray | None = None,
		z0: float | None = None,
		n: int = 1000,
		discard: int = 200,
		n_states: int = 10,
		save_path: str = "mywork/output/salomon_avg_ie_scan.npz",
		timestamp_on_exists: bool = False,
		plot: bool = True,
		save_fig_path: str | None = None,
	) -> np.ndarray:
		"""Scan two parameters and compute average information entropy Hd(p1, p2)."""
		if values1 is None:
			values1 = np.linspace(3.6, 4.0, 41)
		if values2 is None:
			values2 = np.linspace(0.0, 1.0, 41)

		hd_grid = self.IE_scan(
			param1=param1,
			values1=np.asarray(values1, dtype=float),
			param2=param2,
			values2=np.asarray(values2, dtype=float),
			x0=x0,
			z0=z0,
			n=n,
			discard=discard,
			n_states=n_states,
			save_path=save_path,
			timestamp_on_exists=timestamp_on_exists,
			plot=False,
		)

		data_path = self.last_ie_scan_path if self.last_ie_scan_path is not None else str(Path(save_path))
		if plot:
			self.plot_avg_IE_wireframe(data_path=data_path, save_fig_path=save_fig_path)
		return hd_grid

	def IE_scan(
		self,
		param1: str = "mu",
		values1: np.ndarray | None = None,
		param2: str = "lam",
		values2: np.ndarray | None = None,
		x0: np.ndarray | None = None,
		z0: float | None = None,
		n: int = 1000,
		discard: int = 200,
		n_states: int = 10,
		save_path: str = "mywork/output/salomon_ie_scan.npz",
		timestamp_on_exists: bool = False,
		plot: bool = True,
		save_fig_path: str | None = None,
	) -> np.ndarray:
		"""Scan two parameters, compute Hd, and optionally plot a wireframe.

		Default parameter ranges follow the paper-like setting:
			mu in [3.6, 4.0], e(=lam) in [0, 1].
		"""
		if not hasattr(self, param1):
			raise ValueError(f"Unknown parameter: {param1}")
		if not hasattr(self, param2):
			raise ValueError(f"Unknown parameter: {param2}")
		if not isinstance(n, int) or n <= 0:
			raise ValueError("n must be a positive integer")
		if not isinstance(discard, int) or discard < 0:
			raise ValueError("discard must be a non-negative integer")
		if discard >= n:
			raise ValueError(f"discard ({discard}) must be less than n ({n})")
		if not isinstance(n_states, int) or n_states <= 1:
			raise ValueError("n_states must be an integer greater than 1")

		if values1 is None:
			values1 = np.linspace(3.6, 4.0, 41)
		if values2 is None:
			values2 = np.linspace(0.0, 1.0, 41)

		v1 = np.asarray(values1, dtype=float)
		v2 = np.asarray(values2, dtype=float)
		if v1.ndim != 1 or v2.ndim != 1:
			raise ValueError("values1 and values2 must be 1D arrays")
		if v1.size == 0 or v2.size == 0:
			raise ValueError("values1 and values2 must not be empty")

		x0_arr = np.asarray(self.x0 if x0 is None else x0, dtype=float).copy()
		z0_val = float(self.z0 if z0 is None else z0)
		if x0_arr.size != self.L:
			raise ValueError(f"x0 length must equal L={self.L}")

		path = Path(save_path)
		path.parent.mkdir(parents=True, exist_ok=True)

		if path.exists():
			if timestamp_on_exists:
				new_path = self._timestamped_path(path)
				print(f"[ie] File already exists: {path}")
				print(f"[ie] timestamp_on_exists=True, save to: {new_path}")
				path = new_path
				path.parent.mkdir(parents=True, exist_ok=True)
			else:
				print(f"[ie] File already exists: {path}")
				while True:
					choice = input("Choose action: [s]kip / [d]elete and rescan: ").strip().lower()
					if choice in ("s", "skip"):
						print("[ie] Skip current scan. Loading existing Hd grid from file.")
						with np.load(path) as existing:
							if "hd_grid" not in existing:
								raise KeyError(f"'hd_grid' not found in existing file: {path}")
							hd_existing = np.asarray(existing["hd_grid"], dtype=float)
						self.last_ie_scan_path = str(path)
						if plot:
							self.plot_ie_wireframe(data_path=str(path), save_fig_path=save_fig_path)
						return hd_existing
					if choice in ("d", "delete"):
						path.unlink()
						print("[ie] Existing file deleted. Start new scan.")
						break
					print("Invalid input. Please type 's' or 'd'.")

		hd_grid = np.empty((v1.size, v2.size), dtype=float)
		total = int(v1.size * v2.size)

		try:
			with Progress(
				TextColumn("[bold cyan]{task.description}"),
				BarColumn(),
				TaskProgressColumn(),
				TimeElapsedColumn(),
				TimeRemainingColumn(),
			) as progress:
				task = progress.add_task(f"Scanning IE {param1} x {param2}", total=total)

				for i, p1 in enumerate(v1):
					for j, p2 in enumerate(v2):
						self._set_param_value(param1, float(p1))
						self._set_param_value(param2, float(p2))
						self._sync_index_rule()

						x = x0_arr.copy()
						z = z0_val

						for _ in range(discard):
							x, z = self.step(x, z)

						states = np.empty((n, self.L), dtype=float)
						for t in range(n):
							x, z = self.step(x, z)
							states[t, :] = x

						hd, _ = self.average_information_entropy(states=states, n_states=n_states)
						hd_grid[i, j] = hd

						progress.update(task, advance=1)
		finally:
			self._reset_params()

		np.savez_compressed(
			path,
			hd_grid=hd_grid,
			param1_name=param1,
			param2_name=param2,
			param1_values=v1,
			param2_values=v2,
			n=n,
			discard=discard,
			n_states=n_states,
		)
		self.last_ie_scan_path = str(path)
		return hd_grid

	def _plot_colored_wireframe(
		self,
		ax,
		X: np.ndarray,
		Y: np.ndarray,
		Z: np.ndarray,
		cmap_name: str = "viridis",
		linewidth: float = 0.8,
		alpha: float = 1.0,
	):
		"""Draw a wireframe with line colors mapped to local Z values."""
		import matplotlib.pyplot as plt
		from matplotlib.colors import Normalize
		from mpl_toolkits.mplot3d.art3d import Line3DCollection

		X_arr = np.asarray(X, dtype=float)
		Y_arr = np.asarray(Y, dtype=float)
		Z_arr = np.asarray(Z, dtype=float)
		if X_arr.shape != Y_arr.shape or X_arr.shape != Z_arr.shape:
			raise ValueError("X, Y and Z must have the same shape")

		finite_mask = np.isfinite(Z_arr)
		if not np.any(finite_mask):
			raise ValueError("Z must contain at least one finite value")

		z_min = float(np.nanmin(Z_arr))
		z_max = float(np.nanmax(Z_arr))
		if np.isclose(z_min, z_max):
			z_max = z_min + 1e-12

		norm = Normalize(vmin=z_min, vmax=z_max)
		cmap = plt.get_cmap(cmap_name)

		def _add_segments(points: np.ndarray) -> None:
			if points.shape[0] < 2:
				return
			segments = np.stack((points[:-1], points[1:]), axis=1)
			z_mid = 0.5 * (points[:-1, 2] + points[1:, 2])
			colors = cmap(norm(z_mid))
			collection = Line3DCollection(
				segments,
				colors=colors,
				linewidths=linewidth,
				alpha=alpha,
			)
			ax.add_collection3d(collection)

		for i in range(Z_arr.shape[0]):
			row_points = np.column_stack((X_arr[i, :], Y_arr[i, :], Z_arr[i, :]))
			_add_segments(row_points)

		for j in range(Z_arr.shape[1]):
			col_points = np.column_stack((X_arr[:, j], Y_arr[:, j], Z_arr[:, j]))
			_add_segments(col_points)

		return norm, cmap

	def plot_IE_wireframe(
		self,
		data_path: str,
		save_fig_path: str | None = None,
		title: str | None = None,
	) -> None:
		"""Load IE(i, p) scan data and draw IE 3D wireframe."""
		import matplotlib.pyplot as plt

		data = np.load(data_path)
		if "ie_grid" not in data:
			raise KeyError(f"'ie_grid' not found in file: {data_path}")

		ie_grid = np.asarray(data["ie_grid"], dtype=float)
		if "param_values" in data:
			param_values = np.asarray(data["param_values"], dtype=float)
		elif "param_range" in data:
			param_values = np.asarray(data["param_range"], dtype=float)
		else:
			raise KeyError(f"'param_values' not found in file: {data_path}")
		param_name = str(data["param_name"]) if "param_name" in data else "lam"

		if ie_grid.ndim != 2:
			raise ValueError("ie_grid must be a 2D array")
		if param_values.ndim != 1:
			raise ValueError("param_values must be a 1D array")
		if ie_grid.shape[0] != param_values.size:
			raise ValueError(
				f"ie_grid shape {ie_grid.shape} incompatible with param_values size {param_values.size}"
			)

		i_values = np.arange(ie_grid.shape[1], dtype=float)
		X, Y = np.meshgrid(i_values, param_values, indexing="xy")

		fig = plt.figure(figsize=(9, 7))
		ax = fig.add_subplot(111, projection="3d")
		norm, cmap = self._plot_colored_wireframe(
			ax=ax,
			X=X,
			Y=Y,
			Z=ie_grid,
			cmap_name="viridis",
			linewidth=0.9,
		)

		if title is None:
			title = f"Information Entropy Wireframe ({param_name})"
		ax.set_title(title)
		ax.set_xlabel("i")
		ax.set_ylabel(param_name)
		ax.set_zlabel("IE")
		ax.set_xlim(float(np.min(i_values)), float(np.max(i_values)))
		ax.set_ylim(float(np.min(param_values)), float(np.max(param_values)))

		z_min = float(np.nanmin(ie_grid))
		z_max = float(np.nanmax(ie_grid))
		if np.isclose(z_min, z_max):
			pad = 1e-8 if z_min == 0.0 else max(1e-8, abs(z_min) * 0.05)
			z_min -= pad
			z_max += pad
		ax.set_zlim(z_min, z_max)

		mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
		mappable.set_array([])
		fig.colorbar(mappable, ax=ax, shrink=0.7, pad=0.08, label="IE")

		ax.view_init(elev=24, azim=-128)
		ax.grid(True, alpha=0.35)

		plt.tight_layout()
		if save_fig_path is not None:
			fig_path = Path(save_fig_path)
			fig_path.parent.mkdir(parents=True, exist_ok=True)
			fig.savefig(fig_path, dpi=300, bbox_inches="tight")
			print(f"[ie] Saved IE wireframe figure: {fig_path}")
		plt.show()

	def plot_avg_IE_wireframe(
		self,
		data_path: str,
		save_fig_path: str | None = None,
		title: str | None = None,
		colorbar_range: tuple[float, float] | None = None,
	) -> None:
		"""Load avg_IE scan data and draw Hd(mu, e) 3D wireframe."""
		import matplotlib.pyplot as plt

		data = np.load(data_path)
		if "hd_grid" not in data:
			raise KeyError(f"'hd_grid' not found in file: {data_path}")

		hd_grid = np.asarray(data["hd_grid"], dtype=float)
		v1 = np.asarray(data["param1_values"], dtype=float)
		v2 = np.asarray(data["param2_values"], dtype=float)
		p1_name = str(data["param1_name"]) if "param1_name" in data else "mu"
		p2_name = str(data["param2_name"]) if "param2_name" in data else "lam"

		expected_shape = (v1.size, v2.size)
		if hd_grid.shape != expected_shape:
			raise ValueError(f"hd_grid shape {hd_grid.shape} does not match expected {expected_shape}")

		X, Y = np.meshgrid(v1, v2, indexing="ij")
		Z = hd_grid
		if colorbar_range is None:
			vmin, vmax = np.nanmin(Z), np.nanmax(Z)
		else:
			vmin, vmax = colorbar_range

		fig = plt.figure(figsize=(9, 7))
		ax = fig.add_subplot(111, projection="3d")
		norm, cmap = self._plot_colored_wireframe(
			ax=ax,
			X=X,
			Y=Y,
			Z=hd_grid,
			cmap_name="viridis",
			linewidth=0.9,
		)

		if title is None:
			title = f"Average Information Entropy Wireframe ({p1_name} vs {p2_name})"
		ax.set_title(title)
		ax.set_xlabel(p1_name)
		ax.set_ylabel("e (lam)" if p2_name == "lam" else p2_name)
		ax.set_zlabel("Hd")
		ax.set_xlim(float(np.min(v1)), float(np.max(v1)))
		ax.set_ylim(float(np.min(v2)), float(np.max(v2)))

		z_min = float(np.nanmin(hd_grid))
		z_max = float(np.nanmax(hd_grid))
		if np.isclose(z_min, z_max):
			pad = 1e-8 if z_min == 0.0 else max(1e-8, abs(z_min) * 0.05)
			z_min -= pad
			z_max += pad
		ax.set_zlim(z_min, z_max)

		mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
		mappable.set_array([])
		fig.colorbar(mappable, ax=ax, shrink=0.7, pad=0.08, label="Hd")

		ax.view_init(elev=24, azim=-128)
		ax.grid(True, alpha=0.35)

		plt.tight_layout()
		if save_fig_path is not None:
			fig_path = Path(save_fig_path)
			fig_path.parent.mkdir(parents=True, exist_ok=True)
			fig.savefig(fig_path, dpi=300, bbox_inches="tight")
			print(f"[avg_ie] Saved avg_IE wireframe figure: {fig_path}")
		plt.show()

	def plot_avg_AMI_wireframe(
		self,
		data_path: str,
		save_fig_path: str | None = None,
		title: str | None = None,
	) -> None:
		"""Load AMI scan data and draw Ld(mu, e) 3D wireframe."""
		import matplotlib.pyplot as plt

		data = np.load(data_path)
		if "ld_grid" not in data:
			raise KeyError(f"'ld_grid' not found in file: {data_path}")

		ld_grid = np.asarray(data["ld_grid"], dtype=float)
		v1 = np.asarray(data["param1_values"], dtype=float)
		v2 = np.asarray(data["param2_values"], dtype=float)
		p1_name = str(data["param1_name"]) if "param1_name" in data else "mu"
		p2_name = str(data["param2_name"]) if "param2_name" in data else "lam"

		expected_shape = (v1.size, v2.size)
		if ld_grid.shape != expected_shape:
			raise ValueError(f"ld_grid shape {ld_grid.shape} does not match expected {expected_shape}")

		X, Y = np.meshgrid(v1, v2, indexing="ij")

		fig = plt.figure(figsize=(9, 7))
		ax = fig.add_subplot(111, projection="3d")
		norm, cmap = self._plot_colored_wireframe(
			ax=ax,
			X=X,
			Y=Y,
			Z=ld_grid,
			cmap_name="viridis",
			linewidth=0.9,
		)

		if title is None:
			title = f"Average Mutual Information Wireframe ({p1_name} vs {p2_name})"
		ax.set_title(title)
		ax.set_xlabel(p1_name)
		ax.set_ylabel("e (lam)" if p2_name == "lam" else p2_name)
		ax.set_zlabel("Ld")
		ax.set_xlim(float(np.min(v1)), float(np.max(v1)))
		ax.set_ylim(float(np.min(v2)), float(np.max(v2)))

		z_min = float(np.nanmin(ld_grid))
		z_max = float(np.nanmax(ld_grid))
		if np.isclose(z_min, z_max):
			pad = 1e-8 if z_min == 0.0 else max(1e-8, abs(z_min) * 0.05)
			z_min -= pad
			z_max += pad
		ax.set_zlim(z_min, z_max)

		mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
		mappable.set_array([])
		fig.colorbar(mappable, ax=ax, shrink=0.7, pad=0.08, label="Ld")

		ax.view_init(elev=24, azim=-128)
		ax.grid(True, alpha=0.35)

		plt.tight_layout()
		if save_fig_path is not None:
			fig_path = Path(save_fig_path)
			fig_path.parent.mkdir(parents=True, exist_ok=True)
			fig.savefig(fig_path, dpi=300, bbox_inches="tight")
			print(f"[ami] Saved AMI wireframe figure: {fig_path}")
		plt.show()

	def plot_ie_wireframe(
		self,
		data_path: str,
		save_fig_path: str | None = None,
		title: str | None = None,
	) -> None:
		"""Load IE scan data and draw the Hd 3D wireframe."""
		import matplotlib.pyplot as plt

		data = np.load(data_path)
		if "hd_grid" not in data:
			raise KeyError(f"'hd_grid' not found in file: {data_path}")

		hd_grid = np.asarray(data["hd_grid"], dtype=float)
		v1 = np.asarray(data["param1_values"], dtype=float)
		v2 = np.asarray(data["param2_values"], dtype=float)
		p1_name = str(data["param1_name"])
		p2_name = str(data["param2_name"])
		n_states = int(data["n_states"]) if "n_states" in data else 10

		expected_shape = (v1.size, v2.size)
		if hd_grid.shape != expected_shape:
			raise ValueError(f"hd_grid shape {hd_grid.shape} does not match expected {expected_shape}")

		X, Y = np.meshgrid(v1, v2, indexing="ij")

		fig = plt.figure(figsize=(9, 7))
		ax = fig.add_subplot(111, projection="3d")
		norm, cmap = self._plot_colored_wireframe(
			ax=ax,
			X=X,
			Y=Y,
			Z=hd_grid,
			cmap_name="viridis",
			linewidth=0.9,
		)

		if title is None:
			title = f"Average Information Entropy Wireframe ({p1_name} vs {p2_name})"
		ax.set_title(title)
		ax.set_xlabel(p1_name)
		ax.set_ylabel("e (lam)" if p2_name == "lam" else p2_name)
		ax.set_zlabel("Hd")
		ax.set_xlim(float(np.min(v1)), float(np.max(v1)))
		ax.set_ylim(float(np.min(v2)), float(np.max(v2)))
		ax.set_zlim(3, float(np.log2(max(n_states, 2))))
		mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
		mappable.set_array([])
		fig.colorbar(mappable, ax=ax, shrink=0.7, pad=0.08, label="Hd")

		ax.view_init(elev=24, azim=-128)
		ax.grid(True, alpha=0.35)

		plt.tight_layout()
		if save_fig_path is not None:
			fig_path = Path(save_fig_path)
			fig_path.parent.mkdir(parents=True, exist_ok=True)
			fig.savefig(fig_path, dpi=300, bbox_inches="tight")
			print(f"[ie] Saved wireframe figure: {fig_path}")
		plt.show()


	#sym:plot_ked_keb
	def plot_ked_keb(self, data_path: str,save_fig_path: str | None = None) -> None:
		"""Load saved scan data and plot KED/KEB 3D wireframes."""
		import matplotlib.pyplot as plt

		data = np.load(data_path)
		ked = np.asarray(data["ked"], dtype=float)
		keb = np.asarray(data["keb"], dtype=float)
		v1 = np.asarray(data["param1_values"], dtype=float)
		v2 = np.asarray(data["param2_values"], dtype=float)

		p1_name = str(data["param1_name"])
		p2_name = str(data["param2_name"])

		X, Y = np.meshgrid(v1, v2, indexing="ij")
		ked_max = float(np.nanmax(ked)) if ked.size else 0.0
		if not np.isfinite(ked_max) or ked_max <= 0.0:
			ked_max = 1e-12

		fig = plt.figure(figsize=(14, 6))

		ax1 = fig.add_subplot(1, 2, 1, projection="3d")
		norm1, cmap1 = self._plot_colored_wireframe(
			ax=ax1,
			X=X,
			Y=Y,
			Z=ked,
			cmap_name="viridis",
			linewidth=0.85,
		)
		ax1.set_title("KED 3D Wireframe")
		ax1.set_xlabel(p1_name)
		ax1.set_ylabel(p2_name)
		ax1.set_zlabel("KED")
		ax1.set_xlim(float(np.min(v1)), float(np.max(v1)))
		ax1.set_ylim(float(np.min(v2)), float(np.max(v2)))
		ax1.set_zlim(np.nanmin(ked), ked_max)
		mappable1 = plt.cm.ScalarMappable(norm=norm1, cmap=cmap1)
		mappable1.set_array([])
		fig.colorbar(mappable1, ax=ax1, shrink=0.65, pad=0.08, label="KED")

		ax2 = fig.add_subplot(1, 2, 2, projection="3d")
		norm2, cmap2 = self._plot_colored_wireframe(
			ax=ax2,
			X=X,
			Y=Y,
			Z=keb,
			cmap_name="plasma",
			linewidth=0.85,
		)
		ax2.set_title("KEB 3D Wireframe")
		ax2.set_xlabel(p1_name)
		ax2.set_ylabel(p2_name)
		ax2.set_zlabel("KEB")
		ax2.set_xlim(float(np.min(v1)), float(np.max(v1)))
		ax2.set_ylim(float(np.min(v2)), float(np.max(v2)))
		ax2.set_zlim(0.0, 1.2)
		mappable2 = plt.cm.ScalarMappable(norm=norm2, cmap=cmap2)
		mappable2.set_array([])
		fig.colorbar(mappable2, ax=ax2, shrink=0.65, pad=0.08, label="KEB")

		plt.tight_layout()
		plt.show()
		if save_fig_path is not None:
			fig_path = Path(save_fig_path)
			fig_path.parent.mkdir(parents=True, exist_ok=True)
			fig.savefig(fig_path, dpi=300, bbox_inches="tight")
			print(f"[ked_keb] Saved KED/KEB figure: {fig_path}")

	def Bifurcation_diagram(
		self,
		x0,
		z0,
		lattice_index,
		param_name,
		param_range,
		steps=2000,
		discard=1000,
	):
		import matplotlib.pyplot as plt

		if not hasattr(self, param_name):
			raise ValueError(f"Unknown parameter: {param_name}")
		if not (0 <= int(lattice_index) < self.L):
			raise ValueError(f"lattice_index must be in [0, {self.L - 1}]")
		if not isinstance(steps, int) or steps <= 0:
			raise ValueError("steps must be a positive integer")
		if not isinstance(discard, int) or discard < 0:
			raise ValueError("discard must be a non-negative integer")

		x0 = np.asarray(x0, dtype=float).copy()
		z0 = float(z0)
		if x0.size != self.L:
			raise ValueError(f"x0 length must equal L={self.L}")

		param_values = np.asarray(param_range, dtype=float).reshape(-1)
		if param_values.size == 0:
			raise ValueError("param_range is empty")

		x_scatter = np.repeat(param_values, steps)
		y_scatter = np.empty(param_values.size * steps, dtype=float)
		pos = 0

		try:
			with Progress(
				TextColumn("[bold cyan]{task.description}"),
				BarColumn(),
				TaskProgressColumn(),
				TimeElapsedColumn(),
				TimeRemainingColumn(),
			) as progress:
				task = progress.add_task(
					f"Bifurcation scan: {param_name}",
					total=int(param_values.size),
				)

				for p in param_values:
					self._set_param_value(param_name, float(p))
					self._sync_index_rule()

					x = x0.copy()
					z = z0

					for _ in range(discard):
						x, z = self.step(x, z)

					for _ in range(steps):
						x, z = self.step(x, z)
						y_scatter[pos] = x[int(lattice_index)]
						pos += 1

					progress.update(task, advance=1)
		finally:
			self._reset_params()

		plt.figure(figsize=(10, 6))
		plt.scatter(x_scatter, y_scatter,marker='.', color="blue", s=5, alpha=1, edgecolors="none")
		plt.title(f"Bifurcation Diagram")
		plt.xlabel("v")
		plt.ylabel(f"State at Index {lattice_index}")
		plt.xlim(float(np.min(param_values)), float(np.max(param_values)))
		plt.ylim(0, 1)
		plt.grid(False)
		plt.tight_layout()
		plt.show()

		return x_scatter, y_scatter

	def vis_lattice_n(self,lattice_index):
	#展示lattice_index位置的状态随时间的变化
		N = 1000
		x0 = self.x0
		z0 = self.z0
		import matplotlib.pyplot as plt
		x0 = np.asarray(x0, dtype=float).copy()
		z0 = float(z0)
		if x0.size != self.L:
			raise ValueError(f"x0 length must equal L={self.L}")
		x_values = np.empty(N, dtype=float)
		z = z0
		x = x0.copy()
		for t in range(N):
			x, z = self.step(x, z)
			x_values[t] = x[int(lattice_index)]

		plt.figure(figsize=(10, 6))
		plt.plot(range(N), x_values, marker=".", markersize=2.0, alpha=0.7)
		plt.title(f"State Evolution at Lattice Index {lattice_index}")
		plt.xlabel("Time Step")
		plt.ylabel(f"State at Index {lattice_index}")
		plt.grid(True, alpha=0.3)
		plt.tight_layout()
		plt.show()
	def vis_lattice_state(self):
     #展示迭代N步后整个格点的状态分布
		N = 100
		x0 = self.x0
		z0 = self.z0
		import matplotlib.pyplot as plt	
		x0 = np.asarray(x0, dtype=float).copy()
		z0 = float(z0)
		if x0.size != self.L:
			raise ValueError(f"x0 length must equal L={self.L}")
		x_values = np.empty((N, self.L), dtype=float)
		z = z0
		x = x0.copy()
		for t in range(N):
			x, z = self.step(x, z)
			x_values[t, :] = x
		#散点图绘制x_values[-1, :]的分布情况
		plt.figure(figsize=(10, 6))
		plt.scatter(range(self.L), x_values[-1, :], s=20, alpha=0.7)
		plt.title(f"State Distribution at Final Time Step (N={N})")
		plt.xlabel("Lattice Index")	
		plt.ylabel("State Value")
		plt.grid(True, alpha=0.3)
		plt.tight_layout()
		plt.show()
  
	def generate_rdseq(self, N: int) -> np.ndarray:
		"""Generate x-only random numbers via ``step`` iteration.

		Returns an array with shape ``(L, N)``, where each column stores the
		x-state produced by one iteration.
		"""
		if not isinstance(N, int) or N <= 0:
			raise ValueError("N must be a positive integer")

		x = np.asarray(self.x0, dtype=float).copy()
		z = float(self.z0)
		if x.size != self.L:
			raise ValueError(f"x0 length must equal L={self.L}")

		x_values = np.empty((self.L, N), dtype=float)
		for t in range(N):
			x, z = self.step(x, z)
			x_values[:, t] = x

		return x_values

	def generate_rdseq_fast(self, N: int) -> np.ndarray:
		"""Generate the (N, L) random matrix with preallocation and Numba."""
		if not isinstance(N, int) or N <= 0:
			raise ValueError("N must be a positive integer")

		x0 = np.asarray(self.x0, dtype=float).copy()
		if x0.size != self.L:
			raise ValueError(f"x0 length must equal L={self.L}")

		if _generate_rdseq_numba is not None:
			return _generate_rdseq_numba(
				x0=x0,
				p_idx=np.asarray(self._p_idx, dtype=np.int64),
				q_idx=np.asarray(self._q_idx, dtype=np.int64),
				mu=float(self.mu),
				a=float(self.a),
				alpha=float(self.alpha),
				beta=float(self.beta),
				is_mod=bool(self.is_mod),
				N=N,
			)

		fx_left = np.empty(self.L, dtype=float)
		fx_right = np.empty(self.L, dtype=float)
		sum_buffer = np.empty(self.L, dtype=float)
		fx_p = np.empty(self.L, dtype=float)
		fx_q = np.empty(self.L, dtype=float)
		x_values = np.empty((N, self.L), dtype=float)
		x = x0.copy()

		for t in range(N):
			fx = np.asarray(self.f(x), dtype=float)
			fx_left[0] = fx[-1]
			fx_left[1:] = fx[:-1]
			fx_right[-1] = fx[0]
			fx_right[:-1] = fx[1:]
			np.add(fx_left, fx, out=sum_buffer)
			np.add(sum_buffer, fx_right, out=sum_buffer)
			np.multiply(sum_buffer, 2.0 * np.pi, out=sum_buffer)
			np.cos(sum_buffer, out=sum_buffer)
			np.multiply(sum_buffer, -1.0, out=sum_buffer)
			np.add(sum_buffer, 10.0 ** self.alpha, out=sum_buffer)
			np.take(fx, self._p_idx, out=fx_p)
			np.take(fx, self._q_idx, out=fx_q)
			np.multiply(fx_p, fx_p, out=fx_p)
			np.multiply(fx_q, fx_q, out=fx_q)
			np.add(fx_p, fx_q, out=fx_p)
			np.sqrt(fx_p, out=fx_p)
			np.multiply(fx_p, 10.0 ** self.beta, out=fx_p)
			np.add(sum_buffer, fx_p, out=x)
			if self.is_mod:
				np.mod(x, 1.0, out=x)
			x_values[t, :] = x

		return x_values
if __name__ == "__main__":
	L = 100
	params = {
		"mu": 5,
		"lam": 5,
		"a": 20,
		"b": 20,
		"alpha": 5,
		"beta": 5,
		"xi": 1,
		"eta": 1,
	}
	seed = 2026
	np.random.seed(seed)
	x0 = np.random.rand(L)
	z0 = np.random.rand()
	cml = SalomoncouplingCML(L=L, params=params, initstate={"x0": x0, "z0": z0})

	# N = 28
	# cml.generate_rdseq_fast(1)
	# st = time.time()
	# cml.generate_rdseq_fast(N)
	# et = time.time()
	# print(f"Time taken for {N} steps: {et - st:.6f} seconds")


	# cml.vis_lattice_n(lattice_index=25)
	# cml.vis_lattice_state()

	# 示例 1: 分叉图
	# cml.Bifurcation_diagram(
	# 	x0=x0,
	# 	z0=z0,
	# 	lattice_index=1,
	# 	param_name="alpha",
	# 	param_range=np.linspace(0, 5, 1000),
	# 	steps=500,
	# 	discard=50,
	# )

	# 示例 2: 最小 Lyapunov 双参数扫描 + KED/KEB 可视化
# 	demo_scan_path = "mywork/output/salomonV2_lyapunov_scan.npz"
# 	cml.lyap_scan(
# 		param1="mu",
# 		values1=np.linspace(0.1, 10, 50),
# 		param2="a",
# 		values2=np.linspace(0.1, 10, 50),
# 		x0=x0,
# 		z0=z0,
# 		n=250,
# 		discard=50,
# 		epsilon=1e-12,
# 		save_path="mywork/output/salomonV2_lyapunov_scan_mu&v.npz",
# 		timestamp_on_exists=True,
# 	)
# 	plot_path = cml.last_scan_path if cml.last_scan_path is not None else demo_scan_path
# 	cml.plot_ked_keb(plot_path, save_fig_path="mywork/output/salomonV2_lyapunov_scan_mu&v.png")
#  
# 	cml.lyap_scan(
# 		param1="alpha",
# 		values1=np.linspace(0.1, 10, 50),
# 		param2="beta",
# 		values2=np.linspace(0.1, 10, 50),
# 		x0=x0,
# 		z0=z0,
# 		n=250,
# 		discard=50,
# 		epsilon=1e-12,
# 		save_path="mywork/output/salomonV2_lyapunov_scan_alpha_beta.npz",
# 		timestamp_on_exists=True,
# 	)
# 	plot_path = cml.last_scan_path if cml.last_scan_path is not None else None
# 	cml.plot_ked_keb(plot_path, save_fig_path="mywork/output/salomonV2_lyapunov_scan_alpha_beta.png")
#  
	# cml.lyap_scan(
	# 	param1="mu",
	# 	values1=np.linspace(0.1, 10, 50),
	# 	param2="alpha",
	# 	values2=np.linspace(0.1, 10, 50),
	# 	x0=x0,
	# 	z0=z0,
	# 	n=250,
	# 	discard=50,
	# 	epsilon=1e-12,
	# 	save_path="mywork/output/salomonV2_lyapunov_scan_mu_alpha.npz",
	# 	timestamp_on_exists=True,
	# )
	# plot_path = cml.last_scan_path if cml.last_scan_path is not None else None
	# cml.plot_ked_keb(plot_path, save_fig_path="mywork/output/salomonV2_lyapunov_scan_mu_alpha.png")
 


	# 示例 3: NIST 800-22 测试
	# cml.generate_random_bits_file(n_bits=100_000_000, save_path="mywork/output/salomonV2_overall_random.bin", x0=x0, z0=z0, warmup=2000, scale_factor=10**10, bitorder="little")
	# cml.generate_random_bits_file(n_bits=100_000_000, save_path="mywork/output/salomonV2_lattice95_random.bin", x0=x0, z0=z0, warmup=2000, scale_factor=10**10, bitorder="little",mode = "lattice")

	# 示例 4: 信息熵 IE(i, p)（扫描任意单参数）+ 线框图
	# cml.IE(
	# 	param_name="lam",
	# 	param_range=np.linspace(0.0, 1.0, 41),
	# 	x0=x0,
	# 	z0=z0,
	# 	n=1000,
	# 	discard=200,
	# 	n_states=10,
	# 	save_path="mywork/output/salomon_ie_scan_lam.npz",
	# 	timestamp_on_exists=True,
	# 	plot=True,
	# 	save_fig_path="mywork/output/salomon_ie_wireframe_lam.png",
	# )

	# 示例 5: 平均信息熵 Hd(p1, p2) 双参数扫描 + 线框图
	# cml.avg_IE(
	# 	param1="mu",
	# 	values1=np.linspace(0, 5, 50),
	# 	param2="a",
	# 	values2=np.linspace(0, 5, 50),
	# 	n=1000,
	# 	discard=200,
	# 	n_states=10,
	# 	save_path="mywork/output/salomon_avg_ie_mu_a.npz",
	# 	plot=True,
	# 	save_fig_path="mywork/output/salomon_avg_ie_mu_a.png",
	# )

	# # 示例 6: 信息熵IE
	# cml.IE(
	# 	param_name="a",
	# 	param_range=np.linspace(0, 5, 50),
	# 	n=1000,
	# 	discard=200,
	# 	n_states=10,
	# 	save_path="mywork/output/salomon_ie_a.npz",
	# 	plot=True,
	# 	save_fig_path="mywork/output/salomon_ie_a.png",
	# )
 
	# 示例 7: 平均互信息 Ld(p1, p2) 双参数扫描 + 线框图
	# cml.AMI_scan(
	# 	param1="mu",
	# 	values1=np.linspace(1, 10, 25),
	# 	param2="a",
	# 	values2=np.linspace(1, 10, 25),
	# 	x0=x0,
	# 	z0=z0,
	# 	n=150,
	# 	discard=50,
	# 	n_states=10,
	# 	save_path="mywork/output/salomon_avg_ami_mu_a.npz",
	# 	plot=True,
	# 	save_fig_path="mywork/output/salomon_avg_ami_mu_a.png",
	# )
