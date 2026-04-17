from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import sympy as sp
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn


class SalomoncouplingCML:
	"""Salomon coupling CML with non-adjacent p/q indices.

	Core update rule:
		x_{n+1}(i) = 1 - cos(2*pi*(f(x_{i-1}) + f(x_i) + f(x_{i+1})))
					 + 0.1*sqrt(f(x_p)^2 + f(x_q)^2)

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

		required = {"mu", "lam", "a", "b", "xi", "eta"}
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
			"xi": self.xi,
			"eta": self.eta,
		}
		self.last_scan_path: str | None = None

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

		fx = np.asarray(self.f(x), dtype=float)
		fx_left = np.roll(fx, 1)
		fx_right = np.roll(fx, -1)
		fx_p = fx[self._p_idx]
		fx_q = fx[self._q_idx]

		x_next = 1.0 - np.cos(2.0 * np.pi * (fx_left + fx + fx_right))
		x_next += 0.1 * np.sqrt(fx_p * fx_p + fx_q * fx_q)

		if self.is_mod:
			x_next = np.mod(x_next, 1.0)

		z_next = np.mod(self.g(float(z)), 1.0)
		return x_next, float(z_next)

	def iterate_median(
		self,
		x0: np.ndarray,
		z0: float,
		n: int,
		return_states: bool = False,
	) -> float | tuple[np.ndarray, float]:
		"""Iterate N times and compute median over all N*L generated values.

		Args:
			x0: Initial x state, length must equal L.
			z0: Initial z state.
			n: Number of iterations.
			return_states: If True, also return the (n, L) stored states.

		Returns:
			Median value of flattened (n, L) states,
			or (states, median) when return_states=True.
		"""
		if not isinstance(n, int) or n <= 0:
			raise ValueError("n must be a positive integer")

		x = np.asarray(x0, dtype=float).copy()
		z = float(z0)
		if x.size != self.L:
			raise ValueError(f"x0 length must equal L={self.L}")

		states = np.empty((n, self.L), dtype=float)
		for i in range(n):
			x, z = self.step(x, z)
			states[i, :] = x

		median_value = float(np.median(states))
		if return_states:
			return states, median_value
		return median_value

	#sym:generate_random_bits_file
	def generate_random_bits_file(
		self,
		n_bits: int,
		save_path: str = "mywork/output/salomon_random.bin",
		x0: np.ndarray | None = None,
		z0: float | None = None,
		warmup: int = 200,
		threshold: float = 0.5,
		bitorder: str = "little",
	) -> Path:
		"""Generate a binary file from Salomon CML output and print 0/1 counts."""
		if not isinstance(n_bits, int) or n_bits <= 0:
			raise ValueError("n_bits must be a positive integer")
		if not isinstance(warmup, int) or warmup < 0:
			raise ValueError("warmup must be a non-negative integer")

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

		steps = (n_bits + self.L - 1) // self.L
		bits = np.empty(steps * self.L, dtype=np.uint8)
		pos = 0

		with Progress(
			TextColumn("[bold cyan]{task.description}"),
			BarColumn(),
			TaskProgressColumn(),
			TimeElapsedColumn(),
			TimeRemainingColumn(),
		) as progress:
			task = progress.add_task("Generating Salomon random bits", total=steps)
			for _ in range(steps):
				x, z = self.step(x, z)
				row_bits = (x >= threshold).astype(np.uint8)
				bits[pos:pos + self.L] = row_bits
				pos += self.L
				progress.update(task, advance=1)

		bits = bits[:n_bits]
		one_count = int(np.sum(bits))
		zero_count = int(bits.size - one_count)

		pad = (-n_bits) % 8
		if pad:
			bits = np.pad(bits, (0, pad), mode="constant", constant_values=0)

		packed = np.packbits(bits, bitorder=bitorder)
		path.write_bytes(packed.tobytes())

		print(f"[prng] Generated {n_bits} bits -> {packed.size} bytes")
		print(f"[prng] Saved to: {path}")
		print(f"[prng] Ones count: {one_count}")
		print(f"[prng] Zeros count: {zero_count}")
		return path

	def _jacobian_x(self, x: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
		"""Jacobian of Salomon x-update w.r.t. x.

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
			J[i, p_idx] += 0.1 * (fx_p[i] * fp_p[i]) * inv_R[i]
			J[i, q_idx] += 0.1 * (fx_q[i] * fp_q[i]) * inv_R[i]

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
				task = progress.add_task(f"Scanning {param1} x {param2}", total=total)

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

	#sym:ked_keb
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

	#sym:plot_ked_keb
	def plot_ked_keb(self, data_path: str) -> None:
		"""Load saved scan data and plot KED/KEB 3D surfaces."""
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
		surf1 = ax1.plot_surface(X, Y, ked, cmap="viridis", linewidth=0, antialiased=True)
		ax1.set_title("KED 3D Surface")
		ax1.set_xlabel(p1_name)
		ax1.set_ylabel(p2_name)
		ax1.set_zlabel("KED")
		ax1.set_zlim(0.0, ked_max)
		fig.colorbar(surf1, ax=ax1, shrink=0.65, pad=0.08)

		ax2 = fig.add_subplot(1, 2, 2, projection="3d")
		surf2 = ax2.plot_surface(X, Y, keb, cmap="plasma", linewidth=0, antialiased=True)
		ax2.set_title("KEB 3D Surface")
		ax2.set_xlabel(p1_name)
		ax2.set_ylabel(p2_name)
		ax2.set_zlabel("KEB")
		ax2.set_zlim(0.0, 1.2)
		fig.colorbar(surf2, ax=ax2, shrink=0.65, pad=0.08)

		plt.tight_layout()
		plt.show()

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
		plt.plot(
			x_scatter,
			y_scatter,
			marker=".",
			linestyle="",
			markersize=1.0,
			alpha=0.35,
		)
		plt.title(f"Bifurcation Diagram for {param_name} at Lattice Index {lattice_index}")
		plt.xlabel(param_name)
		plt.ylabel(f"State at Index {lattice_index}")
		plt.grid(True, alpha=0.3)
		plt.tight_layout()
		plt.show()

		return x_scatter, y_scatter

if __name__ == "__main__":
	L = 100
	params = {
		"mu": 5,
		"lam": 5,
		"a": 100,
		"b": 200,
		"xi": 1,
		"eta": 1,
	}
	x0 = np.random.rand(L)
	z0 = np.random.rand()
	cml = SalomoncouplingCML(L=L, params=params, initstate={"x0": x0, "z0": z0})

	# 示例 1: 分叉图
	# cml.Bifurcation_diagram(
	# 	x0=x0,
	# 	z0=z0,
	# 	lattice_index=25,
	# 	param_name="mu",
	# 	param_range=np.linspace(0, 5, 400),
	# 	steps=1000,
	# 	discard=200,
	# )

	# 示例 2: 最小 Lyapunov 双参数扫描 + KED/KEB 可视化
	# demo_scan_path = "mywork/output/salomon_lyapunov_scan_demo.npz"
	# cml.lyap_scan(
	# 	param1="mu",
	# 	values1=np.linspace(0, 5.0, 25),
	# 	param2="lam",
	# 	values2=np.linspace(0, 5.0, 25),
	# 	x0=x0,
	# 	z0=z0,
	# 	n=100,
	# 	discard=40,
	# 	epsilon=1e-12,
	# 	save_path=demo_scan_path,
	# 	timestamp_on_exists=True,
	# )
	# plot_path = cml.last_scan_path if cml.last_scan_path is not None else demo_scan_path
	# cml.plot_ked_keb(plot_path)
 
	# 示例 3: NIST 800-22 测试
	# median = cml.iterate_median(x0=x0, z0=z0, n=1_000_000, return_states=False)
	# print(f"Median of 1M values: {median:.6f}")
	median = 0.372636
	# cml.generate_random_bits_file(n_bits=100_000_000, save_path="mywork/output/salomon_random.bin", x0=x0, z0=z0, warmup=2000, threshold=median)
	def inspect_binary_file(filename, bytes_to_read=10000):
		# 读取前 10000 个字节
		with open(filename, 'rb') as f:
			raw_bytes = f.read(bytes_to_read)
		
		# 转换为 0 和 1 的数组
		byte_array = np.frombuffer(raw_bytes, dtype=np.uint8)
		bit_array = np.unpackbits(byte_array)
		
		# 统计 0 和 1 的比例
		ones = np.sum(bit_array)
		zeros = len(bit_array) - ones
		
		print(f"检查了前 {len(bit_array)} 个比特:")
		print(f"0 的数量: {zeros} ({zeros/len(bit_array)*100:.2f}%)")
		print(f"1 的数量: {ones} ({ones/len(bit_array)*100:.2f}%)")
		
		# 打印前 100 个比特直观感受一下
		print("前 100 个比特:", bit_array[:100])

	inspect_binary_file("mywork/output/cml_random.bin", bytes_to_read=10000000)