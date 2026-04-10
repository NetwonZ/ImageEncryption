from __future__ import annotations

from pathlib import Path

import numpy as np
import sympy as sp
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn


class CMLSystem:
	"""Coupled map lattice system based on the provided research formula."""

	def __init__(self, L: int, params: dict[str, float], initstate: dict[str, np.ndarray | float]) -> None:
		self.L = int(L)
		self.mu = float(params["mu"])
		self.lam = float(params["lam"])
		self.a = float(params["a"])
		self.b = float(params["b"])
		self.xi = int(params["xi"])
		self.eta = int(params["eta"])
		self.x0 = initstate["x0"]
		self.z0 = initstate["z0"]
		self.original_params = params.copy()

		# Follow the note in your formula description.
		if self.xi == 0:
			self.eta = self.L
		if self.eta == 0:
			self.xi = self.L

		self._build_symbolic_functions()
		self._build_neighbor_indices()

	def _build_symbolic_functions(self) -> None:
		x = sp.Symbol("x", real=True)
		mu = sp.Symbol("mu", real=True)
		lam = sp.Symbol("lam", real=True)
		a = sp.Symbol("a", real=True)
		b = sp.Symbol("b", real=True)

		f_expr = sp.Abs(sp.sin((5 + 3 * mu) * (1 - (a * x * sp.sin(15 * sp.pi * x * (1 - x))))))
		g_expr = sp.Abs(sp.sin((5 + 3 * lam) * (1 - (b * x * sp.sin(15 * sp.pi * x * (1 - x))))))

		f_diff_expr = sp.diff(f_expr, x)

		self._f = sp.lambdify((x, mu, a), f_expr, modules="numpy")
		self._g = sp.lambdify((x, lam, b), g_expr, modules="numpy")
		self._f_diff = sp.lambdify((x, mu, a), f_diff_expr, modules="numpy")

	def _build_neighbor_indices(self) -> None:
		# 非相邻耦合
		# i = np.arange(self.L, dtype=int)
		# p = ((1 + self.xi) * i) % self.L
		# q = ((self.eta + self.xi * self.eta + 1) * i) % self.L
		# self._p_idx = p.astype(int)
		# self._q_idx = q.astype(int)
  
		# 相邻耦合
		idx = np.arange(self.L, dtype=int)
		self._p_idx = np.roll(idx,1)
		self._q_idx = np.roll(idx,-1)

	def _reset_params(self) -> None:
		for key, value in self.original_params.items():
			setattr(self, key, float(value))
		self._sync_index_rule()

	def f(self, x: np.ndarray | float) -> np.ndarray | float:
		return self._f(x, self.mu, self.a)

	def g(self, z: float) -> float:
		return float(self._g(z, self.lam, self.b))

	def _f_prime(self, x: np.ndarray) -> np.ndarray:
		return np.asarray(self._f_diff(x, self.mu, self.a), dtype=float)

	def step(self, x: np.ndarray, z: float) -> tuple[np.ndarray, float]:
		x = np.asarray(x, dtype=float)
		g_z = self.g(z)
		fx = np.asarray(self.f(x), dtype=float)

		x_next = (1.0 - g_z) * fx + (g_z / 2.0) * (fx[self._p_idx] + fx[self._q_idx])
		x_next = np.mod(x_next, 1.0)

		z_next = np.mod(g_z, 1.0)
		return x_next, float(z_next)

	def iterate(self, x0: np.ndarray, z0: float, steps: int, discard: int = 0) -> tuple[np.ndarray, float]:
		x = np.asarray(x0, dtype=float).copy()
		z = float(z0)

		for _ in range(discard + steps):
			x, z = self.step(x, z)

		return x, z

	def _jacobian_x(self, x: np.ndarray, z: float) -> np.ndarray:
		g_z = self.g(z)
		fp = self._f_prime(x)

		J = np.zeros((self.L, self.L), dtype=float)
		coef_center = 1.0 - g_z
		coef_neighbor = g_z / 2.0

        # method1: loop
		for i in range(self.L):
			J[i, i] += coef_center * fp[i]
			p = self._p_idx[i]
			q = self._q_idx[i]
			J[i, p] += coef_neighbor * fp[p]
			J[i, q] += coef_neighbor * fp[q]
		# method2: vectorized (but may be less clear)
# 		idx = np.arange(self.L, dtype=int)
# 		J = np.zeros((self.L, self.L), dtype=float)
# 
# 		np.add.at(J, (idx, idx), coef_center * fp)
# 		np.add.at(J, (idx, self._p_idx), coef_neighbor * fp[self._p_idx])
# 		np.add.at(J, (idx, self._q_idx), coef_neighbor * fp[self._q_idx])
   

		return J

	def lyapunov_spectrum(
		self,
		x0: np.ndarray,
		z0: float,
		n: int,
		discard: int = 100,
		epsilon: float = 1e-12,
	) -> np.ndarray:
		"""Return full Lyapunov spectrum (length L) in descending order."""
		x = np.asarray(x0, dtype=float).copy()
		z = float(z0)

		Q = np.eye(self.L, dtype=float)
		log_sum = np.zeros(self.L, dtype=float)

		total_steps = discard + n
		for step_idx in range(total_steps):
			J = self._jacobian_x(x, z)
			Z = J @ Q
			Q, R = np.linalg.qr(Z)

			if step_idx >= discard:
				d = np.abs(np.diag(R))
				log_sum += np.log(d + epsilon)

			x, z = self.step(x, z)

		spectrum = log_sum / float(n)
		return np.sort(spectrum)[::-1]

	@staticmethod
	def ked_keb(spectrum: np.ndarray) -> tuple[float, float]:
		"""Compute KED and KEB from one Lyapunov spectrum vector.

		Args:
			spectrum: 1D array, e.g. spectra[i, j, :] with length L.

		Returns:
			(ked, keb)
		 """
		lam = np.asarray(spectrum, dtype=float).reshape(-1)
		positive = lam[lam > 0.0]
		N = lam.size

		ked = float(np.sum(positive) / N)
		keb = float(positive.size / N)
		return ked, keb

	def _set_param_value(self, name: str, value: float) -> None:
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
		save_path: str = "mywork/output/cml_lyapunov_scan.npz",
	) -> np.ndarray:
		"""Scan two parameters and save all Lyapunov spectra for every combination.

		Returns:
			spectra: shape = (len(values1), len(values2), L)
		 """
		path = Path(save_path)
		path.parent.mkdir(parents=True, exist_ok=True)

		if path.exists():
			print(f"[scan] File already exists: {path}")
			while True:
				choice = input("Choose action: [s]kip / [d]elete and rescan: ").strip().lower()
				if choice in ("s", "skip"):
					print("[scan] Skip current scan. Loading existing spectra from file.")
					with np.load(path) as existing:
						if "spectra" not in existing:
							raise KeyError(f"'spectra' not found in existing file: {path}")
						return np.asarray(existing["spectra"], dtype=float)
				if choice in ("d", "delete"):
					path.unlink()
					print("[scan] Existing file deleted. Start new scan.")
					break
				print("Invalid input. Please type 's' or 'd'.")

		v1 = np.asarray(values1, dtype=float)
		v2 = np.asarray(values2, dtype=float)
		spectra = np.empty((v1.size, v2.size, self.L), dtype=float)

		original = {
			"mu": self.mu,
			"lam": self.lam,
			"a": self.a,
			"b": self.b,
			"xi": self.xi,
			"eta": self.eta,
		}
		ked = np.empty((v1.size, v2.size), dtype=float)
		keb = np.empty((v1.size, v2.size), dtype=float)

		total = int(v1.size * v2.size)
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
						x0=x0,
						z0=z0,
						n=n,
						discard=discard,
						epsilon=epsilon,
					)
					ked_, keb_ = self.ked_keb(spectra[i, j, :])
					ked[i, j] = ked_
					keb[i, j] = keb_
					progress.update(task, advance=1)


		for key, value in original.items():
			setattr(self, key, value)
		self._sync_index_rule()

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
		return spectra

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

	def generate_random_bits_file(
    self,
    n_bits: int,
    save_path: str = "mywork/output/cml_random.bin",
    x0: np.ndarray | None = None,
    z0: float | None = None,
    warmup: int = 200,
    threshold: float = 0.5,
    channel_index: int = 0,
    bitorder: str = "little",
    seed: int | None = None,
) -> Path:


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

		print(f"当前的参数设置：{self.params}")
		rng = np.random.default_rng(seed)

		if x0 is None or z0 is None:
			raise ValueError("Both x0 and z0 must be provided as initial conditions for the CML system.")
		# Burn-in to reduce initial transient effect.
		x = np.asarray(x0, dtype=float).copy()
		z = float(z0)
		for _ in range(max(0, int(warmup))):
			x, z = self.step(x, z)

		idx = int(channel_index) % self.L
		bits = np.empty(n_bits, dtype=np.uint8)

		for i in range(n_bits):
			x, z = self.step(x, z)
			bits[i] = 1 if x[idx] >= threshold else 0

		# Pack bits to bytes and write binary file.
		pad = (-n_bits) % 8
		if pad:
			bits = np.pad(bits, (0, pad), mode="constant", constant_values=0)

		packed = np.packbits(bits, bitorder=bitorder)
		path.write_bytes(packed.tobytes())

		print(f"[rng] Generated {n_bits} bits -> {packed.size} bytes")
		print(f"[rng] Saved to: {path}")
		return path

	def vis_states(self,x0,z0,x1,z1,lat_index,steps,view = True):
		x = np.asarray(x0, dtype=float).copy()
		z = float(z0)
		x_ = np.asarray(x1, dtype=float).copy()
		z_ = float(z1)
		states = np.empty((steps, self.L), dtype=float)
		states_ = np.empty((steps, self.L), dtype=float)
		for i in range(steps):
			states[i, :] = x
			states_[i, :] = x_
			x, z = self.step(x, z)
			x_, z_ = self.step(x_, z_)
		import matplotlib.pyplot as plt
		#将两个不同初始状态下的	lat_index位置绘制在同一张图上，展示它们的演化轨迹
		if view:
			print("drawing states...")
			plt.figure(figsize=(10, 6))
			plt.plot(states[:, lat_index], label="State 1", alpha=0.7)
			plt.plot(states_[:, lat_index], label="State 2", alpha=0.7)
			#在额外绘制一条线，表示两个状态的差值的绝对值，展示它们的分叉情况
			diff = np.abs(states[:, lat_index] - states_[:, lat_index])
			plt.plot(diff, label="Absolute Difference", color="red", linestyle="--", alpha=0.7)
			plt.title(f"Evolution of Lattice Index {lat_index}")
			plt.xlabel("Time Step")
			plt.ylabel("State Value")	
			plt.legend()
			plt.grid()
			plt.tight_layout()
			plt.show()
		return states, states_

  


if __name__ == "__main__":
	params = {
		"mu": 5,
		"lam": 5,
		"a": 3.0,
		"b": 5.0,
		"xi": 1,
		"eta": 1,
	}
 
	L = 50
	x0 = np.random.rand(L)
	z0 = 0.37
 
	sys = CMLSystem(L=L, params=params, initstate={"x0": np.random.rand(L), "z0": 0.37})


 
	# sys.lyap_scan(
	# 	param1="mu",
	# 	values1=np.linspace(0, 5, 10),
	# 	param2="lam",
	# 	values2=np.linspace(0, 5, 10),
	# 	x0=x0,
	# 	z0=z0,
	# 	n=200,
	# 	discard=100,
	# 	save_path="mywork/output/cml_lyapunov_scan.npz",
	# )
	# sys.plot_ked_keb("mywork/output/cml_lyapunov_scan.npz")
 
 	#将x0d的第一位加上0.001，得到x1d，其他参数保持不变，观察它们的演化轨迹是否相似
	x0_ = x0.copy()
	x0_[11] += 0.001
	# sys.vis_states(
	# 	x0=x0,
	# 	z0=0.37,
	# 	x1=x0_,
	# 	z1=0.37,
	# 	lat_index=40,
	# 	steps=200,
	# )
 
	# s,_ = sys.vis_states(x0=x0,z0=0.37,x1=x0_,z1=0.37,lat_index=40,steps=10000,view=False)
	# #将s保存成csv在本地用于数据分析
	# np.savetxt("mywork/output/cml_states.csv", s, delimiter=",")
 
	sys.generate_random_bits_file(
		n_bits=1000,
		save_path="mywork/output/cml_random.bin",
		x0=x0,
		z0=z0,	
		warmup=200,
		threshold=0.6375493986616645,
		channel_index=0,
		bitorder="little",	
		seed=42,
	)