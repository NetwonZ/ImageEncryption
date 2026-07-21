# 忆阻混沌神经元图像加密复现

本目录按论文 *An image encryption algorithm based on memristive chaotic neuron in medical internet of things environments* 的 Algorithm 1-3 实现灰度图像的完整流程：

1. 公式(1)的忆阻混沌神经元和 Euler 离散；
2. 公式(2)的梯度驱动递归四分裂、块内 Zigzag/旋转和宏块随机排列；
3. 公式(3)动态 S-box、局部 XOR 及扩散模块；
4. 严格逆扩散和逆置乱，保证 `decrypt(encrypt(image)) == image`。

## 运行

```powershell
python main.py
python main.py --input path\to\gray.png --output-dir output\medical
python -m unittest discover -s tests -v
```

默认会生成并输出 `01_original.png`、`02_scrambled.png`、`03_diffusion_local_xor.png`、`04_ciphertext.png` 和 `05_decrypted.png`。

## 复现边界

论文公开了神经元参数、初值和所有核心公式，但正文的 Algorithm 1 没有给出 Euler 步长 `Δτ` 与预迭代次数，Algorithm 2 也未给出 `Bmax/Bmin` 的具体数值。因此代码将它们设为可配置参数：默认 `Δτ=0.01`、预迭代 `1000`，两轮分块尺寸按图像边长生成可重复的 `32/8`、`16/4`（小图按比例退化）。若掌握论文实验代码或补充材料，可直接传入对应参数以对齐其具体数值。

`ImageCryptosystem` 当前按论文的正方形灰度图像设定工作；论文实验图像也是 512×512 或 1024×1024。所有置乱块的源/目标坐标、变换类型和宏块 permutation 都保存在 `ScrambleKey` 中，供逆置乱使用。

## 差分攻击模式

论文公开的公式(5)是单向线性累计：单像素差异只向螺旋路径后方传播，且差分幅度近似不变，因此它本身无法稳定达到论文表10的 UACI ≈ 33.46%。默认的 `diffusion_mode="hardened"` 使用双向、S-box 非线性反馈扩散，能保持严格可逆并使单点变化扩散到整张图像。若需要逐式运行公开公式，请显式传入 `diffusion_mode="paper"`；该模式适合公式核对，但不应期待表10的差分指标。
