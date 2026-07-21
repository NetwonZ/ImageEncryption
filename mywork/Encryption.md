我已经完整梳理了当前实现。[Encryption.py](/C:/ImageEncryption/mywork/Encryption.py:1) 目前是空文件，本轮没有迁移或修改 Python 代码，只生成了用于理解现有流程的说明图：

- [加密流程图 SVG](/C:/ImageEncryption/mywork/figures/scramble_diffusion_flow.svg)
- [Mermaid 源文件](/C:/ImageEncryption/mywork/figures/scramble_diffusion_flow.mmd)

当前加密入口是 [encrypt_image()](/C:/ImageEncryption/mywork/ScrambleDiffusion.py:1587)。

## 一、整体数据流

当前流程可以概括为：

```text
输入图像
  ↓
RGB uint8 标准化
  ├── 生成混沌随机矩阵 RdMatrix
  ├── 根据图像梯度进行自适应分块
  └── 分解为 24 个位平面
          ↓
按块选择 DNA 编码规则
          ↓
24 位平面 → 12 个 DNA 平面
          ↓
块内三维置乱
          ↓
同尺寸块之间交换
          ↓
块级 DNA 密钥混合
          ↓
全局置乱与 DNA XOR 混合
          ↓
12 个 DNA 平面 → 24 个加密位平面
          ↓
重新组合为 RGB 密文图像
```

主要的数据形状变化为：

| 阶段 | 数据形状 |
|---|---|
| 原始图像 | `(H, W, 3)` |
| 位平面 | `(24, H, W)` |
| DNA 编码矩阵 | `(12, H, W)` |
| 加密位平面 | `(24, H, W)` |
| 最终密文图像 | `(H, W, 3)` |

## 二、输入图像标准化

`_load_rgb_image_array()` 接受：

- 图像路径
- PIL 图像
- NumPy 数组
- 灰度图像

最后统一转换为连续存储的 RGB `uint8` 数组。

其中：

- 灰度图会复制成三个通道；
- 多于三个通道时仅保留前三个；
- `[0,1]` 范围的浮点图像会乘以 255；
- 其他数据会裁剪到 `[0,255]`。

因此后续算法始终面对 `(H,W,3)` 的 8 位 RGB 图像。

## 三、混沌随机矩阵生成

设：

```text
L = H × W
```

代码使用 `seed` 初始化 NumPy 随机数生成器，产生：

```python
x0 = seed_rng.random(L)
z0 = seed_rng.random()
```

然后构造 [SalomoncouplingCML](/C:/ImageEncryption/mywork/SalomonCouplingCML.py:103)，调用：

```python
rd_matrix = cml.generate_rdseq_fast(28)
```

得到：

```text
RdMatrix.shape = (28, H×W)
```

当前默认参数为：

```python
mu = 5
v = 5
alpha = 5
beta = 5
xi = 1
eta = 1
```

混沌映射大致为：

```text
f(x) = |sin((5+3μ) · (1-vx·sin(15πx(1-x))))|

xₙ₊₁(i)
 = 10^α
 - cos(2π(f(xᵢ₋₁)+f(xᵢ)+f(xᵢ₊₁)))
 + 10^β·sqrt(f(xₚ)²+f(x_q)²)
```

最后对结果模 1。

实际使用情况是：

- `RdMatrix` 前 12 行用于块内 DNA 三维置乱；
- 最后一行用于生成每个块的 DNA 编码规则；
- 中间若干行在当前加密主流程中没有使用；
- `z0` 虽然被传入，但当前 `SalomonCouplingCML` 实现没有使用它。

因此，现在生成 28 行随机矩阵存在一定冗余。

## 四、自适应分块

`adaptive_partition()` 根据图像梯度把图像划分为不同尺寸的矩形块。

支持三种梯度：

- `gray`：先转灰度，再计算梯度；
- `lab`：在 CIE Lab 空间计算三通道综合梯度；
- `di_zenzo`：使用 Di Zenzo 彩色向量梯度。

默认使用灰度梯度。

若没有传入阈值，则：

```text
threshold = 1.5 × 全图平均梯度
```

分块方式如下：

1. 先把图像划分成不超过 `b_max × b_max` 的初始块。
2. 计算每个块的平均梯度。
3. 若平均梯度高于阈值，则把块四等分。
4. 对子块递归执行同样操作。
5. 当块的任一边不大于 `b_min`，或者平均梯度低于阈值时停止。

所以：

- 平滑区域通常保留大块；
- 纹理和边缘区域通常得到较小块；
- 边缘处可能出现非正方形块。

每个块由 `ImageBlock` 保存：

```text
row, col, height, width, mean_gradient
```

这里有一个重要性质：分块结果依赖明文图像。将来实现解密时，不能指望从密文重新计算出相同分块，必须保存分块信息或设计其他可重现方式。

## 五、RGB 图像分解为 24 个位平面

`image_to_bitplanes()` 按以下顺序提取位平面：

```text
0～7   ：R7, R6, R5, R4, R3, R2, R1, R0
8～15  ：G7, G6, G5, G4, G3, G2, G1, G0
16～23 ：B7, B6, B5, B4, B3, B2, B1, B0
```

随后相邻两个位平面组成一对：

```text
R7R6, R5R4, R3R2, R1R0
G7G6, G5G4, G3G2, G1G0
B7B6, B5B4, B3B2, B1B0
```

因此每个颜色通道产生 4 个 DNA 平面，RGB 共生成 12 个 DNA 平面。

现在已经是全部 24 位平面参与加密，包括每个通道的最低两位。

## 六、按块进行 DNA 编码

两个二进制位首先组成一个 `[0,3]` 的数：

```text
value = 2 × 高位 + 低位
```

即：

| 位对 | 数值 |
|---|---:|
| 00 | 0 |
| 01 | 1 |
| 10 | 2 |
| 11 | 3 |

代码定义了 8 套 DNA 编码规则，将这四个数映射到：

```text
A、C、G、T
```

内部一般不存字符，而是存：

```text
A=0, C=1, G=2, T=3
```

每个自适应块使用一套规则。正常主流程中的规则编号来自：

```python
floor(RdMatrix[-1, block_index] × 10¹⁰) mod 8 + 1
```

因此规则编号范围是 1～8。

同一个块内：

- 所有像素使用同一 DNA 规则；
- 12 个 DNA 平面也使用同一规则。

最终得到：

```text
encoded_dna_matrix.shape = (12, H, W)
```

## 七、块内三维联合置乱

主流程调用的是 `permute_dna_blocks_v2()`。

对于每个块：

1. 提取其 DNA 数据，形状为 `(12,h,w)`。
2. 从 `RdMatrix` 前 12 行中提取该块对应位置的随机值。
3. 将 DNA 块和随机值都展平为长度 `12×h×w` 的一维数组。
4. 对随机值执行 `argsort()`。
5. 使用排序索引重排 DNA 数据。
6. 再恢复成 `(12,h,w)`。

这不是单纯的像素位置置乱，因为它把：

- DNA 平面维；
- 行坐标；
- 列坐标；

联合展平后一起置乱。因此，一个 DNA 符号既可能换空间位置，也可能进入另一个 DNA 平面。

不过置乱仍局限在当前自适应块内部，不会跨块移动。

## 八、同尺寸块之间交换

`shuffle_blocks_between_groups()` 按 `(height,width)` 对块进行分组。

例如：

```text
所有 16×16 块为一组
所有 8×8 块为一组
所有 8×7 块为另一组
```

只在尺寸完全相同的块之间随机交换，因为不同尺寸无法直接复制到对方位置。

这一步使用：

```text
seed + 1
```

交换时同时移动：

- DNA 块数据；
- 该块的 DNA 编码规则编号；
- 24 个原始位平面对应区域。

移动规则编号是必要的，因为移动后的 DNA 数据仍然需要使用其原来的编码规则进行解码。

不过，在现在“全部位平面都加密”的版本中，保存并交换 `original_bitplanes` 已经基本属于旧版本遗留逻辑：最终 24 个位平面都会被 DNA 解码结果覆盖。

## 九、块级 DNA 扩散 V2

主流程实际调用：

```python
apply_blockwise_dna_diffusionV2()
```

默认参数为：

```text
operation = xor
channel_mode = together
seed = seed + 2
```

对每个块生成一个 DNA 密钥矩阵，然后执行：

```text
D′ = D XOR K
```

这里的 XOR 是对 DNA 数字编码 `0～3` 执行的，其运算表恰好等价于两位整数的按位 XOR。

`channel_mode` 有两种模式：

- `together`：12 个 DNA 平面共享同一个 `(h,w)` 密钥矩阵；
- `separate`：每个 DNA 平面有独立密钥矩阵。

当前默认是 `together`。

需要特别说明：这个函数虽然叫“块扩散”，但实际没有前后像素之间的级联关系。一个位置发生改变不会通过这一层传播到块内其他位置。它更准确的名字应该是“块级 DNA 密钥混合”。

文件里还保留了旧版 `apply_blockwise_dna_diffusion()`，其中存在相邻列混合，但主流程没有调用它。

## 十、全局 DNA 扩散

主流程最后调用 `apply_global_dna_diffusion()`，并强制使用 XOR。

使用 `seed + 3` 生成：

- 全局像素排列 `permutation_indices`；
- 全局 DNA 密钥矩阵；
- 一个全局密钥编码规则。

默认：

```text
scheme = independent
parallel_mode = sequential_groups
parallel_size = 96
```

### 密钥模式

`scheme="synchronous"`：

```text
12 个 DNA 平面共享同一个密钥平面
```

`scheme="independent"`：

```text
12 个 DNA 平面分别生成独立密钥平面
```

当前默认是独立模式。

### whole_batch 模式

设全局排列为 `p`，其行为相当于：

```text
E[j] = I[p[j]] XOR I[p[j-1]] XOR K[p[j]]
```

第一个位置使用排列末尾作为前驱。

结果直接按照排列后的顺序重塑为 `(12,H,W)`，没有再散射回原坐标，因此这一步同时承担了全局位置置乱。

### sequential_groups 模式

把排列后的像素分成若干组。每组使用前一组最后一个原始 DNA 列作为公共前驱：

```text
carry(g) = 前一组末尾的输入列

E[j] = I[p[j]] XOR carry(g) XOR K[p[j]]
```

同一组内的所有位置共享相同的 `carry`，并不是逐位置递推。

因此，`parallel_size` 不只是性能参数，它会改变最终密文结果和算法结构。

## 十一、DNA 解码与密文重构

全局扩散完成后，代码再次按照每个块当前携带的 DNA 规则，把 12 个 DNA 平面解码成 24 个二进制位平面。

每个 DNA 符号恢复为两个位：

```text
DNA code → 2-bit value → 高位、低位
```

最后 `bitplanes_to_image()` 按照：

```text
R7…R0、G7…G0、B7…B0
```

重新组合出三个 8 位颜色通道，得到最终密文图像。

这里的“DNA 解码”不是解密，它只是把已经置乱和扩散后的 DNA 符号转换回普通 RGB 像素，以便保存和显示。

## 十二、当前实现需要在迁移前解决的问题

最重要的不是代码凌乱，而是可逆性。

### 1. 当前全局扩散不是一一映射

以 `whole_batch` 为例：

```text
E[j] = I[p[j]] XOR I[p[j-1]] XOR K[p[j]]
```

如果给所有输入列同时 XOR 同一个常量 `c`：

```text
(I[p[j]] XOR c) XOR (I[p[j-1]] XOR c)
= I[p[j]] XOR I[p[j-1]]
```

两个 `c` 会抵消。因此不同输入可能产生同一输出，无法唯一恢复原始 DNA 数据。

`sequential_groups` 也存在相同的全局常量抵消问题。

所以当前算法可以生成视觉上混乱的密文图像，但还不能保证严格无损解密。迁移到 `Encryption.py` 前，建议先重新定义一个有明确逆运算的全局扩散公式。

### 2. 没有解密流程

当前文件只实现了加密方向，没有：

- 逆全局扩散；
- 逆块扩散；
- 逆块交换；
- 逆块内置乱；
- 完整的 `decrypt()`。

### 3. 加密结果没有携带必要元数据

`encrypt_image()` 最终只返回密文图像，过程中生成的以下信息全部丢失：

- 自适应分块布局；
- 块间交换映射；
- DNA 规则编号；
- 扩散密钥矩阵；
- 全局排列；
- 算法配置和版本。

其中部分信息可以用种子重新生成，但明文相关的自适应分块无法直接从密文重算。

### 4. 随机来源比较分散

当前同时存在：

- `seed`：生成 CML 初始状态；
- `seed+1`：块间交换；
- `seed+2`：块密钥；
- `seed+3`：全局扩散。

这些关系散落在主函数中，尚未形成统一的密钥调度结构。

而且默认种子固定为 `2026`，它只能保证实验可重复，不能视为真正的秘密密钥。

### 5. 名称与实际行为不一致

例如：

- `Blockwise DNA diffusion V2` 实际是逐元素密钥混合；
- 日志中的 `inter-block permutation` 实际对应块内置乱；
- `decode_dna_matrix_to_bitplanes()` 是表示转换，不是解密；
- 部分中文注释存在乱码；
- `DNAEncodingResult.dna_matrix` 声明为数组，初始化时却赋值为 `None`。

### 6. 存在多套未使用的旧实现

当前主流程没有使用：

- `blockwise_dna_encode_legacy()`；
- `permute_dna_blocks()`；
- `apply_blockwise_dna_diffusion()`；
- 部分 DNA 字符格式转换函数；
- 若干可视化和辅助函数。

这正是文件显得杂乱的主要原因之一。

## 十三、迁移到 Encryption.py 时建议的结构

下一步实现时，我建议把 [Encryption.py](/C:/ImageEncryption/mywork/Encryption.py:1) 设计成一个清晰的编排层：

```text
EncryptionConfig
  ├── CML 参数
  ├── 分块参数
  ├── DNA 参数
  └── 扩散参数

EncryptionContext
  ├── 图像尺寸
  ├── 分块信息
  ├── DNA 规则
  ├── 置乱映射
  └── 密钥派生信息

EncryptionResult
  ├── encrypted_image
  ├── context
  └── stage_timings

Encryption
  ├── prepare_image()
  ├── generate_key_material()
  ├── partition()
  ├── dna_encode()
  ├── scramble_inside_blocks()
  ├── shuffle_blocks()
  ├── diffuse_blocks()
  ├── diffuse_globally()
  ├── build_cipher_image()
  ├── encrypt()
  └── decrypt()
```

核心加密逻辑、性能统计和绘图功能应分开。迁移时还应先确定每一步的逆变换，再实现正向变换，并为每个阶段编写可逆性测试：

```text
decode(encode(x)) == x
inverse_permute(permute(x)) == x
inverse_diffuse(diffuse(x)) == x
decrypt(encrypt(image)) == image
```

总体而言，现有代码的设计思想很清晰：自适应分块、全位平面 DNA 表示、块内置乱、块间交换、局部混合和全局混合。但当前实现更接近“生成高混乱度密文图像的实验流水线”，还不是一套已经闭合、可验证、可逆的完整图像密码系统。