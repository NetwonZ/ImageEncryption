"""全局扩散增强开关的正确性验证（跨平面扩散 + 双向扩散）。

覆盖四组开关组合，逐组检查：
  1. 加解密完全可逆（原图逐像素相等）；
  2. 关闭两个开关时与修改前的旧算法逐字节一致（向后兼容）；
  3. 跨平面扩散确实带来了平面间的串扰（开启后单平面扰动会波及其他平面）；
  4. 开启后密文与关闭时不同（功能真的生效，不是空操作）。

用法：python -m mywork.VerifyGlobalDiffusion
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np

from .Encryption import (
    DNA_PLANE_COUNT,
    DeEncrypter,
    Encrypter,
    EncryptionConfig,
    _image_to_bitplanes,
)

IMAGE_PATHS = [
    Path(r"C:\ImageEncryption\images\img1.png"),
    Path(r"C:\ImageEncryption\images\img2.png"),
]

# 修改前（旧算法，单向 + 平面内孤立）的密文哈希基线。
# 用 git HEAD 版本的 Encryption.py 以同样的 user_key 生成，用于证明
# 「两个开关都关闭」时新代码与旧代码逐字节一致（向后兼容）。
BASELINE_SHA256 = {
    # user_key = "VERIFY"
    ("img1.png", "VERIFY"): "2b8bba0ca1419f659aebe68125bf729721f08cf74038f028e1a329ac81b1d9d3",
    ("img2.png", "VERIFY"): "c7c76570c1586114850184469d35b25bb5f68a5de35d1c955b2ab4b775b179d5",
    # user_key = "BASELINE"
    ("img1.png", "BASELINE"): "d166b1cd27242423115434c843ffa08cc2e48f1435a9fa6d6179fb7e14dad82b",
    ("img2.png", "BASELINE"): "09a04b8d1fad7d3c417f9ce3433ffc6586038534528d3fd1558f5407013f1458",
}

BASE_CONFIG = EncryptionConfig(
    seed=2026,
    b_max=64,
    b_min=4,
    block_operation="xor",
    global_parallel_size=2,
    global_permutation_method="global_bit_permutation_v2",
)

COMBINATIONS = [
    # (cross_plane, bidirectional, 标签)
    (False, False, "关闭/关闭 (旧算法)"),
    (True, False, "跨平面/单向"),
    (False, True, "平面内/双向"),
    (True, True, "跨平面/双向 (默认)"),
]


def roundtrip(encrypter, decrypter, image_path: Path, config: EncryptionConfig) -> dict:
    encrypter.config = config
    decrypter.config = config
    encrypter.set_user_key("VERIFY", image=image_path)
    original = encrypter.load_image(image_path)
    try:
        result = encrypter.encrypt(image_path, print_profile=False)
        decrypted = decrypter.decrypt(
            result.encrypted_image, result.metadata, print_profile=False
        ).decrypted_image
    finally:
        encrypter.resume_user_key()
    return {
        "cipher": result.encrypted_image,
        "exact": bool(np.array_equal(original, decrypted)),
        "sha256": hashlib.sha256(result.encrypted_image.tobytes()).hexdigest(),
        "diff_ratio": float(np.mean(original != decrypted)),
    }


def plane_isolation_probe(encrypter, image_path: Path, config: EncryptionConfig) -> dict:
    """跨平面扩散探针：在轨迹起点扰动一个像素，统计最终密文的平面间串扰。

    注意扰动必须放在**第一组的组尾**（轨迹索引 parallel_size-1）。
    原因：反馈递推为
        Z_t = P_t XOR Roll(C_{t-1}) XOR K_t
    递推量只取每组的**末端**密文。所以扰动若不是落在组尾，它根本不会进入
    任何反馈值，开/关移位都只影响那一个像素 —— 测不出差别。只有把扰动
    放在第一组组尾时：
      * 关闭移位：C_0 只落在平面 0 上，扰动沿轨迹始终只在平面 0 内传播；
      * 打开移位：每经过一组 C 就被 Roll 一次，受影响的平面号逐步 +shift，
        经过约 12 组后 12 个平面全部被波及。

    返回受影响平面数与受影响像素比例（雪崩程度）。
    """
    encrypter.config = config
    encrypter.set_user_key("VERIFY", image=image_path)
    try:
        original = encrypter.load_image(image_path)
        result = encrypter.encrypt(image_path, print_profile=False)
        key_material = result.metadata.key_material
        blocks = result.metadata.blocks

        dna = encrypter.encode_to_dna(
            _image_to_bitplanes(original),
            blocks,
            key_material.original_block_rule_ids,
        )
        perturbed = dna.copy()
        # 轨迹索引 parallel_size-1 = 第一组组尾（反馈量的取样点）
        trajectory_index = min(config.global_parallel_size - 1, dna.shape[1] * dna.shape[2] - 1)
        position = int(key_material.global_permutation[trajectory_index])
        row, col = divmod(position, dna.shape[2])
        perturbed[0, row, col] ^= 1

        kwargs = dict(
            parallel_size=config.global_parallel_size,
            cross_plane_diffusion=config.cross_plane_diffusion,
            bidirectional_diffusion=config.bidirectional_diffusion,
        )
        base_cipher = encrypter.apply_global_diffusion(dna, key_material, **kwargs)
        pert_cipher = encrypter.apply_global_diffusion(perturbed, key_material, **kwargs)

        changed = base_cipher != pert_cipher
        planes_touched = [int(p) for p in np.flatnonzero(changed.reshape(DNA_PLANE_COUNT, -1).any(axis=1))]
        return {
            "planes_touched": planes_touched,
            "plane_count": len(planes_touched),
            "changed_bytes": int(changed.sum()),
            "changed_ratio": float(changed.mean()),
        }
    finally:
        encrypter.resume_user_key()


def shift_sweep(encrypter, decrypter, image_path: Path) -> bool:
    """遍历所有合法循环移位量，确认每一档都可逆（供后续调参使用）。

    CROSS_PLANE_SHIFT 取 1..11 均为合法值（0 和 12 等价于不启用移位）。
    这里对每一档做一次加密 + 解密，必须全部逐像素还原。同时打印每档的
    平面串扰数，方便比较不同移位量对扩散范围的影响。
    """
    print("\n" + "=" * 78)
    print("循环移位量扫描（CROSS_PLANE_SHIFT = 1..11，均须可逆）")
    print("=" * 78)
    print(f"{'shift':>6}{'可逆':>8}{'受影响平面':>12}{'受影响像素比例':>18}")
    all_ok = True
    for shift in range(1, DNA_PLANE_COUNT):
        # 通过 config.cross_plane_shift 驱动完整加解密路径（而不是绕过配置
        # 直接给底层函数传参），确保配置文件里改这一项真的生效。
        config = replace(
            BASE_CONFIG,
            cross_plane_diffusion=True,
            bidirectional_diffusion=True,
            cross_plane_shift=shift,
        )
        encrypter.config = config
        decrypter.config = config
        encrypter.set_user_key("VERIFY", image=image_path)
        try:
            original = encrypter.load_image(image_path)
            result = encrypter.encrypt(image_path, print_profile=False)
            decrypted = decrypter.decrypt(
                result.encrypted_image, result.metadata, print_profile=False
            ).decrypted_image
            exact = bool(np.array_equal(original, decrypted))

            # 单点扰动，比较该移位量对跨平面扩散范围的影响
            key_material = result.metadata.key_material
            blocks = result.metadata.blocks
            dna = encrypter.encode_to_dna(
                _image_to_bitplanes(original), blocks, key_material.original_block_rule_ids
            )
            perturbed = dna.copy()
            trajectory_index = config.global_parallel_size - 1
            position = int(key_material.global_permutation[trajectory_index])
            r, c = divmod(position, dna.shape[2])
            perturbed[0, r, c] ^= 1
            kwargs = dict(
                parallel_size=config.global_parallel_size,
                cross_plane_diffusion=True,
                bidirectional_diffusion=True,
                cross_plane_shift=shift,
            )
            base_cipher = encrypter.apply_global_diffusion(dna, key_material, **kwargs)
            pert_cipher = encrypter.apply_global_diffusion(perturbed, key_material, **kwargs)
            changed = base_cipher != pert_cipher
            planes = int(changed.reshape(DNA_PLANE_COUNT, -1).any(axis=1).sum())
            print(f"{shift:>6}{'OK' if exact else 'FAIL':>8}{planes:>12}"
                  f"{changed.mean() * 100:>17.3f}%")
            all_ok &= exact
        finally:
            encrypter.resume_user_key()
    return all_ok


def main() -> None:
    print("=" * 78)
    print("全局扩散增强开关正确性验证")
    print("=" * 78)

    encrypter = Encrypter(BASE_CONFIG)
    decrypter = DeEncrypter(BASE_CONFIG)

    ok_all = True
    for image_path in IMAGE_PATHS:
        print(f"\n>>> {image_path.name}")
        results = {}
        for cross_plane, bidirectional, label in COMBINATIONS:
            config = replace(
                BASE_CONFIG,
                cross_plane_diffusion=cross_plane,
                bidirectional_diffusion=bidirectional,
            )
            data = roundtrip(encrypter, decrypter, image_path, config)
            results[label] = data
            status = "OK " if data["exact"] else "FAIL"
            ok_all &= data["exact"]
            print(f"  {status} {label:<22} exact={data['exact']}  "
                  f"cipher[0:16]={data['sha256'][:16]}")

        # 向后兼容：两个开关都关闭时必须与旧算法逐字节一致
        expected = BASELINE_SHA256.get((image_path.name, "VERIFY"))
        if expected is not None:
            actual = results["关闭/关闭 (旧算法)"]["sha256"]
            match = actual == expected
            ok_all &= match
            print(f"  {'OK ' if match else 'FAIL'} 向后兼容: 关闭开关的密文哈希 "
                  f"{'==' if match else '!='} 修改前基线 (git HEAD 版 Encryption.py)")
            if not match:
                print(f"       期望 {expected}\n       实际 {actual}")

        # 增强确实改变了密文
        off = results["关闭/关闭 (旧算法)"]["sha256"]
        for label in ("跨平面/单向", "平面内/双向", "跨平面/双向 (默认)"):
            changed = results[label]["sha256"] != off
            ok_all &= changed
            print(f"  {'OK ' if changed else 'FAIL'} {label}: 密文与旧算法不同 = {changed}")

    print("\n" + "=" * 78)
    print("跨平面扩散的平面串扰探针（第一组组尾单点扰动）")
    print("=" * 78)
    image_path = IMAGE_PATHS[0]
    probe_rows = []
    for cross_plane, bidirectional, label in COMBINATIONS:
        config = replace(
            BASE_CONFIG,
            cross_plane_diffusion=cross_plane,
            bidirectional_diffusion=bidirectional,
        )
        data = plane_isolation_probe(encrypter, image_path, config)
        probe_rows.append((label, cross_plane, data))
        print(f"  {label:<22} 受影响平面 = {data['plane_count']:>2}/12  "
              f"受影响像素 = {data['changed_bytes']:>7} "
              f"({data['changed_ratio'] * 100:.3f}%)")

    print("\n  判据：cross_plane=关闭 时应只有 1 个平面受影响（平面内孤立）；")
    print("        cross_plane=开启 时应显著多于 1 个平面（跨平面串扰成立）。")
    probe_ok = True
    for label, cross_plane, data in probe_rows:
        if cross_plane:
            good = data["plane_count"] > 1
        else:
            good = data["plane_count"] == 1
        probe_ok &= good
        print(f"    {'OK ' if good else 'FAIL'} {label}: 平面数 {data['plane_count']}")
    ok_all &= probe_ok

    print("\n" + "=" * 78)
    shift_ok = shift_sweep(encrypter, decrypter, IMAGE_PATHS[0])
    ok_all &= shift_ok

    print("\n" + "=" * 78)
    print("总判定:", "全部通过" if ok_all else "存在失败项，请检查上面 FAIL 行")
    print("=" * 78)


if __name__ == "__main__":
    main()
