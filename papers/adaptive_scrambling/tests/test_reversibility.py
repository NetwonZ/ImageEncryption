"""论文流程的单元测试：验证置乱、扩散和完整流程可逆。"""

from __future__ import annotations

import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_scrambling import adaptive_scramble, adaptive_unscramble, choose_round_configs
from chaotic_neuron import generate_chaotic_sequences
from dynamic_diffusion import diffuse_decrypt, diffuse_encrypt
from pipeline import ImageCryptosystem


class ReversibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        y, x = np.mgrid[0:64, 0:64]
        self.image = ((3 * x + 5 * y + (x * y) % 71) % 256).astype(np.uint8)

    def test_chaotic_sequence_shape_and_determinism(self) -> None:
        first = generate_chaotic_sequences(128, pre_iterations=20)
        second = generate_chaotic_sequences(128, pre_iterations=20)
        for a, b in zip(first, second):
            self.assertEqual(a.shape, (128,))
            np.testing.assert_array_equal(a, b)

    def test_one_scramble_round_is_reversible(self) -> None:
        config = choose_round_configs(64)[0]
        scrambled, key = adaptive_scramble(self.image, seed=12345, config=config)
        restored = adaptive_unscramble(scrambled, key)
        np.testing.assert_array_equal(restored, self.image)

    def test_diffusion_is_reversible(self) -> None:
        x, _, z = generate_chaotic_sequences(self.image.size, pre_iterations=20)
        cipher, _, _, _ = diffuse_encrypt(self.image, x, z)
        restored = diffuse_decrypt(cipher, x, z)
        np.testing.assert_array_equal(restored, self.image)

    def test_complete_pipeline_is_reversible(self) -> None:
        result = ImageCryptosystem(pre_iterations=20).encrypt(self.image)
        np.testing.assert_array_equal(result.decrypted, self.image)
        self.assertFalse(np.array_equal(result.ciphertext, self.image))


if __name__ == "__main__":
    unittest.main()
