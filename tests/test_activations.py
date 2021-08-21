import random
import unittest

import torch
from torch.nn import functional as F

from megatron.model.glu_activations import liglu, geglu, reglu, swiglu
from megatron.testing_utils import set_seed


class TestActivations(unittest.TestCase):
    def setUp(self):
        """setup an input of reasonable size"""
        set_seed()
        self.batch_size = random.randint(2, 64)
        self.seq_len = random.randint(256, 1025)
        self.num_channels = random.randint(1, 384) * 2
        self.x = torch.randn(self.batch_size, self.seq_len, self.num_channels)
        self.x1, self.x2 = self.x.chunk(2, dim=-1)

    def test_shapes(self):
        # glu should halve the last dimension
        output_shape = [self.batch_size, self.seq_len, self.num_channels // 2]
        for activation_fn in [liglu, geglu, reglu, swiglu]:
            output = activation_fn(self.x)
            self.assertEqual(list(output.shape), output_shape)

    def test_liglu(self):
        expected = self.x1 * self.x2
        torch.testing.assert_equal(liglu(self.x), expected)

    def test_geglu(self):
        expected = self.x1 * F.gelu(self.x2)
        torch.testing.assert_equal(geglu(self.x), expected)

    def test_reglu(self):
        expected = self.x1 * F.relu(self.x2)
        torch.testing.assert_equal(reglu(self.x), expected)

    def test_swiglu(self):
        expected = self.x1 * F.silu(self.x2)
        torch.testing.assert_equal(swiglu(self.x), expected)
