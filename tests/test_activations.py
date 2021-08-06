import unittest
import math

import torch

from megatron.model.activations import liglu, geglu, reglu, swiglu


class TestActivations(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor([0.1, -0.7])

    def test_shapes(self):
        batch_size = 8
        seq_len = 100
        num_channels = 768
        x = torch.randn(batch_size, seq_len, num_channels)
        # glu should halve the last dimension
        output_shape = [batch_size, seq_len, num_channels // 2]
        for activation_fn in [liglu, geglu, reglu, swiglu]:
            output = activation_fn(x)
            self.assertEqual(list(output.shape), output_shape)

    def test_liglu(self):
        expected = torch.tensor([-0.07])
        self.assertEqual(liglu(self.x), expected)
    
    def test_geglu(self):
        """compute gelu output according to its definition"""
        normal_cdf = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
        gelu_output = self.x[1] * normal_cdf(self.x[1].item())
        expected = self.x[0] * gelu_output
        self.assertAlmostEqual(geglu(self.x).item(), expected.item())

    def test_reglu(self):
        expected = torch.tensor([0.])
        self.assertEqual(reglu(self.x), expected)

    def test_swiglu(self):
        """compute swish output according to its definition"""
        swish_output = self.x[1] / (1 + math.pow(math.e, -self.x[1].item()))
        expected = self.x[0] * swish_output
        self.assertAlmostEqual(swiglu(self.x).item(), expected.item())