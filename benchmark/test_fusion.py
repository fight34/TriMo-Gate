"""Synthetic CPU regressions for all six fusion choices; no training data needed."""

import itertools
from types import SimpleNamespace
import unittest

import torch

from Model import FUSION_CLASSES, GatedFusion, MeanFusion, MultimodalModel, WeightedSumFusion


MASKS = list(itertools.product((False, True), repeat=3))[1:]
ENCODERS = ("smiles_model", "graph_model", "point_cloud_model")
torch.set_num_threads(1)


def settings(flags=(True, True, True)):
    return dict(vocab_size=12, smiles_length=8, d_smiles=8,
                smiles_num_heads=2, smiles_num_layers=1, cnn_kernels=2,
                d_node=3, d_edge=2, d_hidden=8, graph_num_layers=1,
                d_point_input=9, d_point_cloud=16, d_point_output=8,
                d_output=2, use_smiles=flags[0], use_graph=flags[1],
                use_point_cloud=flags[2], dropout=0.0)


def inputs():
    # The second graph is edgeless, exercising the atom-readout fallback too.
    graph = SimpleNamespace(
        x=torch.randn(3, 3), edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_attr=torch.randn(2, 2), batch=torch.tensor([0, 0, 1]))
    smiles = torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0],
                           [2, 3, 4, 5, 6, 7, 0, 0]])
    return smiles, graph, torch.randn(2, 4, 9), torch.tensor([4, 3])


class FusionChecks(unittest.TestCase):
    def test_six_fusions_seven_modalities_backward_and_strict_reload(self):
        self.assertEqual(set(FUSION_CLASSES), {
            "concat", "mean", "weighted", "attention", "gate_no_interaction", "gate"
        })
        torch.manual_seed(42)
        data = inputs()
        for kind, fusion_class in FUSION_CLASSES.items():
            for flags in MASKS:
                with self.subTest(fusion=kind, modalities=flags):
                    torch.manual_seed(42)
                    model = MultimodalModel(**settings(flags), fusion_type=kind)
                    if sum(flags) == 1:
                        self.assertIsNone(model.fusion_model)
                        self.assertIsNone(model.fusion_type)
                    else:
                        self.assertIsInstance(model.fusion_model, fusion_class)
                        self.assertEqual(model.fusion_type, kind)
                    output, embeddings = model(*data, return_embeddings=True)
                    self.assertEqual(output.shape, (2, 2))
                    self.assertTrue(torch.isfinite(output).all())
                    for active, embedding in zip(flags, embeddings):
                        if active:
                            self.assertEqual(embedding.shape, (2, 8))
                            self.assertTrue(torch.isfinite(embedding).all())
                            embedding.retain_grad()
                        else:
                            self.assertIsNone(embedding)
                    output.square().mean().backward()
                    for active, name, embedding in zip(flags, ENCODERS, embeddings):
                        if active:
                            self.assertIsNotNone(embedding.grad, name)
                            self.assertGreater(torch.count_nonzero(embedding.grad).item(), 0, name)
                            self.assertTrue(any(
                                p.grad is not None and torch.count_nonzero(p.grad).item() > 0
                                for p in getattr(model, name).parameters()), name)
                    for name, parameter in model.named_parameters():
                        if parameter.grad is not None:
                            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

                    restored = MultimodalModel(**settings(flags), fusion_type=kind).eval()
                    restored.load_state_dict(model.state_dict(), strict=True)
                    model.eval()
                    with torch.no_grad():
                        torch.testing.assert_close(restored(*data), model(*data), rtol=0, atol=0)

    def test_same_seed_preserves_encoders_and_five_smiles_kernels(self):
        reference = None
        for kind in FUSION_CLASSES:
            with self.subTest(fusion=kind):
                torch.manual_seed(42)
                model = MultimodalModel(**settings(), fusion_type=kind)
                encoders = {name: tensor for name, tensor in model.state_dict().items()
                            if name.startswith(tuple(name + "." for name in ENCODERS))}
                if reference is None:
                    reference = encoders
                self.assertEqual(encoders.keys(), reference.keys())
                for name, tensor in encoders.items():
                    torch.testing.assert_close(tensor, reference[name], rtol=0, atol=0)
                cnn = model.smiles_model.cnn
                self.assertEqual(cnn.kernel_sizes, [1, 2, 3, 4, 5])
                self.assertEqual([layer[0].kernel_size for layer in cnn.conv_layers],
                                 [(1,), (2,), (3,), (4,), (5,)])
                self.assertEqual(cnn.fc[0].in_features, 5 * cnn.out_channels)

    def test_default_gate_and_export_shape_with_missing_modalities(self):
        model = MultimodalModel(**settings())
        self.assertEqual(model.fusion_type, "gate")
        self.assertIsInstance(model.fusion_model, GatedFusion)
        for kind in ("gate", "gate_no_interaction"):
            fusion = FUSION_CLASSES[kind](8, 8, 8, 2, dropout=0.0)
            for flags in MASKS:
                with self.subTest(fusion=kind, modalities=flags):
                    features = [torch.randn(2, 8) if active else None for active in flags]
                    _, returned = fusion(*features, return_gates=True)
                    exported = torch.stack(fusion.gates, dim=1)
                    self.assertEqual(exported.shape, (2, 3, 8))
                    self.assertFalse(exported.requires_grad)
                    self.assertTrue(torch.isfinite(exported).all())
                    torch.testing.assert_close(exported, torch.stack(returned, dim=1))
                    torch.testing.assert_close(exported.sum(dim=1), torch.ones(2, 8))
                    self.assertEqual(torch.count_nonzero(exported[:, ~torch.tensor(flags)]).item(), 0)

    def test_zero_scalar_logits_equal_mean_for_available_modalities(self):
        mean = MeanFusion(8, 8, 8, 2, dropout=0.0)
        weighted = WeightedSumFusion(8, 8, 8, 2, dropout=0.0)
        weighted.load_state_dict(dict(mean.state_dict(), alpha_logits=torch.zeros(3)), strict=True)
        for flags in MASKS:
            with self.subTest(modalities=flags):
                features = [torch.randn(2, 8) if active else None for active in flags]
                output, weights = weighted(*features, return_weights=True)
                torch.testing.assert_close(output, mean(*features), rtol=0, atol=0)
                torch.testing.assert_close(weights.sum(), torch.tensor(1.0))
                self.assertEqual(torch.count_nonzero(weights[~torch.tensor(flags)]).item(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
