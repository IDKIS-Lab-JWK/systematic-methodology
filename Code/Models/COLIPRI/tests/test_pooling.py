from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from COLIPRI.pooling import (  # noqa: E402
    COLIPRIPoolingOptimisedLinearProbe,
    POOLING_NAMES,
    dense_to_tokens,
)


def test_dense_to_tokens_is_a_noncopying_view_with_expected_order() -> None:
    dense = torch.arange(2 * 12 * 2 * 2 * 2, dtype=torch.float32).reshape(
        2, 12, 2, 2, 2
    )
    tokens = dense_to_tokens(dense)

    assert tokens.shape == (2, 8, 12)
    assert tokens.data_ptr() == dense.data_ptr()
    assert not tokens.is_contiguous()
    assert torch.equal(tokens[:, 0, :], dense[:, :, 0, 0, 0])


@pytest.mark.parametrize("pooling", POOLING_NAMES)
def test_all_publication_pooling_heads_produce_five_class_logits(pooling: str) -> None:
    torch.manual_seed(7)
    model = COLIPRIPoolingOptimisedLinearProbe(
        pooling=pooling, embed_dim=12, num_heads=3, num_classes=5
    ).eval()
    dense = torch.randn(2, 12, 2, 2, 2)
    tokens = dense_to_tokens(dense)

    with torch.inference_mode():
        dense_logits = model(dense)
        token_logits = model.forward_tokens(tokens)

    assert dense_logits.shape == (2, 5)
    assert torch.isfinite(dense_logits).all()
    assert torch.equal(dense_logits, token_logits)
    assert model.strict_linear is (pooling in ("average", "max"))


def test_average_and_max_are_exact_spatial_reductions() -> None:
    dense = torch.arange(12 * 2 * 2 * 2, dtype=torch.float32).reshape(
        1, 12, 2, 2, 2
    )
    tokens = dense_to_tokens(dense)

    average = COLIPRIPoolingOptimisedLinearProbe(
        pooling="average", embed_dim=12, num_heads=3, num_classes=5
    )
    maximum = COLIPRIPoolingOptimisedLinearProbe(
        pooling="max", embed_dim=12, num_heads=3, num_classes=5
    )

    assert torch.equal(average.pool(tokens), dense.flatten(2).mean(dim=2))
    assert torch.equal(maximum.pool(tokens), dense.flatten(2).amax(dim=2))

