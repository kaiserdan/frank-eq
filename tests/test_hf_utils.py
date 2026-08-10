import pytest

from frank_eq.data.hf_backend import choose_answer_token_pair, resolve_layer_indices


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        mapping = {" A": [1], " B": [2], "A": [3, 4], "B": [5, 6]}
        return mapping[text]


def test_hf_backend_utility_contracts() -> None:
    assert resolve_layer_indices(24, [0.25, 0.5, 0.75]) == [6, 12, 18]
    labels, ids = choose_answer_token_pair(FakeTokenizer(), [["A", "B"], [" A", " B"]])
    assert labels == (" A", " B")
    assert ids == (1, 2)
    with pytest.raises(ValueError):
        resolve_layer_indices(2, [0.49, 0.51])
