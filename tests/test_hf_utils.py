import pytest

from frank_eq.data.hf_backend import (
    CHAT_SYSTEM_CONTRACT,
    HFModelAdapter,
    choose_answer_token_pair,
    resolve_layer_indices,
)


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        mapping = {" A": [1], " B": [2], "A": [3, 4], "B": [5, 6]}
        return mapping[text]

    chat_template = "{{ bos_token }}{{ messages[0]['content'] }}"


def test_hf_backend_utility_contracts() -> None:
    assert resolve_layer_indices(24, [0.25, 0.5, 0.75]) == [6, 12, 18]
    labels, ids = choose_answer_token_pair(FakeTokenizer(), [["A", "B"], [" A", " B"]])
    assert labels == (" A", " B")
    assert ids == (1, 2)
    with pytest.raises(ValueError):
        resolve_layer_indices(2, [0.49, 0.51])


class StubTokenizer:
    chat_template = "{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n<|im_start|>user\\n' + messages[1]['content'] + '<|im_end|>\\n<|im_start|>assistant\\n' }}"

    def __call__(self, text, add_special_tokens=True, return_tensors="pt", truncation=False):
        return {"input_ids": [[1, 2, 3]]}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        assert add_generation_prompt is True
        return "<|im_start|>system\n" + messages[0]["content"] + "<|im_end|>\n<|im_start|>user\n" + messages[1]["content"] + "<|im_end|>\n<|im_start|>assistant\n"


class StubCapture:
    prompt_format = "chat"
    max_length = 512


class StubSpec:
    model_id = "stub"
    tokenizer_id = None
    hf_id = "stub/hf"
    revision = None
    trust_remote_code = False


def _stub_adapter(prompt_format: str) -> HFModelAdapter:
    adapter = object.__new__(HFModelAdapter)
    adapter.spec = StubSpec()
    adapter.capture = StubCapture()
    adapter.capture.prompt_format = prompt_format
    adapter.tokenizer = StubTokenizer()
    adapter.device = None
    return adapter


def test_chat_format_wraps_world_statement() -> None:
    adapter = _stub_adapter("chat")
    formatted = adapter._format_prefix("world statement")
    assert formatted.startswith("<|im_start|>system\n")
    assert CHAT_SYSTEM_CONTRACT in formatted
    assert "world statement" in formatted
    assert formatted.endswith("<|im_start|>assistant\n")


def test_raw_format_passes_through() -> None:
    adapter = _stub_adapter("raw")
    assert adapter._format_prefix("world statement") == "world statement"


class StubEncoder:
    def __init__(self, width: int = 3):
        self.shape = (1, width)

    def to(self, device):
        return self


def test_chat_prefix_tokenization_suppresses_special_tokens() -> None:
    adapter = _stub_adapter("chat")
    calls = []

    def recording_tokenizer(text, add_special_tokens=True, return_tensors="pt", truncation=False):
        calls.append(add_special_tokens)
        return {"input_ids": StubEncoder()}

    adapter.tokenizer = recording_tokenizer
    adapter._tokenize("anything")
    assert calls == [False]


def test_raw_prefix_tokenization_keeps_special_tokens() -> None:
    adapter = _stub_adapter("raw")
    calls = []

    def recording_tokenizer(text, add_special_tokens=True, return_tensors="pt", truncation=False):
        calls.append(add_special_tokens)
        return {"input_ids": StubEncoder()}

    adapter.tokenizer = recording_tokenizer
    adapter._tokenize("anything")
    assert calls == [True]
