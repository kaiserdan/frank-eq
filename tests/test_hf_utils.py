import pytest
import torch

from frank_eq.data.hf_backend import (
    CHAT_ACKNOWLEDGEMENT,
    CHAT_SYSTEM_CONTRACT,
    HFModelAdapter,
    choose_answer_token_pair,
    resolve_layer_indices,
)


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        mapping = {" A": [1], " B": [2], "A": [3, 4], "B": [5, 6]}
        return mapping[text]

    chat_template = "stub"


def test_hf_backend_utility_contracts() -> None:
    assert resolve_layer_indices(24, [0.25, 0.5, 0.75]) == [6, 12, 18]
    labels, ids = choose_answer_token_pair(FakeTokenizer(), [["A", "B"], [" A", " B"]])
    assert labels == (" A", " B")
    assert ids == (1, 2)
    with pytest.raises(ValueError):
        resolve_layer_indices(2, [0.49, 0.51])


class StubTokenizer:
    chat_template = "stub"

    def __init__(self):
        self.last_messages = None
        self.last_kwargs = None

    @staticmethod
    def _encode_text(text: str) -> torch.Tensor:
        return torch.tensor([[ord(character) + 1 for character in text]], dtype=torch.long)

    def __call__(self, text, add_special_tokens=True, return_tensors="pt", truncation=False):
        return {"input_ids": self._encode_text(text)}

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    ):
        assert tokenize is False
        self.last_messages = [dict(message) for message in messages]
        self.last_kwargs = dict(kwargs)
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered


class StubCapture:
    prompt_format = "chat"
    max_length = 4096
    chat_template_kwargs = {"enable_thinking": False}


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
    adapter.device = torch.device("cpu")
    return adapter


def test_chat_format_preserves_historical_assistant_header() -> None:
    adapter = _stub_adapter("chat")
    formatted = adapter._format_prefix("world statement")
    assert CHAT_SYSTEM_CONTRACT in formatted
    assert "<user>world statement</user>" in formatted
    assert formatted.endswith("<assistant>")


def test_chat_turn_reveals_operation_as_new_user_turn_with_exact_prefix() -> None:
    adapter = _stub_adapter("chat_turn")
    world = "world statement"
    query = "registered operation"
    prefix = adapter._format_prefix(world)
    assert CHAT_ACKNOWLEDGEMENT in prefix
    assert world in prefix
    assert prefix.endswith(f"<assistant>{CHAT_ACKNOWLEDGEMENT}</assistant>")
    prefix_ids = adapter._tokenize(prefix)
    suffix_ids = adapter._query_ids(
        query,
        world_statement=world,
        prefix_ids=prefix_ids,
    )
    combined = torch.cat([prefix_ids, suffix_ids], dim=1)
    expected_text = (
        f"<system>{CHAT_SYSTEM_CONTRACT}\n\n{world}</system>"
        f"<assistant>{CHAT_ACKNOWLEDGEMENT}</assistant>"
        f"<user>{query}</user><assistant>"
    )
    expected = adapter.tokenizer(expected_text, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ]
    assert torch.equal(combined, expected)
    assert [message["role"] for message in adapter.tokenizer.last_messages] == [
        "system",
        "assistant",
        "user",
    ]
    assert adapter.tokenizer.last_kwargs == {"enable_thinking": False}


class QwenLikeTokenizer(StubTokenizer):
    """Mimics Qwen3's context-dependent template: a final assistant message that
    follows the last user message gains a ``<think>`` wrapper that disappears
    once a later user message exists."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        assert tokenize is False
        self.last_messages = [dict(message) for message in messages]
        self.last_kwargs = dict(kwargs)
        last_user = max(
            (index for index, message in enumerate(messages) if message["role"] == "user"),
            default=len(messages) - 1,
        )
        rendered = ""
        for index, message in enumerate(messages):
            if message["role"] == "system":
                rendered += f"<system>{message['content']}</system>"
            elif message["role"] == "user":
                rendered += f"<user>{message['content']}</user>"
            elif message["role"] == "assistant":
                if index > last_user and (index == len(messages) - 1):
                    rendered += f"<assistant><think></think>{message['content']}</assistant>"
                else:
                    rendered += f"<assistant>{message['content']}</assistant>"
        if add_generation_prompt:
            rendered += "<assistant><think></think>"
        return rendered


def test_chat_turn_exact_prefix_under_context_dependent_template() -> None:
    """Regression: Qwen3's template renders a trailing post-query assistant
    message with a think wrapper that vanishes once a later user message
    exists. The chat_turn construction must remain token-prefix stable."""

    adapter = _stub_adapter("chat_turn")
    adapter.tokenizer = QwenLikeTokenizer()
    world = "world statement"
    query = "registered operation"
    prefix = adapter._format_prefix(world)
    assert "<think>" not in prefix
    prefix_ids = adapter._tokenize(prefix)
    suffix_ids = adapter._query_ids(query, world_statement=world, prefix_ids=prefix_ids)
    combined = torch.cat([prefix_ids, suffix_ids], dim=1)
    expected_text = (
        f"<system>{CHAT_SYSTEM_CONTRACT}\n\n{world}</system>"
        f"<assistant>{CHAT_ACKNOWLEDGEMENT}</assistant>"
        f"<user>{query}</user><assistant><think></think>"
    )
    expected = adapter.tokenizer(expected_text, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ]
    assert torch.equal(combined, expected)


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


def test_chat_turn_prefix_tokenization_suppresses_special_tokens() -> None:
    adapter = _stub_adapter("chat_turn")
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
