"""Exact official Qwen3 system/user ChatML form; reject injected role delimiters."""


def render(messages):
    if (
        type(messages) is not list
        or len(messages) != 2
        or [m.get("role") for m in messages] != ["system", "user"]
    ):
        raise ValueError("exact system/user roles required")
    for item in messages:
        if (
            set(item) != {"role", "content"}
            or type(item["content"]) is not str
            or any(
                marker in item["content"] for marker in ["<|im_start|>", "<|im_end|>"]
            )
        ):
            raise ValueError("literal bounded message content required")
    return (
        "".join(
            "<|im_start|>" + item["role"] + "\n" + item["content"] + "<|im_end|>\n"
            for item in messages
        )
        + "<|im_start|>assistant\n"
    )
