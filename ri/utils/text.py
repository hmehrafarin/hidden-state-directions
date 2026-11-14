from __future__ import annotations

from typing import Any


def prompt_text_from_rendered(rendered_prompt: Any) -> str:
    """The text of a rendered prompt, or the last message content of a conversation."""
    if isinstance(rendered_prompt, str):
        return rendered_prompt
    if isinstance(rendered_prompt, (list, tuple)):
        for message in reversed(rendered_prompt):
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
    return ""
