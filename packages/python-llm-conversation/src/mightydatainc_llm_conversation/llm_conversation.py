import json
import re
from typing import Any, Callable

from .llm_providers import get_model_name, identify_llm_provider
from .llm_submit import llm_submit


class LLMConversation(list[dict[str, Any]]):
    """Stateful conversation wrapper around provider clients.

    `LLMConversation` extends `list` so message history can be indexed,
    iterated, and serialized directly. Helper methods cover the full lifecycle:
    appending role-specific messages, submitting to the provider, and reading
    the latest assistant reply in string or dict form.
    """

    def __init__(
        self,
        ai_client: Any | None = None,
        messages: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ):
        """Create a conversation with optional initial history and defaults.

        Args:
            ai_client: Provider client object used by `llm_submit`.
            messages: Optional initial message history.
            model: Optional default model for future `submit` calls.
        """
        super().__init__(messages or [])
        self.ai_client = ai_client
        self.model = model
        self.last_reply: Any = None

    @property
    def llm_provider(self) -> str | None:
        """Return provider name inferred from `ai_client`, or None."""
        if self.ai_client is None:
            return None
        return identify_llm_provider(self.ai_client)

    def assign_messages(self, messages: list[dict[str, Any]] | None = None):
        """Replace conversation history with `messages` and return self."""
        self.clear()
        if messages:
            self.extend(messages)
        return self

    def clone(self):
        """Deep clone this conversation, preserving client/model references.

        Message history and structured `last_reply` values are deep-copied to
        avoid shared mutable references between instances.
        """
        cloned = LLMConversation(ai_client=self.ai_client, model=self.model)

        if self.last_reply is None or isinstance(self.last_reply, str):
            cloned.last_reply = self.last_reply
        else:
            cloned.last_reply = json.loads(json.dumps(self.last_reply))

        cloned.assign_messages(self.to_dict_list())
        return cloned

    def submit(
        self,
        message: str | dict[str, Any] | None = None,
        role: str | None = None,
        model: str | None = None,
        json_response: bool | dict[str, Any] | str | None = None,
        shotgun: int | None = None,
        warning_callback: Callable[[str], None] | None = None,
    ) -> str | dict[str, Any] | list[Any] | int | float | bool | None:
        """Submit this conversation and append the assistant reply.

        Optionally appends a message first, resolves an effective model from
        per-call override -> conversation default -> provider smart default,
        then delegates to `llm_submit`.

        Args:
            message: Optional pre-submit message string or role/content object.
            role: Optional role override for the appended message.
            model: Optional per-call model override.
            json_response: JSON mode toggle or structured schema config.
            shotgun: Parallel worker count when using shotgunning.
            warning_callback: Optional callback for retry/non-fatal warnings.

        Returns:
            The assistant reply as text or parsed JSON value, depending on
            `json_response` mode.

        Raises:
            ValueError: If `ai_client` is not set.
        """
        if self.ai_client is None:
            raise ValueError("AI client is not set. Please provide an AI client.")

        if message:
            message_obj: dict[str, Any] = {
                "role": "user",
                "content": "",
            }
            if isinstance(message, str):
                message_obj["content"] = message
            else:
                message_obj.update(message)

            if role is not None:
                message_obj["role"] = role

            self.add_message(message_obj["role"], message_obj["content"])

        selected_model = (
            model
            or self.model
            or get_model_name(self.llm_provider or "openai", "smart")
        )

        llm_reply = llm_submit(
            messages=self.to_dict_list(),
            ai_client=self.ai_client,
            model=selected_model,
            json_response=json_response,
            shotgun=shotgun,
            warning_callback=warning_callback,
        )

        self.add_assistant_message(llm_reply)
        return llm_reply

    def add_message(self, role: str, content: Any):
        """Append a message with role-aware content normalization.

        Strings are kept as-is, lists are preserved (multimodal payloads), dicts
        are pretty-JSON serialized, and all other values are coerced to string.
        """
        normalized_content: str | list[Any]
        if isinstance(content, str):
            normalized_content = content
        elif isinstance(content, list):
            normalized_content = content
        elif isinstance(content, dict):
            normalized_content = json.dumps(content, indent=2)
        else:
            normalized_content = f"{content}"

        self.append({"role": role, "content": normalized_content})
        return self

    def add_user_message(self, content: Any):
        """Append a `user` message and return self."""
        return self.add_message("user", content)

    def add_assistant_message(self, content: Any):
        """Append an `assistant` message, update `last_reply`, and return self."""
        self.last_reply = content
        return self.add_message("assistant", content)

    def add_system_message(self, content: Any):
        """Append a `system` message and return self."""
        return self.add_message("system", content)

    def add_developer_message(self, content: Any):
        """Append a `developer` message and return self."""
        return self.add_message("developer", content)

    def add_image(self, role: str, text: str, image_data_url: str):
        """Append a multimodal text+image message for the active provider.

        OpenAI receives `input_text` and `input_image` content blocks.
        Anthropic receives `image` + `text` blocks and accepts either an HTTP(S)
        URL or a base64 data URL for the image source.
        """
        llm_provider = self.llm_provider
        if not llm_provider:
            raise ValueError(
                "LLM provider cannot be identified. Please set an AI client with a supported provider."
            )

        if llm_provider == "openai" or llm_provider == "deepseek":
            self.append(
                {
                    "role": role,
                    "content": [
                        {
                            "type": "input_text",
                            "text": text,
                        },
                        {
                            "type": "input_image",
                            "image_url": image_data_url,
                            "detail": "high",
                        },
                    ],
                }
            )
            return self

        if llm_provider == "anthropic":
            trimmed_image_data_url = image_data_url.strip()
            image_source: dict[str, str]

            if re.match(r"^https?://", trimmed_image_data_url, re.IGNORECASE):
                image_source = {
                    "type": "url",
                    "url": trimmed_image_data_url,
                }
            else:
                data_url_match = re.match(
                    r"^data:(image\/[a-zA-Z0-9.+-]+);base64,([\s\S]+)$",
                    trimmed_image_data_url,
                    re.IGNORECASE,
                )
                if not data_url_match:
                    raise ValueError(
                        "Anthropic image input must be an HTTP(S) URL or a base64 data URL, such as data:image/png;base64,..."
                    )

                media_type = data_url_match.group(1).lower()
                if media_type == "image/jpg":
                    media_type = "image/jpeg"

                if media_type not in [
                    "image/jpeg",
                    "image/png",
                    "image/gif",
                    "image/webp",
                ]:
                    raise ValueError(
                        "Unsupported Anthropic image media type: "
                        + f"{media_type}. Supported types: image/jpeg, image/png, image/gif, image/webp."
                    )

                image_source = {
                    "type": "base64",
                    "media_type": media_type,
                    "data": re.sub(r"\s+", "", data_url_match.group(2)),
                }

            self.append(
                {
                    "role": role,
                    "content": [
                        {
                            "type": "image",
                            "source": image_source,
                        },
                        {
                            "type": "text",
                            "text": text,
                        },
                    ],
                }
            )
            return self

        raise ValueError(
            f"LLM provider {llm_provider} does not support image messages."
        )

    def submit_message(self, role: str, content: Any):
        """Append a message with `role`, then submit the conversation."""
        self.add_message(role, content)
        return self.submit()

    def submit_user_message(self, content: Any):
        """Append a `user` message, then submit the conversation."""
        self.add_user_message(content)
        return self.submit()

    def submit_assistant_message(self, content: Any):
        """Append an `assistant` message, then submit the conversation."""
        self.add_assistant_message(content)
        return self.submit()

    def submit_system_message(self, content: Any):
        """Append a `system` message, then submit the conversation."""
        self.add_system_message(content)
        return self.submit()

    def submit_developer_message(self, content: Any):
        """Append a `developer` message, then submit the conversation."""
        self.add_developer_message(content)
        return self.submit()

    def submit_image(self, role: str, text: str, image_data_url: str):
        """Append an image message, then submit the conversation."""
        self.add_image(role, text, image_data_url)
        return self.submit()

    def get_last_message(self):
        """Return the last message dict, or None when history is empty."""
        return self[-1] if self else None

    def get_messages_by_role(self, role: str):
        """Return all messages whose `role` exactly matches the input."""
        return [message for message in self if message.get("role") == role]

    def get_last_reply_str(self) -> str:
        """Return `last_reply` when it is a string; otherwise return empty string."""
        return self.last_reply if isinstance(self.last_reply, str) else ""

    def get_last_reply_dict(self) -> dict[str, Any]:
        """Return a deep-cloned dict version of `last_reply`, or `{}`.

        Non-dict values (including lists/primitives) return an empty dict.
        """
        try:
            cloned_reply = json.loads(json.dumps(self.last_reply))
            if isinstance(cloned_reply, dict):
                return cloned_reply
            return {}
        except Exception:
            return {}

    def get_last_reply_dict_field(self, field_name: str, default: Any = None):
        """Return one field from `last_reply` dict, or `default` when missing."""
        try:
            last_reply_dict = self.get_last_reply_dict()
            return last_reply_dict.get(field_name, default)
        except Exception:
            return default

    def to_dict_list(self) -> list[dict[str, Any]]:
        """Return a deep-copied plain list representation of message history."""
        return json.loads(json.dumps(list(self)))
