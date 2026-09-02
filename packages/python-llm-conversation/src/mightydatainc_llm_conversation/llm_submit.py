import json
import time
from typing import Any, Callable

from .helpers import current_datetime_system_message, parse_first_json_value
from .llm_providers import (
    LLM_RETRY_BACKOFF_TIME_SECONDS_DEFAULT,
    LLM_RETRY_LIMIT_DEFAULT,
    get_model_name,
    identify_llm_provider,
)
from .llm_submit_shotgun import llm_submit_shotgun


def _update_model_token_usage(
    ai_client: Any,
    model_name: str,
    token_count_input: int,
    token_count_output: int,
) -> None:
    if (
        not hasattr(ai_client, "token_usage")
        or getattr(ai_client, "token_usage") is None
    ):
        ai_client.token_usage = {
            "all_models": {"total": 0, "input": 0, "output": 0},
            "by_model": {},
        }

    ai_client.token_usage["all_models"]["total"] += (
        token_count_input + token_count_output
    )
    ai_client.token_usage["all_models"]["input"] += token_count_input
    ai_client.token_usage["all_models"]["output"] += token_count_output

    if model_name not in ai_client.token_usage["by_model"]:
        ai_client.token_usage["by_model"][model_name] = {
            "total": 0,
            "input": 0,
            "output": 0,
        }

    ai_client.token_usage["by_model"][model_name]["total"] += (
        token_count_input + token_count_output
    )
    ai_client.token_usage["by_model"][model_name]["input"] += token_count_input
    ai_client.token_usage["by_model"][model_name]["output"] += token_count_output


def is_retryable_openai_error(error: Exception) -> bool:
    """Detect retryable OpenAI-style errors by class name."""
    name = error.__class__.__name__ or ""
    return "OpenAI" in name or "APIError" in name


def is_retryable_anthropic_error(error: Exception) -> bool:
    """Detect retryable Anthropic-style errors by class name."""
    name = error.__class__.__name__ or ""
    return "Anthropic" in name or "APIError" in name


# Prompt to prohibit unicode characters in model output.
# Currently not used, but may be useful in the future.
PROMPT_UNICODE_PROHIBITION = (
    "ABSOLUTELY NO UNICODE ALLOWED.\n"
    "Only use typeable keyboard characters. Do not try to circumvent this rule\n"
    "with escape sequences, backslashes, or other tricks. Use double dashes (--)\n"
    "instead of em-dashes or en-dashes; use straight quotes (\") and single quotes (')\n"
    " instead of curly versions, use hyphens instead of bullets, etc."
)


def llm_submit(
    messages: list[Any],
    ai_client: Any,
    model: str | None = None,
    json_response: bool | dict[str, Any] | str | None = None,
    system_announcement_message: str | None = None,
    retry_limit: int | None = None,
    retry_backoff_time_seconds: int | float | None = None,
    shotgun: int | None = None,
    warning_callback: Callable[[str], None] | None = None,
) -> str | dict[str, Any] | list[Any] | int | float | bool | None:
    """Submit messages to the detected provider with retry/json/shotgun support."""
    options: dict[str, Any] = {
        "model": model,
        "json_response": json_response,
        "system_announcement_message": system_announcement_message,
        "retry_limit": retry_limit,
        "retry_backoff_time_seconds": retry_backoff_time_seconds,
        "shotgun": shotgun,
        "warning_callback": warning_callback,
    }

    if options["shotgun"] and options["shotgun"] > 1:
        return llm_submit_shotgun(messages, ai_client, options, options["shotgun"])

    llm_provider_name = identify_llm_provider(ai_client)

    model = options["model"] or get_model_name(llm_provider_name, "smart")
    retry_limit_int: int = (
        retry_limit if retry_limit is not None else LLM_RETRY_LIMIT_DEFAULT
    )
    retry_backoff_seconds_float: float = float(
        retry_backoff_time_seconds
        if retry_backoff_time_seconds is not None
        else LLM_RETRY_BACKOFF_TIME_SECONDS_DEFAULT
    )

    system_announcement = f"{(options['system_announcement_message'] or '').strip()}"

    failed_error: Exception | None = None

    # Clone and prepare a request payload without mutating the original list.
    prepared_messages = json.loads(json.dumps(messages))
    prepared_messages = [
        message
        for message in prepared_messages
        if not (
            message.get("role") == "system"
            and f"{message.get('content')}".startswith("!DATETIME:")
        )
    ]

    # Always prepend a fresh datetime system message for temporal context.
    prepared_messages.insert(0, current_datetime_system_message())

    # Optional caller-provided system announcement takes absolute first position.
    if system_announcement:
        prepared_messages.insert(0, {"role": "system", "content": system_announcement})

    for num_try in range(retry_limit_int + 1):
        llm_reply = ""

        try:
            if llm_provider_name == "openai" or llm_provider_name == "deepseek":
                payload_body: dict[str, Any] = {
                    "model": model,
                    "input": prepared_messages,
                }
                if options["json_response"]:
                    if isinstance(options["json_response"], bool):
                        payload_body["text"] = {"format": {"type": "json_object"}}
                    else:
                        try:
                            payload_body["text"] = json.loads(
                                json.dumps(options["json_response"])
                            )
                        except Exception as error:
                            raise TypeError(str(error)) from error

                llm_response = ai_client.responses.create(**payload_body)

                if not isinstance(getattr(llm_response, "output_text", None), str):
                    raise TypeError("OpenAI API response output_text must be a string")

                usage = getattr(llm_response, "usage", None)
                if usage is not None:
                    _update_model_token_usage(
                        ai_client,
                        model or "",
                        int(getattr(usage, "input_tokens", 0) or 0),
                        int(getattr(usage, "output_tokens", 0) or 0),
                    )

                llm_reply = llm_response.output_text.strip()

                if options["warning_callback"]:
                    if getattr(llm_response, "error", None):
                        options["warning_callback"](
                            f"ERROR: OpenAI API returned an error: {llm_response.error}"
                        )
                    if getattr(llm_response, "incomplete_details", None):
                        options["warning_callback"](
                            "ERROR: OpenAI API returned incomplete details: "
                            + f"{llm_response.incomplete_details}"
                        )

            elif llm_provider_name == "anthropic":
                # Anthropic receives system content via a separate `system` field.
                anthropic_system_prompt = ""
                while len(prepared_messages) > 0:
                    first_msg = prepared_messages[0]
                    if first_msg.get("role") not in ["system", "developer"]:
                        break
                    anthropic_system_prompt += f"{first_msg.get('content')}\n\n"
                    prepared_messages.pop(0)

                anthropic_system_prompt = anthropic_system_prompt.strip()

                payload_body: dict[str, Any] = {
                    "model": model,
                    "max_tokens": 16384,
                    "messages": prepared_messages,
                }
                if anthropic_system_prompt:
                    payload_body["system"] = anthropic_system_prompt

                if options["json_response"]:
                    if isinstance(options["json_response"], bool):
                        # Encourage raw JSON-only response in freeform JSON mode.
                        payload_body["messages"].append(
                            {
                                "role": "user",
                                "content": (
                                    "Respond with a JSON object. Do not include any text before or after the JSON. "
                                    "The JSON should be the only content in your response, and it must be properly "
                                    "formatted with opening and closing curly braces. Do not put the JSON inside a "
                                    "code block or use any other formatting -- just the raw JSON, starting with an "
                                    "opening curly brace."
                                ),
                            }
                        )
                    else:
                        try:
                            payload_body["output_config"] = json.loads(
                                json.dumps(options["json_response"])
                            )
                        except Exception as error:
                            raise TypeError(str(error)) from error

                        payload_body["output_config"]["format"] = {
                            "type": "json_schema",
                            "schema": payload_body["output_config"]["format"]["schema"],
                        }

                llm_response = ai_client.messages.create(**payload_body)

                usage = getattr(llm_response, "usage", None)
                if usage is not None:
                    _update_model_token_usage(
                        ai_client,
                        model or "",
                        int(getattr(usage, "input_tokens", 0) or 0),
                        int(getattr(usage, "output_tokens", 0) or 0),
                    )

                text_blocks = [
                    block
                    for block in llm_response.content
                    if getattr(block, "type", None) == "text"
                ]
                llm_reply = "".join(
                    [f"{getattr(block, 'text', '')}" for block in text_blocks]
                ).strip()

                if (
                    options["warning_callback"]
                    and getattr(llm_response, "stop_reason", None) == "max_tokens"
                ):
                    options["warning_callback"](
                        "ERROR: Anthropic API response was truncated (max_tokens reached)"
                    )
            else:
                raise ValueError(f"Unsupported LLM provider: {llm_provider_name}")

            if not options["json_response"]:
                return f"{llm_reply}"

            return parse_first_json_value(llm_reply)

        except Exception as error:
            if isinstance(error, SyntaxError):
                failed_error = error
                if options["warning_callback"]:
                    options["warning_callback"](
                        "JSON decode error:\n\n"
                        + f"{error}.\n\n"
                        + f"Raw text of LLM Reply:\n{llm_reply}\n\n"
                        + f"Retrying (attempt {num_try + 1} of {retry_limit_int}) immediately..."
                    )
                continue

            if is_retryable_openai_error(error) or is_retryable_anthropic_error(error):
                failed_error = error
                if options["warning_callback"]:
                    options["warning_callback"](
                        f"LLM Provider ({llm_provider_name}) API error:\n\n{error}.\n\n"
                        + f"Retrying (attempt {num_try + 1} of {retry_limit_int}) in {retry_backoff_seconds_float} seconds..."
                    )

                # Sleep before next retry attempt.
                time.sleep(retry_backoff_seconds_float)
                continue

            raise

    if failed_error:
        raise failed_error

    raise RuntimeError("Unknown error occurred in llmSubmit")
