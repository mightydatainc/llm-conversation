import json
import sys
import unittest
from importlib import import_module
from pathlib import Path
from typing import Any

# Ensure local src layout is importable when running:
# python -m unittest discover tests
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

pkg = import_module("mightydatainc_llm_conversation")

llm_submit = getattr(pkg, "llm_submit")


class FakeAnthropicTextBlock:
    def __init__(self, text: Any = ""):
        self.type = "text"
        self.text = text


class FakeUsage:
    def __init__(self, input_tokens: int = 0, output_tokens: int = 0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeAnthropicResponse:
    def __init__(
        self,
        text: Any = "",
        stop_reason: str = "end_turn",
        usage: Any = None,
    ):
        if text is None:
            self.content = None
        else:
            self.content = [FakeAnthropicTextBlock(text)]
        self.stop_reason = stop_reason
        self.usage = usage


class FakeAnthropicMessagesAPI:
    def __init__(self, side_effects: list[Any] | None = None):
        self.side_effects = list(side_effects or [])
        self.create_calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any):
        self.create_calls.append(kwargs)

        if not self.side_effects:
            return FakeAnthropicResponse()

        next_effect = self.side_effects.pop(0)
        if isinstance(next_effect, BaseException):
            raise next_effect

        return next_effect


class FakeAnthropicClient:
    def __init__(self, side_effects: list[Any] | None = None):
        self.messages = FakeAnthropicMessagesAPI(side_effects)
        self.token_usage: dict[str, Any] = {
            "all_models": {"total": 0, "input": 0, "output": 0},
            "by_model": {},
        }


class AnthropicError(Exception):
    pass


class BadRequestError(Exception):
    pass


class TestClaudeLlmSubmit(unittest.TestCase):
    def test_uses_default_model_when_json_mode_is_disabled(self):
        client = FakeAnthropicClient([FakeAnthropicResponse("ok")])

        result = llm_submit(
            messages=[{"role": "user", "content": "Hello"}], ai_client=client
        )

        self.assertEqual(result, "ok")
        self.assertEqual(len(client.messages.create_calls), 1)
        request = client.messages.create_calls[0]
        self.assertEqual(request.get("model"), "claude-opus-4-6")

    def test_passes_datetime_in_system_param_and_keeps_user_messages(self):
        client = FakeAnthropicClient([FakeAnthropicResponse("ok")])
        messages = [{"role": "user", "content": "Hello"}]

        llm_submit(messages=messages, ai_client=client)

        request = client.messages.create_calls[0]
        self.assertIsInstance(request.get("system"), str)
        self.assertTrue(request["system"].startswith("!DATETIME:"))
        self.assertEqual(request.get("messages"), messages)

    def test_replaces_stale_datetime_messages_and_keeps_other_system_messages(self):
        client = FakeAnthropicClient([FakeAnthropicResponse("ok")])
        messages = [
            {"role": "system", "content": "!DATETIME: old timestamp"},
            {"role": "system", "content": "keep me"},
            {"role": "user", "content": "hello"},
        ]

        llm_submit(messages=messages, ai_client=client)

        request = client.messages.create_calls[0]
        system_text = request.get("system", "")
        self.assertEqual(system_text.count("!DATETIME:"), 1)
        self.assertIn("keep me", system_text)
        self.assertEqual(
            request.get("messages"), [{"role": "user", "content": "hello"}]
        )

    def test_supports_json_response_true_and_parses_json(self):
        client = FakeAnthropicClient([FakeAnthropicResponse('{"value":1}')])

        result = llm_submit(
            messages=[{"role": "user", "content": "json"}],
            ai_client=client,
            json_response=True,
        )

        self.assertEqual(result, {"value": 1})

    def test_tracks_token_usage_on_client(self):
        client = FakeAnthropicClient(
            [FakeAnthropicResponse("ok", usage=FakeUsage(3, 7))]
        )

        llm_submit(messages=[{"role": "user", "content": "hello"}], ai_client=client)

        self.assertEqual(client.token_usage["all_models"]["total"], 10)
        self.assertEqual(client.token_usage["all_models"]["input"], 3)
        self.assertEqual(client.token_usage["all_models"]["output"], 7)
        self.assertEqual(client.token_usage["by_model"]["claude-opus-4-6"]["total"], 10)
        self.assertEqual(client.token_usage["by_model"]["claude-opus-4-6"]["input"], 3)
        self.assertEqual(client.token_usage["by_model"]["claude-opus-4-6"]["output"], 7)

    def test_finds_and_parses_json_even_with_leading_and_trailing_cruft(self):
        client = FakeAnthropicClient(
            [
                FakeAnthropicResponse(
                    'Sure! Here is some JSON! {"value":1} Need anything else?'
                ),
            ]
        )

        result = llm_submit(
            messages=[{"role": "user", "content": "json"}],
            ai_client=client,
            json_response=True,
        )

        self.assertEqual(result, {"value": 1})

    def test_parses_first_json_object_when_response_contains_trailing_json(self):
        client = FakeAnthropicClient([FakeAnthropicResponse('{"first":1}{"second":2}')])

        result = llm_submit(
            messages=[{"role": "user", "content": "json"}],
            ai_client=client,
            json_response=True,
        )

        self.assertEqual(result, {"first": 1})

    def test_retries_anthropic_errors_and_succeeds(self):
        warnings: list[str] = []
        client = FakeAnthropicClient(
            [
                AnthropicError("temporary"),
                FakeAnthropicResponse("ok"),
            ]
        )

        result = llm_submit(
            messages=[{"role": "user", "content": "hello"}],
            ai_client=client,
            retry_limit=2,
            retry_backoff_time_seconds=0,
            warning_callback=warnings.append,
        )

        self.assertEqual(result, "ok")
        self.assertEqual(len(client.messages.create_calls), 2)
        self.assertEqual(len(warnings), 1)
        self.assertIn("anthropic", warnings[0].lower())
        self.assertIn("API error", warnings[0])
        self.assertIn("Retrying (attempt 1 of 2)", warnings[0])

    def test_retries_json_decode_errors_and_succeeds(self):
        warnings: list[str] = []
        client = FakeAnthropicClient(
            [
                FakeAnthropicResponse("not json"),
                FakeAnthropicResponse('{"ok":true}'),
            ]
        )

        result = llm_submit(
            messages=[{"role": "user", "content": "hello"}],
            ai_client=client,
            json_response=True,
            retry_limit=2,
            warning_callback=warnings.append,
        )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(len(client.messages.create_calls), 2)
        self.assertEqual(len(warnings), 1)
        self.assertIn("JSON decode error", warnings[0])

    def test_throws_bad_request_error_immediately_without_retry(self):
        bad_request_error = BadRequestError(
            "Invalid type for 'messages': expected array of message objects."
        )
        client = FakeAnthropicClient([bad_request_error])

        with self.assertRaises(BadRequestError):
            llm_submit(
                messages=[{"role": "user", "content": "hello"}],
                ai_client=client,
                retry_limit=5,
                retry_backoff_time_seconds=30,
            )

        self.assertEqual(len(client.messages.create_calls), 1)

    def test_throws_for_malformed_response_content_without_retry(self):
        client = FakeAnthropicClient([FakeAnthropicResponse(None)])

        with self.assertRaises(TypeError):
            llm_submit(
                messages=[{"role": "user", "content": "hello"}],
                ai_client=client,
                retry_limit=5,
            )

        self.assertEqual(len(client.messages.create_calls), 1)

    def test_sends_output_config_when_json_response_is_schema_object(self):
        schema = {
            "format": {
                "type": "json_schema",
                "name": "test_output",
                "description": "A test JSON schema for output formatting",
                "schema": {
                    "type": "object",
                    "properties": {"value": {"type": "number"}},
                    "required": ["value"],
                },
            },
        }
        client = FakeAnthropicClient([FakeAnthropicResponse('{"value":42}')])

        result = llm_submit(
            messages=[{"role": "user", "content": "give me a number"}],
            ai_client=client,
            json_response=schema,
        )

        self.assertEqual(result, {"value": 42})
        request = client.messages.create_calls[0]

        schema_expected = json.loads(json.dumps(schema))
        if "name" in schema_expected["format"]:
            del schema_expected["format"]["name"]
        if "description" in schema_expected["format"]:
            del schema_expected["format"]["description"]

        self.assertEqual(request.get("output_config"), schema_expected)

    def test_throws_immediately_if_json_response_object_cannot_be_jsonized(self):
        client = FakeAnthropicClient([FakeAnthropicResponse('{"unused":true}')])

        recursive_object: dict[str, Any] = {"foo": "bar"}
        recursive_object["self"] = recursive_object

        with self.assertRaises(TypeError):
            llm_submit(
                messages=[{"role": "user", "content": "hello"}],
                ai_client=client,
                json_response=recursive_object,
            )

        self.assertEqual(len(client.messages.create_calls), 0)


class TestClaudeLlmSubmitShotgun(unittest.TestCase):
    messages = [{"role": "user", "content": "Hello"}]

    @staticmethod
    def make_client() -> FakeAnthropicClient:
        return FakeAnthropicClient()

    def test_api_calls_increase_linearly_with_shotgun_count(self):
        client = self.make_client()
        llm_submit(messages=self.messages, ai_client=client, shotgun=3)
        num_api_calls_with_shotgun3 = len(client.messages.create_calls)

        client = self.make_client()
        llm_submit(messages=self.messages, ai_client=client, shotgun=6)
        num_api_calls_with_shotgun6 = len(client.messages.create_calls)

        client = self.make_client()
        llm_submit(messages=self.messages, ai_client=client, shotgun=9)
        num_api_calls_with_shotgun9 = len(client.messages.create_calls)

        self.assertEqual(
            num_api_calls_with_shotgun9 - num_api_calls_with_shotgun6,
            num_api_calls_with_shotgun6 - num_api_calls_with_shotgun3,
        )

    def test_shotgun_0_is_same_as_no_shotgun(self):
        client = self.make_client()
        llm_submit(messages=self.messages, ai_client=client)
        num_api_calls_with_no_shotgun = len(client.messages.create_calls)

        client = self.make_client()
        llm_submit(messages=self.messages, ai_client=client, shotgun=0)
        num_api_calls_with_shotgun0 = len(client.messages.create_calls)

        self.assertEqual(num_api_calls_with_shotgun0, num_api_calls_with_no_shotgun)

    def test_shotgun_1_is_same_as_no_shotgun(self):
        client = self.make_client()
        llm_submit(messages=self.messages, ai_client=client)
        num_api_calls_with_no_shotgun = len(client.messages.create_calls)

        client = self.make_client()
        llm_submit(messages=self.messages, ai_client=client, shotgun=1)
        num_api_calls_with_shotgun1 = len(client.messages.create_calls)

        self.assertEqual(num_api_calls_with_shotgun1, num_api_calls_with_no_shotgun)

    def test_shotgun_2_makes_more_calls_than_no_shotgun(self):
        client = self.make_client()
        llm_submit(messages=self.messages, ai_client=client)
        num_api_calls_with_no_shotgun = len(client.messages.create_calls)

        client = self.make_client()
        llm_submit(messages=self.messages, ai_client=client, shotgun=2)
        num_api_calls_with_shotgun2 = len(client.messages.create_calls)

        self.assertGreater(num_api_calls_with_shotgun2, num_api_calls_with_no_shotgun)

    def test_shotgun_overhead_not_invoked_when_shotgun_not_used(self):
        client = self.make_client()
        llm_submit(messages=self.messages, ai_client=client)
        num_api_calls_with_no_shotgun = len(client.messages.create_calls)

        client = self.make_client()
        llm_submit(messages=self.messages, ai_client=client, shotgun=3)
        num_api_calls_with_shotgun3 = len(client.messages.create_calls)

        client = self.make_client()
        llm_submit(messages=self.messages, ai_client=client, shotgun=6)
        num_api_calls_with_shotgun6 = len(client.messages.create_calls)

        delta3_from_no_shotgun = (
            num_api_calls_with_shotgun3 - num_api_calls_with_no_shotgun
        )
        delta3_from_shotgun3 = num_api_calls_with_shotgun6 - num_api_calls_with_shotgun3

        self.assertGreater(delta3_from_no_shotgun, delta3_from_shotgun3)


if __name__ == "__main__":
    unittest.main()
