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


class FakeUsage:
    def __init__(self, input_tokens: int = 0, output_tokens: int = 0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeOpenAIResponse:
    def __init__(
        self,
        output_text: Any = "",
        error: Any = None,
        incomplete_details: Any = None,
        usage: Any = None,
    ):
        self.output_text = output_text
        self.error = error
        self.incomplete_details = incomplete_details
        self.usage = usage


class FakeOpenAIResponsesAPI:
    def __init__(self, side_effects: list[Any] | None = None):
        self.side_effects = list(side_effects or [])
        self.create_calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any):
        self.create_calls.append(kwargs)

        if not self.side_effects:
            return FakeOpenAIResponse()

        next_effect = self.side_effects.pop(0)
        if isinstance(next_effect, BaseException):
            raise next_effect

        return next_effect


class FakeOpenAIClient:
    def __init__(self, side_effects: list[Any] | None = None):
        self.responses = FakeOpenAIResponsesAPI(side_effects)
        self.token_usage: dict[str, Any] = {
            "all_models": {"total": 0, "input": 0, "output": 0},
            "by_model": {},
        }


class OpenAIError(Exception):
    pass


class BadRequestError(Exception):
    pass


class TestGPTLlmSubmit(unittest.TestCase):
    def test_uses_default_model_and_omits_text_config_when_json_mode_disabled(self):
        client = FakeOpenAIClient([FakeOpenAIResponse("ok")])

        result = llm_submit(
            messages=[{"role": "user", "content": "Hello"}],
            ai_client=client,
        )

        self.assertEqual(result, "ok")
        self.assertEqual(len(client.responses.create_calls), 1)
        request = client.responses.create_calls[0]
        self.assertEqual(request.get("model"), "gpt-4.1")
        self.assertNotIn("text", request)

    def test_prepends_datetime_system_message_and_keeps_user_messages_after_it(self):
        client = FakeOpenAIClient([FakeOpenAIResponse("ok")])
        messages = [{"role": "user", "content": "Hello"}]

        llm_submit(messages=messages, ai_client=client)

        submitted = client.responses.create_calls[0]["input"]
        self.assertEqual(submitted[0]["role"], "system")
        self.assertTrue(str(submitted[0]["content"]).startswith("!DATETIME:"))
        self.assertEqual(submitted[1:], messages)

    def test_replaces_stale_datetime_messages_and_keeps_other_system_messages(self):
        client = FakeOpenAIClient([FakeOpenAIResponse("ok")])
        messages = [
            {"role": "system", "content": "!DATETIME: old timestamp"},
            {"role": "system", "content": "keep me"},
            {"role": "user", "content": "hello"},
        ]

        llm_submit(messages=messages, ai_client=client)

        submitted = client.responses.create_calls[0]["input"]
        datetime_messages = [
            m
            for m in submitted
            if m.get("role") == "system"
            and isinstance(m.get("content"), str)
            and m["content"].startswith("!DATETIME:")
        ]
        self.assertEqual(len(datetime_messages), 1)
        self.assertEqual(submitted[1:], messages[1:])

    def test_supports_json_response_true_with_json_object_text_format(self):
        client = FakeOpenAIClient([FakeOpenAIResponse('{"value":1}')])

        result = llm_submit(
            messages=[{"role": "user", "content": "json"}],
            ai_client=client,
            json_response=True,
        )

        self.assertEqual(result, {"value": 1})
        request = client.responses.create_calls[0]
        self.assertEqual(request.get("text"), {"format": {"type": "json_object"}})

    def test_tracks_token_usage_on_client(self):
        client = FakeOpenAIClient([FakeOpenAIResponse("ok", usage=FakeUsage(4, 6))])

        llm_submit(messages=[{"role": "user", "content": "hello"}], ai_client=client)

        self.assertEqual(client.token_usage["all_models"]["total"], 10)
        self.assertEqual(client.token_usage["all_models"]["input"], 4)
        self.assertEqual(client.token_usage["all_models"]["output"], 6)
        self.assertEqual(client.token_usage["by_model"]["gpt-4.1"]["total"], 10)
        self.assertEqual(client.token_usage["by_model"]["gpt-4.1"]["input"], 4)
        self.assertEqual(client.token_usage["by_model"]["gpt-4.1"]["output"], 6)

    def test_finds_and_parses_json_even_with_leading_and_trailing_cruft(self):
        client = FakeOpenAIClient(
            [
                FakeOpenAIResponse(
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
        client = FakeOpenAIClient([FakeOpenAIResponse('{"first":1}{"second":2}')])

        result = llm_submit(
            messages=[{"role": "user", "content": "json"}],
            ai_client=client,
            json_response=True,
        )

        self.assertEqual(result, {"first": 1})

    def test_retries_openai_errors_and_succeeds(self):
        warnings: list[str] = []
        client = FakeOpenAIClient(
            [
                OpenAIError("temporary"),
                FakeOpenAIResponse("ok"),
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
        self.assertEqual(len(client.responses.create_calls), 2)
        self.assertEqual(len(warnings), 1)
        self.assertIn("openai", warnings[0].lower())
        self.assertIn("API error", warnings[0])
        self.assertIn("Retrying (attempt 1 of 2)", warnings[0])

    def test_retries_json_decode_errors_and_succeeds(self):
        warnings: list[str] = []
        client = FakeOpenAIClient(
            [
                FakeOpenAIResponse("not json"),
                FakeOpenAIResponse('{"ok":true}'),
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
        self.assertEqual(len(client.responses.create_calls), 2)
        self.assertEqual(len(warnings), 1)
        self.assertIn("JSON decode error", warnings[0])

    def test_throws_bad_request_error_immediately_without_retry(self):
        bad_request_error = BadRequestError(
            "Invalid type for 'input[11].content': expected one of an array of objects or string, but got an object instead."
        )
        client = FakeOpenAIClient([bad_request_error])

        with self.assertRaises(BadRequestError):
            llm_submit(
                messages=[{"role": "user", "content": "hello"}],
                ai_client=client,
                retry_limit=5,
                retry_backoff_time_seconds=30,
            )

        self.assertEqual(len(client.responses.create_calls), 1)

    def test_throws_for_malformed_response_output_text_without_retry(self):
        client = FakeOpenAIClient([FakeOpenAIResponse(None)])

        with self.assertRaises(TypeError):
            llm_submit(
                messages=[{"role": "user", "content": "hello"}],
                ai_client=client,
                retry_limit=5,
            )

        self.assertEqual(len(client.responses.create_calls), 1)

    def test_forwards_json_response_schema_object_as_text_format_parameter(self):
        schema = {
            "format": {
                "type": "json_schema",
                "name": "test_output",
                "schema": {
                    "type": "object",
                    "properties": {"value": {"type": "number"}},
                    "required": ["value"],
                },
            },
        }
        client = FakeOpenAIClient([FakeOpenAIResponse('{"value":42}')])

        result = llm_submit(
            messages=[{"role": "user", "content": "give me a number"}],
            ai_client=client,
            json_response=schema,
        )

        self.assertEqual(result, {"value": 42})
        request = client.responses.create_calls[0]
        self.assertEqual(request.get("text"), schema)

    def test_throws_immediately_if_json_response_object_cannot_be_jsonized(self):
        client = FakeOpenAIClient([FakeOpenAIResponse('{"unused":true}')])

        recursive_object: dict[str, Any] = {"foo": "bar"}
        recursive_object["self"] = recursive_object

        with self.assertRaises(TypeError):
            llm_submit(
                messages=[{"role": "user", "content": "hello"}],
                ai_client=client,
                json_response=recursive_object,
            )

        self.assertEqual(len(client.responses.create_calls), 0)


class TestGPTLlmSubmitShotgun(unittest.TestCase):
    messages = [{"role": "user", "content": "Hello"}]

    @staticmethod
    def make_client() -> FakeOpenAIClient:
        return FakeOpenAIClient()

    def test_api_calls_increase_linearly_with_shotgun_count(self):
        client = self.make_client()
        llm_submit(messages=self.messages, ai_client=client, shotgun=3)
        num_api_calls_with_shotgun3 = len(client.responses.create_calls)

        client = self.make_client()
        llm_submit(messages=self.messages, ai_client=client, shotgun=6)
        num_api_calls_with_shotgun6 = len(client.responses.create_calls)

        client = self.make_client()
        llm_submit(messages=self.messages, ai_client=client, shotgun=9)
        num_api_calls_with_shotgun9 = len(client.responses.create_calls)

        self.assertEqual(
            num_api_calls_with_shotgun9 - num_api_calls_with_shotgun6,
            num_api_calls_with_shotgun6 - num_api_calls_with_shotgun3,
        )

    def test_shotgun_0_is_same_as_no_shotgun(self):
        client = self.make_client()
        llm_submit(messages=self.messages, ai_client=client)
        num_api_calls_with_no_shotgun = len(client.responses.create_calls)

        client = self.make_client()
        llm_submit(messages=self.messages, ai_client=client, shotgun=0)
        num_api_calls_with_shotgun0 = len(client.responses.create_calls)

        self.assertEqual(num_api_calls_with_shotgun0, num_api_calls_with_no_shotgun)

    def test_shotgun_1_is_same_as_no_shotgun(self):
        client = self.make_client()
        llm_submit(messages=self.messages, ai_client=client)
        num_api_calls_with_no_shotgun = len(client.responses.create_calls)

        client = self.make_client()
        llm_submit(messages=self.messages, ai_client=client, shotgun=1)
        num_api_calls_with_shotgun1 = len(client.responses.create_calls)

        self.assertEqual(num_api_calls_with_shotgun1, num_api_calls_with_no_shotgun)

    def test_shotgun_2_makes_more_calls_than_no_shotgun(self):
        client = self.make_client()
        llm_submit(messages=self.messages, ai_client=client)
        num_api_calls_with_no_shotgun = len(client.responses.create_calls)

        client = self.make_client()
        llm_submit(messages=self.messages, ai_client=client, shotgun=2)
        num_api_calls_with_shotgun2 = len(client.responses.create_calls)

        self.assertGreater(num_api_calls_with_shotgun2, num_api_calls_with_no_shotgun)

    def test_shotgun_overhead_not_invoked_when_shotgun_not_used(self):
        client = self.make_client()
        llm_submit(messages=self.messages, ai_client=client)
        num_api_calls_with_no_shotgun = len(client.responses.create_calls)

        client = self.make_client()
        llm_submit(messages=self.messages, ai_client=client, shotgun=3)
        num_api_calls_with_shotgun3 = len(client.responses.create_calls)

        client = self.make_client()
        llm_submit(messages=self.messages, ai_client=client, shotgun=6)
        num_api_calls_with_shotgun6 = len(client.responses.create_calls)

        delta3_from_no_shotgun = (
            num_api_calls_with_shotgun3 - num_api_calls_with_no_shotgun
        )
        delta3_from_shotgun3 = num_api_calls_with_shotgun6 - num_api_calls_with_shotgun3

        self.assertGreater(delta3_from_no_shotgun, delta3_from_shotgun3)


if __name__ == "__main__":
    unittest.main()
