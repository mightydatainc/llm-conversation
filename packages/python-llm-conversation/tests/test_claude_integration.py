import os
import sys
import unittest
import base64
from importlib import import_module
from pathlib import Path
from typing import Any

# Ensure local src layout is importable when running:
# python -m unittest discover tests
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Load environment variables from package-local .env when available.
dotenv_module = import_module("dotenv")
dotenv_module.load_dotenv(Path(__file__).resolve().parents[1] / ".env")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
if not ANTHROPIC_API_KEY:
    raise RuntimeError(
        "ANTHROPIC_API_KEY is required for live API tests. Configure your test environment to provide it."
    )

pkg = import_module("mightydatainc_llm_conversation")
providers = import_module("mightydatainc_llm_conversation.llm_providers")

LLMConversation = getattr(pkg, "LLMConversation")
JSONSchemaFormat = getattr(pkg, "JSONSchemaFormat")
get_model_name = getattr(providers, "get_model_name")


def create_client() -> Any:
    anthropic_module = import_module("anthropic")
    return anthropic_module.Anthropic(api_key=ANTHROPIC_API_KEY)


IMAGE_IDENTIFICATION_SCHEMA = JSONSchemaFormat(
    {
        "image_subject_enum": [
            "house",
            "chair",
            "boat",
            "car",
            "cat",
            "dog",
            "telephone",
            "duck",
            "city_skyline",
            "still_life",
            "bed",
            "headphones",
            "skull",
            "photo_camera",
            "unknown",
            "none",
            "error",
        ],
    },
    "ImageIdentification",
    "A test schema for image identification response",
)


class TestClaudeIntegration(unittest.TestCase):
    def test_should_repeat_hello_world(self):
        anthropic_client = create_client()
        convo = LLMConversation(ai_client=anthropic_client)

        convo.add_user_message("""
This is a test to see if I'm correctly calling the Anthropic API to invoke Claude.
If you can see this, please respond with "Hello World" -- just like that,
with no additional text or explanation. Do not include punctuation or quotation
marks. Emit only the words "Hello World", capitalized as shown.
""")
        convo.submit()

        reply = convo.get_last_reply_str()
        self.assertEqual(reply, "Hello World")

    def test_should_invoke_llm_with_nominal_intelligence(self):
        anthropic_client = create_client()
        convo = LLMConversation(ai_client=anthropic_client)

        # Test the submit_user_message convenience method.
        convo.submit_user_message("""
I'm conducting a test of my REST API response parsing systems.
If you can see this, please reply with the capital of France.
Reply only with the name of the city, with no additional text, punctuation,
or explanation. I'll be comparing your output string to a standard known
value, so it's important to the integrity of my system that the only
response you produce be just the name of the city. Standard capitalization
please -- first letter capitalized, all other letters lower-case.
""")

        reply = convo.get_last_reply_str()
        self.assertEqual(reply, "Paris")

    def test_should_reply_with_general_form_json_object(self):
        anthropic_client = create_client()
        convo = LLMConversation(ai_client=anthropic_client)

        convo.add_user_message("""
This is a test to see if I'm correctly calling the Anthropic API to invoke Claude.

Please reply with the following JSON object, exactly as shown:

{
  "text": "Hello World",
  "success": true,
  "sample_array_data": [1, 2, {"nested_key": "nested_value"}]
}
""")
        convo.submit(json_response=True)

        reply_obj = convo.get_last_reply_dict()

        self.assertEqual(reply_obj.get("text"), "Hello World")
        self.assertEqual(reply_obj.get("success"), True)

        sample_array_data = reply_obj.get("sample_array_data")
        self.assertIsInstance(sample_array_data, list)
        sample_array_data = sample_array_data or []
        self.assertEqual(len(sample_array_data), 3)
        self.assertEqual(sample_array_data[0], 1)
        self.assertEqual(sample_array_data[1], 2)

        self.assertIsInstance(sample_array_data[2], dict)
        self.assertEqual(sample_array_data[2].get("nested_key"), "nested_value")

        self.assertEqual(convo.get_last_reply_dict_field("text"), "Hello World")
        self.assertEqual(convo.get_last_reply_dict_field("success"), True)
        self.assertEqual(len(convo.get_last_reply_dict_field("sample_array_data")), 3)

    def test_should_reply_with_structured_json_using_json_schema_spec(self):
        anthropic_client = create_client()
        convo = LLMConversation(ai_client=anthropic_client)

        schema = {
            "format": {
                "type": "json_schema",
                "strict": True,
                "name": "TestSchema",
                "description": "A test schema for structured JSON response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "success": {"type": "boolean"},
                        "sample_array_data": {
                            "type": "array",
                            "items": {"type": "number"},
                        },
                        "nested_dict": {
                            "type": "object",
                            "properties": {
                                "nested_key": {"type": "string"},
                            },
                            "required": ["nested_key"],
                            "additionalProperties": False,
                        },
                    },
                    "required": [
                        "text",
                        "success",
                        "sample_array_data",
                        "nested_dict",
                    ],
                    "additionalProperties": False,
                },
            }
        }

        convo.add_user_message("""
Please reply with a JSON object that contains the following data:

Success flag: true
Text: "Hello World"
Sample array data (2 elements long):
    Element 0: 5
    Element 1: 33
Nested dict (1 item long):
    Value under "nested_key": "foobar"
""")
        convo.submit(json_response=schema)

        self.assertEqual(convo.get_last_reply_dict_field("success"), True)
        self.assertEqual(convo.get_last_reply_dict_field("text"), "Hello World")

        nested_dict = convo.get_last_reply_dict_field("nested_dict")
        self.assertIsInstance(nested_dict, dict)
        nested_dict = nested_dict or {}
        self.assertEqual(nested_dict.get("nested_key"), "foobar")
        self.assertEqual(len(nested_dict.keys()), 1)

        sample_array_data = convo.get_last_reply_dict_field("sample_array_data")
        self.assertIsInstance(sample_array_data, list)
        sample_array_data = sample_array_data or []
        self.assertEqual(len(sample_array_data), 2)
        self.assertEqual(sample_array_data[0], 5)
        self.assertEqual(sample_array_data[1], 33)

    def test_should_reply_with_structured_json_using_json_formatter_shorthand(self):
        anthropic_client = create_client()
        convo = LLMConversation(ai_client=anthropic_client)

        schema = JSONSchemaFormat(
            {
                "text": str,
                "success": bool,
                "sample_array_data": [int],
                "nested_dict": {
                    "nested_key": str,
                },
            },
            "TestSchema",
            "A test schema for structured JSON response",
        )

        convo.add_user_message("""
Please reply with a JSON object that contains the following data:

Success flag: true
Text: "Hello World"
Sample array data (2 elements long):
    Element 0: 5
    Element 1: 33
Nested dict (1 item long):
    Value under "nested_key": "foobar"
""")
        convo.submit(json_response=schema)

        self.assertEqual(convo.get_last_reply_dict_field("success"), True)
        self.assertEqual(convo.get_last_reply_dict_field("text"), "Hello World")

        nested_dict = convo.get_last_reply_dict_field("nested_dict")
        self.assertIsInstance(nested_dict, dict)
        nested_dict = nested_dict or {}
        self.assertEqual(nested_dict.get("nested_key"), "foobar")
        self.assertEqual(len(nested_dict.keys()), 1)

        sample_array_data = convo.get_last_reply_dict_field("sample_array_data")
        self.assertIsInstance(sample_array_data, list)
        sample_array_data = sample_array_data or []
        self.assertEqual(len(sample_array_data), 2)
        self.assertEqual(sample_array_data[0], 5)
        self.assertEqual(sample_array_data[1], 33)

    def test_should_perform_image_recognition_with_manual_content_message(self):
        anthropic_client = create_client()
        convo = LLMConversation(
            ai_client=anthropic_client,
            model=get_model_name("anthropic", "vision"),
        )

        # Load the image ./fixtures/phoenix.png
        fixture_path = Path(__file__).resolve().parent / "fixtures" / "phoenix.png"
        img_base64 = base64.b64encode(fixture_path.read_bytes()).decode("ascii")

        claude_msg_with_image = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_base64,
                    },
                },
                {
                    "type": "text",
                    "text": "An image submitted by a user, needing identification",
                },
            ],
        }

        # Build the multimodal message directly instead of using helper methods.
        convo.append(claude_msg_with_image)
        convo.add_user_message("What is this a picture of?")

        convo.submit(json_response=IMAGE_IDENTIFICATION_SCHEMA)

        self.assertEqual(convo.get_last_reply_dict_field("image_subject_enum"), "cat")

    def test_should_perform_image_recognition_with_convenience_methods(self):
        anthropic_client = create_client()
        convo = LLMConversation(
            ai_client=anthropic_client,
            model=get_model_name("anthropic", "vision"),
        )

        # Load the image ./fixtures/phoenix.png
        fixture_path = Path(__file__).resolve().parent / "fixtures" / "phoenix.png"
        img_base64 = base64.b64encode(fixture_path.read_bytes()).decode("ascii")
        img_data_url = f"data:image/png;base64,{img_base64}"

        convo.add_image(
            "user",
            "An image submitted by a user, needing identification",
            img_data_url,
        )
        convo.add_user_message("What is this a picture of?")

        convo.submit(json_response=IMAGE_IDENTIFICATION_SCHEMA)

        self.assertEqual(convo.get_last_reply_dict_field("image_subject_enum"), "cat")

    def test_should_track_token_usage(self):
        anthropic_client = create_client()
        convo = LLMConversation(ai_client=anthropic_client)

        convo.add_user_message(
            "Write a 500-word story about a raccoon that steals the Mona Lisa."
        )
        convo.submit()

        token_usage = anthropic_client.token_usage
        self.assertIsNotNone(token_usage)
        self.assertGreater(token_usage["all_models"]["input"], 35)
        self.assertLess(token_usage["all_models"]["input"], 65)
        self.assertGreater(token_usage["all_models"]["output"], 550)
        self.assertLess(token_usage["all_models"]["output"], 775)
        self.assertEqual(
            token_usage["all_models"]["total"],
            token_usage["all_models"]["input"] + token_usage["all_models"]["output"],
        )

        self.assertEqual(len(token_usage["by_model"]), 1)
        model_name = next(iter(token_usage["by_model"].keys()))
        model_usage = token_usage["by_model"][model_name]
        self.assertEqual(model_usage["input"], token_usage["all_models"]["input"])
        self.assertEqual(model_usage["output"], token_usage["all_models"]["output"])
        self.assertEqual(model_usage["total"], token_usage["all_models"]["total"])

        convo.add_user_message(
            "Now write a 500-word story about that raccoon's arch-rival, a fox detective."
        )
        convo.submit()

        token_usage = anthropic_client.token_usage
        self.assertIsNotNone(token_usage)
        self.assertGreater(token_usage["all_models"]["input"], 700)
        self.assertLess(token_usage["all_models"]["input"], 900)
        self.assertGreater(token_usage["all_models"]["output"], 1300)
        self.assertLess(token_usage["all_models"]["output"], 1600)
        self.assertEqual(
            token_usage["all_models"]["total"],
            token_usage["all_models"]["input"] + token_usage["all_models"]["output"],
        )

        self.assertEqual(len(token_usage["by_model"]), 1)
        model_name = next(iter(token_usage["by_model"].keys()))
        model_usage = token_usage["by_model"][model_name]
        self.assertEqual(model_usage["input"], token_usage["all_models"]["input"])
        self.assertEqual(model_usage["output"], token_usage["all_models"]["output"])
        self.assertEqual(model_usage["total"], token_usage["all_models"]["total"])


if __name__ == "__main__":
    unittest.main()
