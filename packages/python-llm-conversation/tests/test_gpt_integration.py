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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is required for live API tests. Configure your test environment to provide it."
    )

pkg = import_module("mightydatainc_llm_conversation")
providers = import_module("mightydatainc_llm_conversation.llm_providers")

LLMConversation = getattr(pkg, "LLMConversation")
JSONSchemaFormat = getattr(pkg, "JSONSchemaFormat")
get_model_name = getattr(providers, "get_model_name")


def create_client() -> Any:
    openai_module = import_module("openai")
    return openai_module.OpenAI(api_key=OPENAI_API_KEY)


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


class TestGPTIntegration(unittest.TestCase):
    def test_should_repeat_hello_world(self):
        openai_client = create_client()
        convo = LLMConversation(ai_client=openai_client)

        convo.add_user_message("""
This is a test to see if I'm correctly calling the OpenAI API to invoke GPT.
If you can see this, please respond with "Hello World" -- just like that,
with no additional text or explanation. Do not include punctuation or quotation
marks. Emit only the words "Hello World", capitalized as shown.
""")
        convo.submit()

        reply = convo.get_last_reply_str()
        self.assertEqual(reply, "Hello World")

    def test_should_invoke_llm_with_nominal_intelligence(self):
        openai_client = create_client()
        convo = LLMConversation(ai_client=openai_client)

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
        openai_client = create_client()
        convo = LLMConversation(ai_client=openai_client)

        convo.add_user_message("""
This is a test to see if I'm correctly calling the OpenAI API to invoke GPT.

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
        openai_client = create_client()
        convo = LLMConversation(ai_client=openai_client)

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
        openai_client = create_client()
        convo = LLMConversation(ai_client=openai_client)

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
        openai_client = create_client()
        convo = LLMConversation(
            ai_client=openai_client,
            model=get_model_name("openai", "vision"),
        )

        # Load the image ./fixtures/phoenix.png
        fixture_path = Path(__file__).resolve().parent / "fixtures" / "phoenix.png"
        img_base64 = base64.b64encode(fixture_path.read_bytes()).decode("ascii")
        img_data_url = f"data:image/png;base64,{img_base64}"

        gpt_msg_with_image = {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "An image submitted by a user, needing identification",
                },
                {
                    "type": "input_image",
                    "image_url": img_data_url,
                    "detail": "high",
                },
            ],
        }

        # Build the multimodal message directly instead of using helper methods.
        convo.append(gpt_msg_with_image)
        convo.add_user_message("What is this a picture of?")

        convo.submit(json_response=IMAGE_IDENTIFICATION_SCHEMA)

        self.assertEqual(convo.get_last_reply_dict_field("image_subject_enum"), "cat")

    def test_should_perform_image_recognition_with_convenience_methods(self):
        openai_client = create_client()
        convo = LLMConversation(
            ai_client=openai_client,
            model=get_model_name("openai", "vision"),
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
        openai_client = create_client()
        convo = LLMConversation(ai_client=openai_client)

        convo.add_user_message(
            "Write a 500-word story about a raccoon that steals the Mona Lisa."
        )
        convo.submit()

        token_usage = openai_client.token_usage
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

        token_usage = openai_client.token_usage
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

    def test_should_use_shotgun_for_reliability_on_unreliable_question(self):
        # Large barrel count is intentional to make this live test less flaky.
        NUM_SHOTGUN_BARRELS = 4

        openai_client = create_client()
        convo = LLMConversation(ai_client=openai_client)

        convo.add_developer_message("""
Count the number of times each letter of the alphabet appears in a key phrase
that the user will give you.

Ignore spaces, and treat all letters as lowercase for counting purposes.
Do not count any characters other than the 26 letters of the English alphabet.

Return a JSON object with the following structure:

{
    "scratchpad": string,
    "lettercounts": {
        "a": {
            "locations": string[],
            "count": number,
        },
        "b": {
            "locations": string[],
            "count": number,
        },
        "c": {
            "locations": string[],
            "count": number,
        },
        ...etc for all 26 letters...
    },
    "miscounts": string
}

The fields in this JSON object are defined as follows:

scratchpad: An internal deliberation you can have with yourself about how to best answer
        the question. Use this as a whiteboard to work through your reasoning process.
        PRO TIP: Be very careful to not count any position twice. If you find that you're
        counting one letter for position n, and then counting another letter for position n,
        then one or both must be wrong.

lettercounts: An object where each key is a letter of the English alphabet, and each value
        is an object with two fields:
        locations: An explicit list of the places where you found this letter. It should describe
                <count> distinct locations in the key phrase where this letter appears. Actually write
                out the text at the locations to prove that you found them, like this:
                [position 2, the first "o" in "foo": f *o* o],
                [position 3, the second "o" in "foo": f o *o*],
                etc.
                This will make it very easy to see if you mis-count, because if you highlight the
                wrong letter, then you will clearly be able to see the mismatch.
        count: The total number of times this letter appears in the key phrase.

miscounts: A retrospective examination of the counts you just provided, with particular
        attention to any letters where the location or position does not match the letter being
        counted -- e.g. if you said something like [position 7, the second "b" in "s t r a w b *e* r r y"]
        then you can clearly see that the letter you counted as "b" is not actually a "b".
""")
        convo.add_user_message("strawberry milkshake")

        formatparam: dict[str, Any] = {
            "scratchpad": str,
            "lettercounts": {},
            "miscounts": str,
        }

        for letter in "abcdefghijklmnopqrstuvwxyz":
            formatparam["lettercounts"][letter] = {
                "locations": [str],
                "count": int,
            }

        convo_before_submit = convo.clone()

        json_schema = JSONSchemaFormat(formatparam)
        convo.submit(shotgun=NUM_SHOTGUN_BARRELS, json_response=json_schema)

        reply = convo.get_last_reply_dict()

        expected_counts: dict[str, int] = {
            "a": 2,
            "b": 1,
            "c": 0,
            "d": 0,
            "e": 2,
            "f": 0,
            "g": 0,
            "h": 1,
            "i": 1,
            "j": 0,
            "k": 2,
            "l": 1,
            "m": 1,
            "n": 0,
            "o": 0,
            "p": 0,
            "q": 0,
            "r": 3,
            "s": 2,
            "t": 1,
            "u": 0,
            "v": 0,
            "w": 1,
            "x": 0,
            "y": 1,
            "z": 0,
        }

        observed_counts = {
            letter: (((reply.get("lettercounts") or {}).get(letter) or {}).get("count"))
            for letter in expected_counts
        }
        self.assertEqual(observed_counts, expected_counts)

        # Validate that non-shotgun mode is unreliable by requiring at least one miss.
        NUM_FAILURE_TRIALS = 8
        results = [
            convo_before_submit.clone().submit(json_response=json_schema)
            for _ in range(NUM_FAILURE_TRIALS)
        ]

        does_each_result_equal_expected: list[bool] = []
        for result in results:
            if not isinstance(result, dict):
                does_each_result_equal_expected.append(False)
                continue

            observed = {
                letter: (
                    ((result.get("lettercounts") or {}).get(letter) or {}).get("count")
                )
                for letter in expected_counts
            }
            does_each_result_equal_expected.append(observed == expected_counts)

        self.assertFalse(all(does_each_result_equal_expected))


if __name__ == "__main__":
    unittest.main()
