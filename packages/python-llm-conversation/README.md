# mightydatainc-llm-conversation

Utilities for managing multi-shot LLM conversations and structured JSON responses in a provider-agnostic way, with current support for OpenAI and Anthropic clients.

## Purpose and Rationale

This package exists to reduce the size, complexity, and repetitiveness of code used for interacting with LLM services.

Provider SDKs are flexible, but application code often repeats the same plumbing:

- message shaping and role management
- retry and transient failure handling
- forcing machine-readable JSON output
- keeping conversation state coherent over multiple turns or "shots"

This package gives you small, composable building blocks for those recurring concerns. The design goal is to keep your app code focused on product logic while these utilities handle the repetitive conversation and formatting mechanics.

## Components And Why They Exist

| Component          | Why it exists                                                                                                               | When to use it                                                                                           |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `llm_submit`       | Centralizes a robust "submit messages and return reply" workflow, including retries and optional structured output parsing. | One-off prompts or service-layer functions where you already manage message history yourself.            |
| `LLMConversation`  | Wraps a message list with role-aware helpers and submit methods, so stateful chat flows stay readable and less error-prone. | Multi-turn workflows where you want to append/submit messages incrementally.                             |
| `JSONSchemaFormat` | Provides a compact Python DSL to describe structured output schemas without hand-writing large JSON Schema dictionaries.    | You need stricter contracts than "just return JSON" and want fields/types/ranges/enums defined up front. |

## Quick Start

## Testing

Run unit tests with Python's built-in `unittest` runner from the package root:

```bash
python -m unittest discover tests -v
```

Technically, this runs both the unit tests *and* the integration tests, which have been built using the unit test framework. The integration test modules require provider API keys and are intended for live API validation. They presume the existence of a .env file.

If you want to run *just* the local unit tests that use fakes for cloud services, you can call:

```bash
python -m unittest tests.test_json_schema_format tests.test_llm_conversation tests.test_gpt_llm_submit tests.test_claude_llm_submit -v
```


### Stateless Prompt Submission With `llm_submit`

Use this pattern as a wrapper around either provider client. `llm_submit` provides error handling, smart retries, structured input/output type management, and shotgunning.

```python
from openai import OpenAI
from mightydatainc_llm_conversation import llm_submit

client = OpenAI()

story = llm_submit(
    messages=[
        {
            "role": "user",
            "content": "Write a short story about a raccoon who steals the Mona Lisa.",
        }
    ],
    ai_client=client,
)

print(story)
```

### Stateless Prompt Submission With Claude (`anthropic`)

The same helper works with an Anthropic client.

```python
import anthropic
from mightydatainc_llm_conversation import llm_submit

client = anthropic.Anthropic()

reply = llm_submit(
    messages=[
        {
            "role": "user",
            "content": "Summarize the causes of the French Revolution.",
        }
    ],
    ai_client=client,
)

print(reply)
```

### Stateful Chat Flows With `LLMConversation`

Use this pattern when your application relies on multi-stage (multi-"shot") conversation flows, especially if any of the shots are conditional.

```python
from openai import OpenAI
from mightydatainc_llm_conversation import LLMConversation

client = OpenAI()
conversation = LLMConversation(ai_client=client)

should_include_sidekick = True
should_emphasize_character_development = True

# The `submit*` methods make a call to the LLM, and take a few seconds to run.
# They'll return the string reply produced by the LLM.
story = conversation.submit_user_message(
    "Write a short story about a raccoon who steals the Mona Lisa."
)

if should_include_sidekick or should_emphasize_character_development:
    # The `add*` methods append messages to the conversation without actually
    # sending a request to the LLM, allowing us to queue multiple instructions
    # or conditionally adjust conversation topology.
    conversation.add_user_message(
        "That's a good first draft. But I'd now like you to enhance your story as follows."
    )
    if should_include_sidekick:
        # Note that, in this example, we don't know if the LLM did or didn't include
        # a sidekick in its original draft. We're performing a blind multi-shot.
        # We're essentially saying, "Look, I don't know what you just wrote, but
        # we're willing to bet that, whatever it was, it needs a sidekick."
        conversation.add_user_message(
            "- If the story doesn't already have a sidekick, add one."
        )
    if should_emphasize_character_development:
        # Again, this blind multi-shot conversation is essentially structured to
        # implicitly flow with the understanding that we don't know what story
        # the LLM originally wrote -- but whatever it was, we bet it needs more
        # character development.
        conversation.add_user_message(
            "- Focus more on the protagonist's character development."
        )

    # Use a chain-of-thought stage to let the LLM talk through its intended changes.
    # This is what "thinking" models actually do under-the-hood. Here, you can get
    # specialized "thinking" performance for your own specific needs, by telling the
    # LLM exactly what it needs to deliberate with itself about.
    # This is a `submit*` method, which will actually send a call to the LLM.
    # Its reply will be implicitly added to the conversation history.
    conversation.submit_user_message(
        "Discuss how you'd go about revising your story to integrate these suggestions. "
        + "Don't actually write a new draft yet. Just talk about it for now."
    )

    story = conversation.submit_user_message(
        "Now emit your final draft of the story, starting with the title."
    )

print(story)
```

## JSON Response Mode and `JSONSchemaFormat`

Modern LLM providers can emit responses in JSON format and support schema-constrained structured output flows. This package is designed to remain provider-agnostic, with built-in handling for OpenAI and Anthropic today.

This functionality is extremely valuable in integrating AI capabilities into traditional procedural workflows, but it's overlooked by many developers due in part to the syntactic complexity of its invocation. Instead, many developers tend to take much more fragile approaches, such as using string parsing (e.g. regexes and substring matching) to extract answers from LLM responses -- a technique that often ends up falling back on prompt engineering to beg, plead, cajole, bribe, or threaten the AI to please just produce a properly formatted reply. This package's JSON response mode capabilities are designed to make AI-integrated data processing feel more like software engineering and less like an Inquisitorial confession session.

The function `llm_submit` and the `LLMConversation` class's `submit` method both take an optional named argument: `json_response`. This can be set simply to `True` for a "lazy" JSON response using a structure that's described in plain English in the bodies of prior messages. Or, it can be set to a structured format description object (with the help of `JSONSchemaFormat`) to enforce a schema of your specification.

### Unenforced ("advisory") JSON response: `json_response=True`

The "lazy" approach to producing JSON output is to set the optional `json_response` argument to `True`. This allows you to specify a desired JSON format by simply describing it in plain English in the body of your submitted message strings.

```python
from openai import OpenAI
from mightydatainc_llm_conversation import LLMConversation

client = OpenAI()
conversation = LLMConversation(ai_client=client)

# The `submit*` methods actually send a real call to the LLM, and will
# take a few seconds to run. When it's finished, the story will implicitly
# get stored as part of the conversation history.
story = conversation.submit_user_message(
    "Write a short story about a raccoon who steals the Mona Lisa."
)

# We'll use an `add*` method, because the `submit*` methods are primarily
# convenience methods. The real workhorse is the `submit(...)` method itself, which
# exposes a finer array of options and variants. We'll call that next.
conversation.add_user_message(
    """
Answer a few questions about the story you just wrote.
- What is the name of the protagonist?
- Where does the story take place?
- How many attempts at theft does the protagonist make during the course of the story?
- Does the protagonist ultimately succeed?

Provide your response as a JSON object using the following structure:
{
    "protagonist_name": (string)
    "city": (string, one of: "Paris", "New York", "Tokyo", "Other")
    "number_of_theft_attempts": (int, between 0 and 10)
    "do_they_ultimately_succeed": (boolean)
}
"""
)
conversation.submit(json_response=True)

# In this example, we get the entire response object as a dict.
# Alternatively, we could use the helper method `get_last_reply_dict_field(...)`.
questionnaire = conversation.get_last_reply_dict()

print(story)
print("Protagonist: ", questionnaire["protagonist_name"])
print("City where the story takes place: ", questionnaire["city"])
print("Number of theft attempts: ", questionnaire["number_of_theft_attempts"])
print("Ultimately successful? ", questionnaire["do_they_ultimately_succeed"])
```

In this example, we made the AI come up with its own document and then answer structured questions about that document. However, in practice, you could of course submit your _own_ source document (e.g. in a call to `add_user_message(...)`), and have the AI answer questions about it.

Some caveats to keep in mind when using this approach:

- You _must_ have a message in your conversation history that describes a JSON structure.
- The JSON structure is not enforced by any kind of calling framework. The LLM takes your JSON structure description "under advisement", but is not obligated to adhere to it in any way.
- Particularly large or complex structures can cause the LLM to hang.

### Structured JSON response (OpenAI example): `json_response={"format": {...}}`

Structured-output schemas vary by provider. The example below uses OpenAI's `json_schema` format, but `json_response` is intentionally provider-facing so adapter logic can map to provider-specific payloads.

OpenAI structured output reference:

https://developers.openai.com/api/docs/guides/structured-outputs/

```python
from openai import OpenAI
from mightydatainc_llm_conversation import LLMConversation

client = OpenAI()
conversation = LLMConversation(ai_client=client)

story = conversation.submit_user_message(
    "Write a short story about a raccoon who steals the Mona Lisa."
)

# We *could* add a separate user message telling the AI to answer a few questions,
# but honestly just submitting this JSON query is enough to make the AI "understand"
# what we want it to do.
conversation.submit(
    json_response={
        "format": {
            "type": "json_schema",
            "name": "raccoon_story_questionnaire",
            "description": "Answer a few questions about this raccoon story.",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "protagonist_name": {
                        "type": "string",
                        "description": "What is the name of the protagonist?",
                    },
                    "city": {
                        "type": "string",
                        "description": "Where does the story take place?",
                        "enum": ["Paris", "New York", "Tokyo", "Other"],
                    },
                    "number_of_theft_attempts": {
                        "type": "integer",
                        "description": "How many attempts at theft does the protagonist make during the course of the story?",
                        "minimum": 0,
                        "maximum": 10,
                    },
                    "do_they_ultimately_succeed": {
                        "type": "boolean",
                        "description": "Does the protagonist ultimately succeed?",
                    },
                },
                "required": [
                    "protagonist_name",
                    "city",
                    "number_of_theft_attempts",
                    "do_they_ultimately_succeed",
                ],
                "additionalProperties": False,
            },
        }
    }
)

print(story)

# Use the helper method `get_last_reply_dict_field(...)` to get the parsed JSON responses.
print(
    "Protagonist: ",
    conversation.get_last_reply_dict_field("protagonist_name"),
)
print(
    "City where the story takes place: ",
    conversation.get_last_reply_dict_field("city"),
)
print(
    "Number of theft attempts: ",
    conversation.get_last_reply_dict_field("number_of_theft_attempts"),
)
print(
    "Ultimately successful? ",
    conversation.get_last_reply_dict_field("do_they_ultimately_succeed"),
)
```

### Structured JSON response with JSONSchemaFormat: `json_response=JSONSchemaFormat({...})`

While provider calling conventions are powerful and flexible, they are not particularly laconic or human-readable. As such, this package provides a specialized helper function called `JSONSchemaFormat`, which provides a translation layer that allows you to provide a `json_response` argument in a convenient form of shorthand.

JSONSchemaFormat takes a data structure that "looks like" the structure you want returned. It's extremely flexible (to the point of being somewhat sloppy), and not quite as expressive as the real `json_schema` structure. However, it's much more compact and readable.

How to use `JSONSchemaFormat` shorthand:

- A field's value is simply the type that you want that field to be returned as. E.g. `"protagonist_name": str`
- If a field's name isn't self-explanatory, it can be specified as a tuple containing type and string, where the string is a description. E.g. `"city": (str, "What city does the story take place in?")`
- If a field is a string, then that tuple can include a list specifying valid values. E.g. `"species": (str, "What is the species of the accomplice?", ["raccoon", "squirrel", "human", "pigeon", "other"])`
- If a field is a number, then that tuple can include a 2-element tuple specifying min and max values (`None` means open-ended on one side). E.g. `"number_of_accomplices": (int, "How many accomplices did the protagonist have?", (0, 5))`
- To make a field's value be returned as a list, present that field's value as a list.
- Fields can be objects, with this entire pattern nested multiple layers deep.

```python
from openai import OpenAI
from mightydatainc_llm_conversation import LLMConversation, JSONSchemaFormat

client = OpenAI()
conversation = LLMConversation(ai_client=client)

story = conversation.submit_user_message(
    "Write a short story about a raccoon who steals the Mona Lisa. "
    "Give the raccoon between 0 and 4 accomplices who help them in their heist, "
    "each of a different species."
)

# We *could* add a separate user message telling the AI to answer a few questions,
# but honestly just submitting this JSON query is enough to make the AI "understand"
# what we want it to do.
conversation.submit(
    json_response=JSONSchemaFormat(
        {
            "protagonist_name": str,
            "city": (str, "Where does this story take place?"),
            "number_of_theft_attempts": (
                int,
                "How many attempts do they make during the course of the story?",
                (0, 10),
            ),
            "do_they_ultimately_succeed": bool,
            "accomplices": [
                {
                    "name": str,
                    "species": (
                        str,
                        "What species is this accomplice?",
                        [
                            "raccoon",
                            "cat",
                            "dog",
                            "ferret",
                            "squirrel",
                            "pigeon",
                            "human",
                            "other",
                        ],
                    ),
                    "species_unusual": (
                        str,
                        "If the species is 'other', please specify it here. Otherwise, leave this blank.",
                    ),
                }
            ],
        }
    )
)

print(story)

# Use the helper method `get_last_reply_dict_field(...)` to get the parsed JSON responses.
print(
    "Protagonist: ",
    conversation.get_last_reply_dict_field("protagonist_name"),
)
print(
    "City where the story takes place: ",
    conversation.get_last_reply_dict_field("city"),
)
print(
    "Number of theft attempts: ",
    conversation.get_last_reply_dict_field("number_of_theft_attempts"),
)
print(
    "Ultimately successful? ",
    conversation.get_last_reply_dict_field("do_they_ultimately_succeed"),
)

for accomplice in conversation.get_last_reply_dict_field("accomplices") or []:
    species = str(accomplice.get("species", ""))
    if species == "other":
        species = str(accomplice.get("species_unusual", "")) + " (unusual)"
    print(f"Accomplice {accomplice.get('name')} is a {species}.")
```

Remember that, ultimately, `JSONSchemaFormat` is just a translation function. It produces a data structure that is converted into the schema format expected by the selected provider.

## Shotgunning

The `llm_submit` function and the `LLMConversation` class's `submit` method take an optional `shotgun` argument. `shotgun` takes a numerical argument that specifies the number of "barrels" (parallel workers) to launch. Use this when you want to spend extra cost/latency for potentially better output quality.

```python
from openai import OpenAI
from mightydatainc_llm_conversation import LLMConversation

client = OpenAI()
conversation = LLMConversation(ai_client=client)

conversation.submit_user_message(
    "Write a short story about a raccoon who steals the Mona Lisa."
)
conversation.add_user_message(
    "Think of one improvement that can be made to the story, and write a revised version of the story that incorporates that improvement."
)
conversation.submit(shotgun=3)
story = conversation.get_last_reply_str()

print(story)
```

Shotgunning is particularly useful when you have a multi-component AI-related task where any given component tends to pass, but at least one component somewhere in the task tends to fail -- i.e. a different component fails every time. Under such circumstances, running 3 or 4 parallel workers could dramatically improve the reliability of the final output. For details, study the shotgunning unit test.

## API

### LLMConversation

Stateful conversation container built on top of `llm_submit`.

What it does:

- Stores message history as a mutable list of `{role, content}` message objects.
- Provides role-specific helpers for appending and submitting messages.
- Tracks the latest assistant result in `last_reply` for easy typed access.

Initialization:

- `ai_client`: Provider client (or compatible object) used for submissions. Built-in adapters currently support OpenAI and Anthropic.
- `messages`: Optional initial message list.
- `model`: Optional default model used by `submit(...)` when no per-call model is provided.

Core submission behavior:

- `submit(...)` optionally appends a message, then calls `llm_submit(...)` with current history.
- Supports per-call `model`, `json_response`, and `shotgun` options.
- Appends the final assistant reply back into conversation history and updates `last_reply`.

Message helpers:

- `add_message(role, content)` plus role helpers: `add_user_message`, `add_assistant_message`, `add_system_message`, `add_developer_message`.
- `add_image(role, text, image_data_url)` for multimodal text+image message payloads.
- Non-string `dict` content is serialized to JSON text; list content is preserved for multimodal payloads.

Submit convenience methods:

- `submit_message(role, content)`
- `submit_user_message(content)`
- `submit_assistant_message(content)`
- `submit_system_message(content)`
- `submit_developer_message(content)`
- `submit_image(role, text, image_data_url)`

Inspection and utility methods:

- `get_last_message()`
- `get_messages_by_role(role)`
- `get_last_reply_str()`
- `get_last_reply_dict()`
- `get_last_reply_dict_field(field_name, default=None)`
- `to_dict_list()`
- `clone()`
- `assign_messages(messages)`

Failure behavior:

- Raises `ValueError` if `submit(...)` is called without an `ai_client`.
- Exceptions from `llm_submit(...)` are propagated.

### llm_submit

Primary stateless submit helper for provider API calls, with built-in adapters for OpenAI Responses API and Anthropic Messages API.

What it does:

- Submits a list of role/content messages and returns the model reply.
- Injects a fresh `!DATETIME` system message on each call.
- Optionally prepends a `system_announcement_message` before the datetime message.
- Supports retry/backoff behavior for transient failures.
- Supports JSON response modes and optional shotgunning.

Arguments:

- `messages`: Conversation payload in provider-compatible role/content message format.
- `ai_client`: Provider client compatible with current adapters (OpenAI: `responses.create(...)`; Anthropic: `messages.create(...)`).
- `model`: Optional model override. Defaults to the package smart model.
- `json_response`: Output mode control.
- `system_announcement_message`: Optional additional top-level system instruction.
- `retry_limit`: Maximum retry attempts for retryable failures.
- `retry_backoff_time_seconds`: Backoff delay between retry attempts.
- `shotgun`: Number of parallel worker barrels to launch (`>1` enables reconciliation mode).
- `warning_callback`: Optional callback for non-fatal warnings/retry notices.

`json_response` modes:

- `None` or `False`: Return plain text.
- `True`: Request JSON object mode (`{"format": {"type": "json_object"}}`).
- `dict`: Use provided provider-compatible structured-output config.

Return value:

- Text mode: `str` (trimmed).
- JSON mode: parsed JSON object from the model output.

Failure behavior:

- Retries transient provider API failures up to `retry_limit` with backoff (built-in retry classifiers currently cover OpenAI/Anthropic SDK errors).
- Retries JSON parsing failures in JSON mode.
- Raises immediately (no retry) for non-retryable protocol/request failures.

## Installation and usage

```bash
pip install mightydatainc-llm-conversation openai anthropic
```

```python
from mightydatainc_llm_conversation import llm_submit, LLMConversation, JSONSchemaFormat
```

Requires Python `>=3.13`.
