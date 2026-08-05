import { readFileSync } from 'fs';
import { join } from 'path';
import { OpenAI } from 'openai';
import { describe, expect, it } from 'vitest';
import { LLMConversation } from '../src/llmConversation.js';
import { JSONSchemaFormat } from '../src/jsonSchemaFormat.js';
import { GPT_MODEL_VISION, TokenUsage } from '../src/llmProviders.js';

const OPENAI_API_KEY = process.env.OPENAI_API_KEY?.trim();
if (!OPENAI_API_KEY) {
  throw new Error(
    'OPENAI_API_KEY is required for live API tests. Configure your test environment to provide it.'
  );
}

const createClient = (): OpenAI =>
  new OpenAI({
    apiKey: OPENAI_API_KEY,
  });

const IMAGE_IDENTIFICATION_SCHEMA = JSONSchemaFormat(
  {
    image_subject_enum: [
      'house',
      'chair',
      'boat',
      'car',
      'cat',
      'dog',
      'telephone',
      'duck',
      'city_skyline',
      'still_life',
      'bed',
      'headphones',
      'skull',
      'photo_camera',
      'unknown',
      'none',
      'error',
    ],
  },
  'ImageIdentification',
  'A test schema for image identification response'
);

describe('GPT integration (live API)', () => {
  it('should repeat Hello World', async () => {
    const openaiClient = createClient();
    const convo = new LLMConversation(openaiClient);

    convo.addUserMessage(`
This is a test to see if I'm correctly calling the OpenAI API to invoke GPT.
If you can see this, please respond with "Hello World" -- just like that,
with no additional text or explanation. Do not include punctuation or quotation
marks. Emit only the words "Hello World", capitalized as shown.
`);
    await convo.submit();

    const reply = convo.getLastReplyStr();
    expect(reply).toBe('Hello World');
  }, 180000);

  it('should invoke an LLM with some nominal intelligence', async () => {
    const openaiClient = createClient();
    const convo = new LLMConversation(openaiClient);

    // We'll use this opportunity to test the submitUserMessage convenience method.
    await convo.submitUserMessage(`
I'm conducting a test of my REST API response parsing systems.
If you can see this, please reply with the capital of France.
Reply *only* with the name of the city, with no additional text, punctuation,
or explanation. I'll be comparing your output string to a standard known
value, so it's important to the integrity of my system that the *only*
response you produce be *just* the name of the city. Standard capitalization
please -- first letter capitalized, all other letters lower-case.
`);

    const reply = convo.getLastReplyStr();
    expect(reply).toBe('Paris');
  }, 180000);

  it('should reply with a general-form JSON object', async () => {
    const openaiClient = createClient();
    const convo = new LLMConversation(openaiClient);

    convo.addUserMessage(`
This is a test to see if I'm correctly calling the OpenAI API to invoke GPT.

Please reply with the following JSON object, exactly as shown:

{
  "text": "Hello World",
  "success": true,
  "sample_array_data": [1, 2, {"nested_key": "nested_value"}]
}
`);
    await convo.submit(undefined, undefined, {
      jsonResponse: true,
    });

    const replyObj = convo.getLastReplyDict();

    // We'll test the structure of the JSON object piecemeal.
    expect(replyObj).toHaveProperty('text', 'Hello World');
    expect(replyObj).toHaveProperty('success', true);
    expect(replyObj).toHaveProperty('sample_array_data');
    expect(Array.isArray(replyObj.sample_array_data)).toBe(true);
    expect(replyObj.sample_array_data).toHaveLength(3);
    const arr = replyObj.sample_array_data as unknown[];
    expect(arr[0]).toBe(1);
    expect(arr[1]).toBe(2);
    expect(arr[2]).toHaveProperty('nested_key', 'nested_value');

    // While we're here, we'll also test to make sure that the shortcut
    // accessors also work.
    expect(convo.getLastReplyDictField('text')).toBe('Hello World');
    expect(convo.getLastReplyDictField('success')).toBe(true);
    expect(convo.getLastReplyDictField('sample_array_data')).toHaveLength(3);
  }, 180000);

  it('should reply with structured JSON using JSON schema spec', async () => {
    const openaiClient = createClient();
    const convo = new LLMConversation(openaiClient);

    const schema = {
      format: {
        type: 'json_schema',
        strict: true,
        name: 'TestSchema',
        description: 'A test schema for structured JSON response',
        schema: {
          type: 'object',
          properties: {
            text: { type: 'string' },
            success: { type: 'boolean' },
            sample_array_data: {
              type: 'array',
              items: { type: 'number' },
            },
            nested_dict: {
              type: 'object',
              properties: {
                nested_key: { type: 'string' },
              },
              required: ['nested_key'],
              additionalProperties: false,
            },
          },
          required: ['text', 'success', 'sample_array_data', 'nested_dict'],
          additionalProperties: false,
        },
      },
    };

    convo.addUserMessage(`
Please reply with a JSON object that contains the following data:

Success flag: true
Text: "Hello World"
Sample array data (2 elements long):
    Element 0: 5
    Element 1: 33
Nested dict (1 item long):
    Value under "nested_key": "foobar"
`);
    await convo.submit(undefined, undefined, {
      jsonResponse: schema,
    });

    expect(convo.getLastReplyDictField('success')).toBe(true);
    expect(convo.getLastReplyDictField('text')).toBe('Hello World');

    const nestedDict = convo.getLastReplyDictField('nested_dict') as Record<
      string,
      unknown
    >;
    expect(nestedDict).toHaveProperty('nested_key', 'foobar');
    expect(Object.keys(nestedDict)).toHaveLength(1);

    const sampleArrayData = convo.getLastReplyDictField(
      'sample_array_data'
    ) as unknown[];
    expect(sampleArrayData).toHaveLength(2);
    expect(sampleArrayData[0]).toBe(5);
    expect(sampleArrayData[1]).toBe(33);
  }, 180000);

  it('should reply with structured JSON using JSON formatter shorthand', async () => {
    const openaiClient = createClient();
    const convo = new LLMConversation(openaiClient);

    const schema = JSONSchemaFormat(
      {
        text: String,
        success: Boolean,
        sample_array_data: [Number],
        nested_dict: {
          nested_key: String,
        },
      },
      'TestSchema',
      'A test schema for structured JSON response'
    );

    convo.addUserMessage(`
Please reply with a JSON object that contains the following data:

Success flag: true
Text: "Hello World"
Sample array data (2 elements long):
    Element 0: 5
    Element 1: 33
Nested dict (1 item long):
    Value under "nested_key": "foobar"
`);
    await convo.submit(undefined, undefined, {
      jsonResponse: schema,
    });

    expect(convo.getLastReplyDictField('success')).toBe(true);
    expect(convo.getLastReplyDictField('text')).toBe('Hello World');

    const nestedDict = convo.getLastReplyDictField('nested_dict') as Record<
      string,
      unknown
    >;
    expect(nestedDict).toHaveProperty('nested_key', 'foobar');
    expect(Object.keys(nestedDict)).toHaveLength(1);

    const sampleArrayData = convo.getLastReplyDictField(
      'sample_array_data'
    ) as unknown[];
    expect(sampleArrayData).toHaveLength(2);
    expect(sampleArrayData[0]).toBe(5);
    expect(sampleArrayData[1]).toBe(33);
  }, 180000);

  it('should perform image recognition with manual content message', async () => {
    const openaiClient = createClient();
    const convo = new LLMConversation(
      openaiClient,
      undefined,
      GPT_MODEL_VISION
    );

    // Load the image ./fixtures/phoenix.png
    const pngBuffer = readFileSync(join(__dirname, 'fixtures', 'phoenix.png'));
    const imgDataUrl = `data:image/png;base64,${pngBuffer.toString('base64')}`;

    const gptMsgWithImage = {
      role: 'user',
      content: [
        {
          type: 'input_text',
          text: 'An image submitted by a user, needing identification',
        },
        {
          type: 'input_image',
          image_url: imgDataUrl,
          detail: 'high',
        },
      ],
    };
    // We don't use our convenience methods here.
    // We just build the message directly for now.
    // We'll test the image message add methods later.
    convo.push(gptMsgWithImage);

    convo.addUserMessage('What is this a picture of?');

    await convo.submit(undefined, undefined, {
      jsonResponse: IMAGE_IDENTIFICATION_SCHEMA,
    });

    expect(convo.getLastReplyDictField('image_subject_enum')).toBe('cat');
  }, 180000);

  it('should perform image recognition with convenience methods', async () => {
    const openaiClient = createClient();
    const convo = new LLMConversation(
      openaiClient,
      undefined,
      GPT_MODEL_VISION
    );

    // Load the image ./fixtures/phoenix.png
    const pngBuffer = readFileSync(join(__dirname, 'fixtures', 'phoenix.png'));
    const imgDataUrl = `data:image/png;base64,${pngBuffer.toString('base64')}`;

    convo.addImage(
      'user',
      'An image submitted by a user, needing identification',
      imgDataUrl
    );
    convo.addUserMessage('What is this a picture of?');

    await convo.submit(undefined, undefined, {
      jsonResponse: IMAGE_IDENTIFICATION_SCHEMA,
    });

    expect(convo.getLastReplyDictField('image_subject_enum')).toBe('cat');
  }, 180000);

  it('should track token usage', async () => {
    const openaiClient = createClient();
    const convo = new LLMConversation(openaiClient);

    const stage1 = async () => {
      convo.addUserMessage(
        'Write a 500-word story about a raccoon that steals the Mona Lisa.'
      );
      await convo.submit();

      // Keep the type checker from freaking out over the monkey-patch.
      // @ts-ignore
      const tokenUsage: TokenUsage = openaiClient.token_usage;

      expect(tokenUsage).toBeDefined();
      expect(tokenUsage.all_models.input).toBeGreaterThan(35);
      expect(tokenUsage.all_models.input).toBeLessThan(65);
      expect(tokenUsage.all_models.output).toBeGreaterThan(550);
      expect(tokenUsage.all_models.output).toBeLessThan(750);
      expect(tokenUsage.all_models.total).toEqual(
        tokenUsage.all_models.input + tokenUsage.all_models.output
      );

      const modelsUsed = Object.keys(tokenUsage.by_model);
      expect(modelsUsed.length).toEqual(1);
      const modelName = modelsUsed[0];

      const modelUsage = tokenUsage.by_model[modelName];
      expect(modelUsage).toBeDefined();
      expect(modelUsage.input).toEqual(tokenUsage.all_models.input);
      expect(modelUsage.output).toEqual(tokenUsage.all_models.output);
      expect(modelUsage.total).toEqual(tokenUsage.all_models.total);
    };
    await stage1();

    const stage2 = async () => {
      convo.addUserMessage(
        "Now write a 500-word story about that raccoon's arch-rival, a fox detective."
      );
      await convo.submit();

      // Keep the type checker from freaking out over the monkey-patch.
      // @ts-ignore
      const tokenUsage: TokenUsage = openaiClient.token_usage;

      expect(tokenUsage).toBeDefined();
      expect(tokenUsage.all_models.input).toBeGreaterThan(700);
      expect(tokenUsage.all_models.input).toBeLessThan(900);
      expect(tokenUsage.all_models.output).toBeGreaterThan(1300);
      expect(tokenUsage.all_models.output).toBeLessThan(1600);
      expect(tokenUsage.all_models.total).toEqual(
        tokenUsage.all_models.input + tokenUsage.all_models.output
      );

      const modelsUsed = Object.keys(tokenUsage.by_model);
      expect(modelsUsed.length).toEqual(1);
      const modelName = modelsUsed[0];

      const modelUsage = tokenUsage.by_model[modelName];
      expect(modelUsage).toBeDefined();
      expect(modelUsage.input).toEqual(tokenUsage.all_models.input);
      expect(modelUsage.output).toEqual(tokenUsage.all_models.output);
      expect(modelUsage.total).toEqual(tokenUsage.all_models.total);
    };
    await stage2();
  }, 180_000);

  it('should use shotgun to get a reliable answer on an unreliable question', async () => {
    // Adjust this number as needed to achieve a reliable pass rate.
    // Huge number of shotguns barrels is needed to get a consistent pass on this test,
    // because the question is so unreliable.
    const NUM_SHOTGUN_BARRELS = 4;

    const openaiClient = createClient();
    const convo = new LLMConversation(openaiClient);

    convo.addDeveloperMessage(`
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
`);
    convo.addUserMessage('strawberry milkshake');

    const formatparam: any = {
      scratchpad: String,
      lettercounts: {},
      miscounts: String,
    };

    for (const letter of 'abcdefghijklmnopqrstuvwxyz') {
      formatparam['lettercounts'][letter] = {
        locations: [String],
        count: Number,
      };
    }

    // Clone the convo to a "savepoint". This will allow us to validate its performance
    // against different run modalities.
    const convoBeforeSubmit = convo.clone();

    // Without shotgunning, this fails about 30% of the time.
    // Notably, it fails a *different* way each time.
    // Because of this multivariate leverage, the shotgun approach
    // is astronomically more likely to get a perfect answer than
    // any single attempt is on its own.
    // It's a lot of barrels, but we can't let this test be flaky.
    const jsonSchema = JSONSchemaFormat(formatparam);
    await convo.submit(undefined, undefined, {
      shotgun: NUM_SHOTGUN_BARRELS,
      jsonResponse: jsonSchema,
    });

    const reply = convo.getLastReplyDict() as any;

    // strawberry milkshake
    // s(2) t(1) r(3) a(2) w(1) b(1) e(2) y(1) m(1) i(1) l(1) k(2) h(1)
    const expectedCounts: Record<string, number> = {
      a: 2,
      b: 1,
      c: 0,
      d: 0,
      e: 2,
      f: 0,
      g: 0,
      h: 1,
      i: 1,
      j: 0,
      k: 2,
      l: 1,
      m: 1,
      n: 0,
      o: 0,
      p: 0,
      q: 0,
      r: 3,
      s: 2,
      t: 1,
      u: 0,
      v: 0,
      w: 1,
      x: 0,
      y: 1,
      z: 0,
    };
    const observedCounts = Object.fromEntries(
      Object.keys(expectedCounts).map((letter) => [
        letter,
        reply['lettercounts'][letter].count,
      ])
    );
    expect(observedCounts).toEqual(expectedCounts);

    // We need to validate that it reliably fails without shotgunning to confirm that the shotgun
    // approach is actually necessary to get a consistent pass on this test. If it passes 100%
    // of the time without shotgunning, then this test isn't actually doing anything useful.
    // Run several attempts in parallel (no shotgun) and assert that at least one fails.
    const NUM_FAILURE_TRIALS = 8;
    const results = await Promise.all(
      Array.from({ length: NUM_FAILURE_TRIALS }, () =>
        convoBeforeSubmit
          .clone()
          .submit(undefined, undefined, { jsonResponse: jsonSchema })
      )
    );

    const doesEachResultEqualExpected = results.map((result) => {
      const r = result as any;
      const observedCounts = Object.fromEntries(
        Object.keys(expectedCounts).map((letter) => [
          letter,
          r['lettercounts'][letter].count,
        ])
      );
      const isCorrect =
        JSON.stringify(observedCounts) === JSON.stringify(expectedCounts);
      return isCorrect;
    });

    // Assert that at least one of the attempts failed to get the correct answer.
    expect(doesEachResultEqualExpected.every(Boolean)).toBe(false);
  }, 360000);
});
