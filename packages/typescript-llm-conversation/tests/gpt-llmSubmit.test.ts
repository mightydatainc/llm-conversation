import { describe, expect, it } from 'vitest';

import {
  GPT_MODEL_SMART,
  OpenAIClientLike,
  TokenUsage,
} from '../src/llmProviders.js';
import { llmSubmit } from '../src/llmSubmit.js';

class FakeOpenAIResponse {
  output_text: any;
  error: any;
  incomplete_details: any;

  constructor(
    outputText: any = '',
    error: any = null,
    incompleteDetails: any = null
  ) {
    this.output_text = outputText;
    this.error = error;
    this.incomplete_details = incompleteDetails;
  }
}

class FakeOpenAIResponsesAPI {
  sideEffects: any[];
  createCalls: Array<Record<string, unknown>>;

  constructor(sideEffects: any[] = []) {
    this.sideEffects = [...sideEffects];
    this.createCalls = [];
  }

  create(kwargs: Record<string, unknown>) {
    this.createCalls.push(kwargs);

    if (!this.sideEffects.length) {
      return new FakeOpenAIResponse();
    }

    const next = this.sideEffects.shift();
    if (next instanceof Error) {
      throw next;
    }
    return next;
  }
}

class FakeOpenAIClient implements OpenAIClientLike {
  responses: FakeOpenAIResponsesAPI;
  token_usage?: TokenUsage;

  constructor(sideEffects: any[] = []) {
    this.responses = new FakeOpenAIResponsesAPI(sideEffects);
  }
}

class OpenAIError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'OpenAIError';
  }
}

class BadRequestError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'BadRequestError';
  }
}

describe('GPT llmSubmit', () => {
  it('uses default model and omits text config when json mode is disabled', async () => {
    const client = new FakeOpenAIClient([new FakeOpenAIResponse('ok')]);

    const result = await llmSubmit(
      [{ role: 'user', content: 'Hello' }],
      client
    );

    expect(result).toBe('ok');
    expect(client.responses.createCalls).toHaveLength(1);
    const request = client.responses.createCalls[0];
    expect(request.model).toBe(GPT_MODEL_SMART);
    expect('text' in request).toBe(false);
  });

  it('prepends datetime system message and keeps user messages after it', async () => {
    const client = new FakeOpenAIClient([new FakeOpenAIResponse('ok')]);
    const messages = [{ role: 'user', content: 'Hello' }];

    await llmSubmit(messages, client);

    const submitted = client.responses.createCalls[0].input as Array<
      Record<string, string>
    >;
    expect(submitted[0].role).toBe('system');
    expect(submitted[0].content.startsWith('!DATETIME:')).toBe(true);
    expect(submitted.slice(1)).toEqual(messages);
  });

  it('replaces stale datetime messages and keeps other system messages', async () => {
    const client = new FakeOpenAIClient([new FakeOpenAIResponse('ok')]);
    const messages = [
      { role: 'system', content: '!DATETIME: old timestamp' },
      { role: 'system', content: 'keep me' },
      { role: 'user', content: 'hello' },
    ];

    await llmSubmit(messages, client);

    const submitted = client.responses.createCalls[0].input as Array<
      Record<string, string>
    >;
    const datetimeMessages = submitted.filter(
      (m) =>
        m.role === 'system' &&
        typeof m.content === 'string' &&
        m.content.startsWith('!DATETIME:')
    );
    expect(datetimeMessages).toHaveLength(1);
    expect(submitted.slice(1)).toEqual(messages.slice(1));
  });

  it('supports json_response=true with json_object text format', async () => {
    const client = new FakeOpenAIClient([
      new FakeOpenAIResponse('{"value":1}'),
    ]);

    const result = await llmSubmit(
      [{ role: 'user', content: 'json' }],
      client,
      {
        jsonResponse: true,
      }
    );

    expect(result).toEqual({ value: 1 });
    const request = client.responses.createCalls[0];
    expect(request.text).toEqual({ format: { type: 'json_object' } });
  });

  it('Finds and parses JSON even with leading and trailing cruft', async () => {
    // Our code provides the leading curly brace. In the real world, Claude will
    // pick up where it left off and produce the rest of the JSON object, without
    // the leading curly brace. We thus omit the leading curly brace in our virtual
    // response to simulate this.
    const client = new FakeOpenAIClient([
      new FakeOpenAIResponse(
        'Sure! Here is some JSON! {"value":1} Need anything else?'
      ),
    ]);

    const result = await llmSubmit(
      [{ role: 'user', content: 'json' }],
      client,
      {
        jsonResponse: true,
      }
    );

    expect(result).toEqual({ value: 1 });
  });

  it('parses first json object when response contains trailing json', async () => {
    const client = new FakeOpenAIClient([
      new FakeOpenAIResponse('{"first":1}{"second":2}'),
    ]);

    const result = await llmSubmit(
      [{ role: 'user', content: 'json' }],
      client,
      {
        jsonResponse: true,
      }
    );

    expect(result).toEqual({ first: 1 });
  });

  it('retries openai errors and succeeds', async () => {
    const warnings: string[] = [];
    const client = new FakeOpenAIClient([
      new OpenAIError('temporary'),
      new FakeOpenAIResponse('ok'),
    ]);

    const result = await llmSubmit(
      [{ role: 'user', content: 'hello' }],
      client,
      {
        retryLimit: 2,
        retryBackoffTimeSeconds: 0,
        warningCallback: (message: any) => warnings.push(message),
      }
    );

    expect(result).toBe('ok');
    expect(client.responses.createCalls).toHaveLength(2);
    expect(warnings).toHaveLength(1);
    expect(warnings[0].toLowerCase()).toContain('openai');
    expect(warnings[0]).toContain('API error');
    expect(warnings[0]).toContain('Retrying (attempt 1 of 2)');
  });

  it('retries json decode errors and succeeds', async () => {
    const warnings: string[] = [];
    const client = new FakeOpenAIClient([
      new FakeOpenAIResponse('not json'),
      new FakeOpenAIResponse('{"ok":true}'),
    ]);

    const result = await llmSubmit(
      [{ role: 'user', content: 'hello' }],
      client,
      {
        jsonResponse: true,
        retryLimit: 2,
        warningCallback: (message) => warnings.push(message),
      }
    );

    expect(result).toEqual({ ok: true });
    expect(client.responses.createCalls).toHaveLength(2);
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toContain('JSON decode error');
  });

  it('throws BadRequestError immediately without retry', async () => {
    const badRequestError = new BadRequestError(
      "Invalid type for 'input[11].content': expected one of an array of objects or string, but got an object instead."
    );
    const client = new FakeOpenAIClient([badRequestError]);

    await expect(
      llmSubmit([{ role: 'user', content: 'hello' }], client, {
        retryLimit: 5,
        retryBackoffTimeSeconds: 30,
      })
    ).rejects.toBeInstanceOf(BadRequestError);

    expect(client.responses.createCalls).toHaveLength(1);
  });

  it('throws for malformed response output_text without retry', async () => {
    const client = new FakeOpenAIClient([new FakeOpenAIResponse(null)]);

    await expect(
      llmSubmit([{ role: 'user', content: 'hello' }], client, {
        retryLimit: 5,
      })
    ).rejects.toBeInstanceOf(TypeError);

    expect(client.responses.createCalls).toHaveLength(1);
  });

  it('forwards jsonResponse schema object as text format parameter', async () => {
    const schema = {
      format: {
        type: 'json_schema',
        name: 'test_output',
        schema: {
          type: 'object',
          properties: { value: { type: 'number' } },
          required: ['value'],
        },
      },
    };
    const client = new FakeOpenAIClient([
      new FakeOpenAIResponse('{"value":42}'),
    ]);

    const result = await llmSubmit(
      [{ role: 'user', content: 'give me a number' }],
      client,
      { jsonResponse: schema }
    );

    expect(result).toEqual({ value: 42 });
    const request = client.responses.createCalls[0];
    expect(request.text).toEqual(schema);
  });

  it('throws immediately if jsonResponse object cannot be JSONized', async () => {
    const client = new FakeOpenAIClient([
      new FakeOpenAIResponse('{"unused":true}'),
    ]);

    const recursiveObject: any = { foo: 'bar' };
    recursiveObject.self = recursiveObject;

    await expect(
      llmSubmit([{ role: 'user', content: 'hello' }], client, {
        jsonResponse: recursiveObject,
      })
    ).rejects.toBeInstanceOf(TypeError);

    expect(client.responses.createCalls).toHaveLength(0);
  });

  it('tracks token usage totals and per-model counts across calls', async () => {
    const firstResponse = new FakeOpenAIResponse('first') as any;
    firstResponse.model = 'gpt-test-a';
    firstResponse.usage = {
      input_tokens: 10,
      output_tokens: 4,
    };

    const secondResponse = new FakeOpenAIResponse('second') as any;
    secondResponse.model = 'gpt-test-b';
    secondResponse.usage = {
      input_tokens: 3,
      output_tokens: 8,
    };

    const thirdResponse = new FakeOpenAIResponse('third') as any;
    thirdResponse.model = 'gpt-test-a';
    thirdResponse.usage = {
      input_tokens: 2,
      output_tokens: 1,
    };

    const client = new FakeOpenAIClient([
      firstResponse,
      secondResponse,
      thirdResponse,
    ]);

    await llmSubmit([{ role: 'user', content: 'hello 1' }], client);
    await llmSubmit([{ role: 'user', content: 'hello 2' }], client);
    await llmSubmit([{ role: 'user', content: 'hello 3' }], client);

    expect(client.token_usage).toEqual({
      all_models: {
        total: 28,
        input: 15,
        output: 13,
      },
      by_model: {
        'gpt-test-a': {
          total: 17,
          input: 12,
          output: 5,
        },
        'gpt-test-b': {
          total: 11,
          input: 3,
          output: 8,
        },
      },
    });
  });

  it('does not initialize token_usage when provider usage is missing', async () => {
    const client = new FakeOpenAIClient([new FakeOpenAIResponse('ok')]);

    await llmSubmit([{ role: 'user', content: 'hello' }], client);

    expect(client.token_usage).toBeUndefined();
  });
});

describe('GPT llmSubmit shotgun', () => {
  const messages = [{ role: 'user', content: 'Hello' }];

  function makeClient(): FakeOpenAIClient {
    return new FakeOpenAIClient();
  }

  async function callSubmit(
    client: FakeOpenAIClient,
    options: Record<string, unknown> = {}
  ) {
    return llmSubmit(messages, client, options as any);
  }

  it('api calls increase linearly with shotgun count', async () => {
    const client3 = makeClient();
    await callSubmit(client3, { shotgun: 3 });
    const numApiCallsWithShotgun3 = client3.responses.createCalls.length;

    const client6 = makeClient();
    await callSubmit(client6, { shotgun: 6 });
    const numApiCallsWithShotgun6 = client6.responses.createCalls.length;

    const client9 = makeClient();
    await callSubmit(client9, { shotgun: 9 });
    const numApiCallsWithShotgun9 = client9.responses.createCalls.length;

    // Confirm that it's a linear increase.
    // Verify that f(9) - f(6) == f(6) - f(3)
    expect(numApiCallsWithShotgun9 - numApiCallsWithShotgun6).toBe(
      numApiCallsWithShotgun6 - numApiCallsWithShotgun3
    );
  });

  it('shotgun=0 is same as no shotgun', async () => {
    const clientNone = makeClient();
    await callSubmit(clientNone);
    const numApiCallsWithNoShotgun = clientNone.responses.createCalls.length;

    const client0 = makeClient();
    await callSubmit(client0, { shotgun: 0 });
    const numApiCallsWithShotgun0 = client0.responses.createCalls.length;

    expect(numApiCallsWithShotgun0).toBe(numApiCallsWithNoShotgun);
  });

  it('shotgun=1 is same as no shotgun', async () => {
    const clientNone = makeClient();
    await callSubmit(clientNone);
    const numApiCallsWithNoShotgun = clientNone.responses.createCalls.length;

    const client1 = makeClient();
    await callSubmit(client1, { shotgun: 1 });
    const numApiCallsWithShotgun1 = client1.responses.createCalls.length;

    expect(numApiCallsWithShotgun1).toBe(numApiCallsWithNoShotgun);
  });

  it('shotgun=2 makes more calls than no shotgun', async () => {
    const clientNone = makeClient();
    await callSubmit(clientNone);
    const numApiCallsWithNoShotgun = clientNone.responses.createCalls.length;

    const client2 = makeClient();
    await callSubmit(client2, { shotgun: 2 });
    const numApiCallsWithShotgun2 = client2.responses.createCalls.length;

    expect(numApiCallsWithShotgun2).toBeGreaterThan(numApiCallsWithNoShotgun);
  });

  it('shotgun overhead is not invoked when shotgun is not used', async () => {
    // For shotgun >= 2, the call count is f(x) = x + c, where c is the fixed
    // overhead of the ponder and final reconciliation calls. f(0) = 1 (no overhead).
    // This means the jump from f(0) to f(3) is larger than the jump from f(3) to
    // f(6), because f(0) is missing the overhead constant c that all shotgun
    // invocations include.
    const clientNone = makeClient();
    await callSubmit(clientNone);
    const numApiCallsWithNoShotgun = clientNone.responses.createCalls.length;

    const client3 = makeClient();
    await callSubmit(client3, { shotgun: 3 });
    const numApiCallsWithShotgun3 = client3.responses.createCalls.length;

    const client6 = makeClient();
    await callSubmit(client6, { shotgun: 6 });
    const numApiCallsWithShotgun6 = client6.responses.createCalls.length;

    const delta3FromNoShotgun =
      numApiCallsWithShotgun3 - numApiCallsWithNoShotgun;
    const delta3FromShotgun3 =
      numApiCallsWithShotgun6 - numApiCallsWithShotgun3;

    // f(3) - f(0) > f(6) - f(3), because f(0) lacks the overhead constant c.
    expect(delta3FromNoShotgun).toBeGreaterThan(delta3FromShotgun3);
  });

  it('preserves warningCallback in shotgun mode when workers retry', async () => {
    const warnings: string[] = [];
    const client = new FakeOpenAIClient([
      new OpenAIError('temporary'),
      new FakeOpenAIResponse('worker ok 1'),
      new FakeOpenAIResponse('worker ok 2'),
      new FakeOpenAIResponse('ponder ok'),
      new FakeOpenAIResponse('final ok'),
    ]);

    const result = await callSubmit(client, {
      shotgun: 2,
      retryLimit: 1,
      retryBackoffTimeSeconds: 0,
      warningCallback: (message: string) => warnings.push(message),
    });

    expect(result).toBeTypeOf('string');
    expect(client.responses.createCalls.length).toBeGreaterThan(4);
    // Intended behavior: shotgun should preserve warning callbacks, so retry
    // warnings from worker calls are still emitted.
    expect(warnings.length).toBeGreaterThan(0);
    expect(warnings[0]).toContain('API error');
  });
});
