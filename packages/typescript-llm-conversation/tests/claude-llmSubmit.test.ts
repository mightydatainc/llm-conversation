import { describe, expect, it } from 'vitest';

import {
  CLAUDE_MODEL_SMART,
  AnthropicClientLike,
  TokenUsage,
} from '../src/llmProviders.js';
import { llmSubmit } from '../src/llmSubmit.js';

class FakeAnthropicResponse {
  content: any;
  stop_reason: string;

  constructor(text: any = '', stopReason: string = 'end_turn') {
    if (text === null) {
      this.content = null;
    } else {
      this.content = [{ type: 'text', text }];
    }
    this.stop_reason = stopReason;
  }
}

class FakeAnthropicMessagesAPI {
  sideEffects: any[];
  createCalls: Array<Record<string, unknown>>;

  constructor(sideEffects: any[] = []) {
    this.sideEffects = [...sideEffects];
    this.createCalls = [];
  }

  create(kwargs: Record<string, unknown>) {
    this.createCalls.push(kwargs);

    if (!this.sideEffects.length) {
      return new FakeAnthropicResponse();
    }

    const next = this.sideEffects.shift();
    if (next instanceof Error) {
      throw next;
    }
    return next;
  }
}

class FakeAnthropicClient implements AnthropicClientLike {
  messages: FakeAnthropicMessagesAPI;
  tokenUsage?: TokenUsage;

  constructor(sideEffects: any[] = []) {
    this.messages = new FakeAnthropicMessagesAPI(sideEffects);
  }
}

class AnthropicError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'AnthropicError';
  }
}

class BadRequestError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'BadRequestError';
  }
}

describe('Claude llmSubmit', () => {
  it('uses default model when json mode is disabled', async () => {
    const client = new FakeAnthropicClient([new FakeAnthropicResponse('ok')]);

    const result = await llmSubmit(
      [{ role: 'user', content: 'Hello' }],
      client
    );

    expect(result).toBe('ok');
    expect(client.messages.createCalls).toHaveLength(1);
    const request = client.messages.createCalls[0];
    expect(request.model).toBe(CLAUDE_MODEL_SMART);
  });

  it('passes datetime in system param and keeps user messages in messages param', async () => {
    const client = new FakeAnthropicClient([new FakeAnthropicResponse('ok')]);
    const messages = [{ role: 'user', content: 'Hello' }];

    await llmSubmit(messages, client);

    const request = client.messages.createCalls[0];
    // System messages (including datetime) go in the 'system' param
    expect(typeof request.system).toBe('string');
    expect((request.system as string).startsWith('!DATETIME:')).toBe(true);
    // Only non-system messages go in 'messages'
    expect(request.messages).toEqual(messages);
  });

  it('replaces stale datetime messages and keeps other system messages', async () => {
    const client = new FakeAnthropicClient([new FakeAnthropicResponse('ok')]);
    const messages = [
      { role: 'system', content: '!DATETIME: old timestamp' },
      { role: 'system', content: 'keep me' },
      { role: 'user', content: 'hello' },
    ];

    await llmSubmit(messages, client);

    const request = client.messages.createCalls[0];
    const systemText = request.system as string;
    // Should have exactly one datetime message
    const datetimeMatches = systemText.match(/!DATETIME:/g);
    expect(datetimeMatches).toHaveLength(1);
    // Should preserve other system messages
    expect(systemText).toContain('keep me');
    // Non-system messages should only contain the user message
    expect(request.messages).toEqual([{ role: 'user', content: 'hello' }]);
  });

  it('supports json_response=true and parses JSON', async () => {
    // Our code provides the leading curly brace. In the real world, Claude will
    // pick up where it left off and produce the rest of the JSON object, without
    // the leading curly brace. We thus omit the leading curly brace in our virtual
    // response to simulate this.
    const client = new FakeAnthropicClient([
      new FakeAnthropicResponse('{"value":1}'),
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

  it('Finds and parses JSON even with leading and trailing cruft', async () => {
    // Our code provides the leading curly brace. In the real world, Claude will
    // pick up where it left off and produce the rest of the JSON object, without
    // the leading curly brace. We thus omit the leading curly brace in our virtual
    // response to simulate this.
    const client = new FakeAnthropicClient([
      new FakeAnthropicResponse(
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
    const client = new FakeAnthropicClient([
      new FakeAnthropicResponse('{"first":1}{"second":2}'),
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

  it('retries anthropic errors and succeeds', async () => {
    const warnings: string[] = [];
    const client = new FakeAnthropicClient([
      new AnthropicError('temporary'),
      new FakeAnthropicResponse('ok'),
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
    expect(client.messages.createCalls).toHaveLength(2);
    expect(warnings).toHaveLength(1);
    expect(warnings[0].toLowerCase()).toContain('anthropic');
    expect(warnings[0]).toContain('API error');
    expect(warnings[0]).toContain('Retrying (attempt 1 of 2)');
  });

  it('retries json decode errors and succeeds', async () => {
    const warnings: string[] = [];
    const client = new FakeAnthropicClient([
      new FakeAnthropicResponse('not json'),
      new FakeAnthropicResponse('{"ok":true}'),
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
    expect(client.messages.createCalls).toHaveLength(2);
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toContain('JSON decode error');
  });

  it('throws BadRequestError immediately without retry', async () => {
    const badRequestError = new BadRequestError(
      "Invalid type for 'messages': expected array of message objects."
    );
    const client = new FakeAnthropicClient([badRequestError]);

    await expect(
      llmSubmit([{ role: 'user', content: 'hello' }], client, {
        retryLimit: 5,
        retryBackoffTimeSeconds: 30,
      })
    ).rejects.toBeInstanceOf(BadRequestError);

    expect(client.messages.createCalls).toHaveLength(1);
  });

  it('throws for malformed response content without retry', async () => {
    const client = new FakeAnthropicClient([new FakeAnthropicResponse(null)]);

    await expect(
      llmSubmit([{ role: 'user', content: 'hello' }], client, {
        retryLimit: 5,
      })
    ).rejects.toBeInstanceOf(TypeError);

    expect(client.messages.createCalls).toHaveLength(1);
  });

  it('sends output_config when jsonResponse is a schema object', async () => {
    const schema = {
      format: {
        type: 'json_schema',
        name: 'test_output',
        description: 'A test JSON schema for output formatting',
        schema: {
          type: 'object',
          properties: { value: { type: 'number' } },
          required: ['value'],
        },
      },
    };
    const client = new FakeAnthropicClient([
      new FakeAnthropicResponse('{"value":42}'),
    ]);

    const result = await llmSubmit(
      [{ role: 'user', content: 'give me a number' }],
      client,
      { jsonResponse: schema }
    );

    expect(result).toEqual({ value: 42 });
    const request = client.messages.createCalls[0];

    // Anthropic structured output uses output_config.format
    // BUT! It strips out everything but "type" and "schema"!
    const schemaExpected = JSON.parse(JSON.stringify(schema));
    delete schemaExpected.format.name;
    delete schemaExpected.format.description;
    expect(request.output_config).toEqual(schemaExpected);
  });

  it('throws immediately if jsonResponse object cannot be JSONized', async () => {
    const client = new FakeAnthropicClient([
      new FakeAnthropicResponse('{"unused":true}'),
    ]);

    const recursiveObject: any = { foo: 'bar' };
    recursiveObject.self = recursiveObject;

    await expect(
      llmSubmit([{ role: 'user', content: 'hello' }], client, {
        jsonResponse: recursiveObject,
      })
    ).rejects.toBeInstanceOf(TypeError);

    expect(client.messages.createCalls).toHaveLength(0);
  });

  it('tracks token usage totals and per-model counts across calls', async () => {
    const firstResponse = new FakeAnthropicResponse('first') as any;
    firstResponse.model = 'claude-test-a';
    firstResponse.usage = {
      input_tokens: 9,
      output_tokens: 5,
    };

    const secondResponse = new FakeAnthropicResponse('second') as any;
    secondResponse.model = 'claude-test-b';
    secondResponse.usage = {
      input_tokens: 4,
      output_tokens: 6,
    };

    const thirdResponse = new FakeAnthropicResponse('third') as any;
    thirdResponse.model = 'claude-test-a';
    thirdResponse.usage = {
      input_tokens: 1,
      output_tokens: 2,
    };

    const client = new FakeAnthropicClient([
      firstResponse,
      secondResponse,
      thirdResponse,
    ]);

    await llmSubmit([{ role: 'user', content: 'hello 1' }], client);
    await llmSubmit([{ role: 'user', content: 'hello 2' }], client);
    await llmSubmit([{ role: 'user', content: 'hello 3' }], client);

    expect(client.tokenUsage).toEqual({
      allModels: {
        total: 27,
        input: 14,
        output: 13,
      },
      byModel: {
        'claude-test-a': {
          total: 17,
          input: 10,
          output: 7,
        },
        'claude-test-b': {
          total: 10,
          input: 4,
          output: 6,
        },
      },
    });
  });

  it('does not initialize tokenUsage when provider usage is missing', async () => {
    const client = new FakeAnthropicClient([new FakeAnthropicResponse('ok')]);

    await llmSubmit([{ role: 'user', content: 'hello' }], client);

    expect(client.tokenUsage).toBeUndefined();
  });
});

describe('Claude llmSubmit shotgun', () => {
  const messages = [{ role: 'user', content: 'Hello' }];

  function makeClient(): FakeAnthropicClient {
    return new FakeAnthropicClient();
  }

  async function callSubmit(
    client: FakeAnthropicClient,
    options: Record<string, unknown> = {}
  ) {
    return llmSubmit(messages, client, options as any);
  }

  it('api calls increase linearly with shotgun count', async () => {
    const client3 = makeClient();
    await callSubmit(client3, { shotgun: 3 });
    const numApiCallsWithShotgun3 = client3.messages.createCalls.length;

    const client6 = makeClient();
    await callSubmit(client6, { shotgun: 6 });
    const numApiCallsWithShotgun6 = client6.messages.createCalls.length;

    const client9 = makeClient();
    await callSubmit(client9, { shotgun: 9 });
    const numApiCallsWithShotgun9 = client9.messages.createCalls.length;

    // Confirm that it's a linear increase.
    // Verify that f(9) - f(6) == f(6) - f(3)
    expect(numApiCallsWithShotgun9 - numApiCallsWithShotgun6).toBe(
      numApiCallsWithShotgun6 - numApiCallsWithShotgun3
    );
  });

  it('shotgun=0 is same as no shotgun', async () => {
    const clientNone = makeClient();
    await callSubmit(clientNone);
    const numApiCallsWithNoShotgun = clientNone.messages.createCalls.length;

    const client0 = makeClient();
    await callSubmit(client0, { shotgun: 0 });
    const numApiCallsWithShotgun0 = client0.messages.createCalls.length;

    expect(numApiCallsWithShotgun0).toBe(numApiCallsWithNoShotgun);
  });

  it('shotgun=1 is same as no shotgun', async () => {
    const clientNone = makeClient();
    await callSubmit(clientNone);
    const numApiCallsWithNoShotgun = clientNone.messages.createCalls.length;

    const client1 = makeClient();
    await callSubmit(client1, { shotgun: 1 });
    const numApiCallsWithShotgun1 = client1.messages.createCalls.length;

    expect(numApiCallsWithShotgun1).toBe(numApiCallsWithNoShotgun);
  });

  it('shotgun=2 makes more calls than no shotgun', async () => {
    const clientNone = makeClient();
    await callSubmit(clientNone);
    const numApiCallsWithNoShotgun = clientNone.messages.createCalls.length;

    const client2 = makeClient();
    await callSubmit(client2, { shotgun: 2 });
    const numApiCallsWithShotgun2 = client2.messages.createCalls.length;

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
    const numApiCallsWithNoShotgun = clientNone.messages.createCalls.length;

    const client3 = makeClient();
    await callSubmit(client3, { shotgun: 3 });
    const numApiCallsWithShotgun3 = client3.messages.createCalls.length;

    const client6 = makeClient();
    await callSubmit(client6, { shotgun: 6 });
    const numApiCallsWithShotgun6 = client6.messages.createCalls.length;

    const delta3FromNoShotgun =
      numApiCallsWithShotgun3 - numApiCallsWithNoShotgun;
    const delta3FromShotgun3 =
      numApiCallsWithShotgun6 - numApiCallsWithShotgun3;

    // f(3) - f(0) > f(6) - f(3), because f(0) lacks the overhead constant c.
    expect(delta3FromNoShotgun).toBeGreaterThan(delta3FromShotgun3);
  });
});
