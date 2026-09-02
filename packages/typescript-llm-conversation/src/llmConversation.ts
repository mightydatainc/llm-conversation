import {
  AIClientLike,
  getModelName,
  identifyLLMProvider,
  LLMProviderName,
} from './index.js';
import { llmSubmit } from './llmSubmit.js';

/**
 * A single message in a conversation, with a role (e.g. `"user"`,
 * `"assistant"`, `"system"`, `"developer"`) and a content value that is
 * either a plain string or an array of content parts (used for multimodal
 * messages such as images).
 */
export interface ConversationMessage {
  role: string;
  content: string | unknown[];
}

/**
 * Per-call options for {@link LLMConversation.submit}.
 *
 * @property model - Overrides the conversation's default model for this call
 *   only.
 * @property jsonResponse - When `true`, requests a plain JSON object response.
 *   When an object, that object is forwarded to the model as a JSON schema.
 * @property shotgun - When greater than 1, fires this many parallel requests
 *   and reconciles the results. See {@link gptSubmit} for details.
 */
export interface SubmitOptions {
  model?: string;
  jsonResponse?: boolean | Record<string, unknown>;
  shotgun?: number;
}

/**
 * A stateful conversation wrapper around the OpenAI Responses API.
 *
 * `LLMConversation` extends `Array<ConversationMessage>` so the message
 * history can be iterated, spread, and indexed directly. Helper methods cover
 * the full lifecycle: adding messages, submitting to the API, and reading back
 * the last reply in various forms.
 *
 * @example
 * ```ts
 * const convo = new LLMConversation([], { openaiClient: client });
 * await convo.submitUserMessage('Hello!');
 * console.log(convo.getLastReplyStr());
 * ```
 */
export class LLMConversation extends Array<ConversationMessage> {
  static get [Symbol.species](): ArrayConstructor {
    return Array;
  }

  #aiClient?: AIClientLike;
  #model?: string;
  #lastReply: unknown = null;

  get aiClient(): AIClientLike | undefined {
    return this.#aiClient;
  }

  set aiClient(value: AIClientLike | undefined) {
    this.#aiClient = value;
  }

  get model(): string | undefined {
    return this.#model;
  }

  set model(value: string | undefined) {
    this.#model = value;
  }

  get lastReply(): unknown {
    return this.#lastReply;
  }

  set lastReply(value: unknown) {
    this.#lastReply = value;
  }

  get LLMProvider(): LLMProviderName | undefined {
    if (!this.#aiClient) {
      return undefined;
    }
    return identifyLLMProvider(this.#aiClient);
  }

  /**
   * Creates a new `LLMConversation`, optionally pre-populated with messages
   * and configured with an OpenAI client and model.
   *
   * @param messages - Initial conversation history.
   * @param options - Optional client and model defaults.
   */
  constructor(
    aiClient?: AIClientLike,
    messages?: ConversationMessage[],
    model?: string
  ) {
    super(...(messages || []));
    this.#aiClient = aiClient;
    this.#model = model;
  }

  /**
   * Replaces the entire message history with `messages`, clearing the array
   * first. If `messages` is empty or undefined, the conversation is left
   * empty.
   *
   * @param messages - The new message history to assign.
   * @returns `this`, for chaining.
   */
  assignMessages(messages?: ConversationMessage[]): this {
    this.length = 0;
    if (messages?.length) {
      this.push(...messages);
    }
    return this;
  }

  /**
   * Returns a deep clone of this conversation, including all messages,
   * member values, and the `openaiClient` reference.
   *
   * @returns A new `LLMConversation` instance with the same history and
   *   configuration.
   */
  clone(): LLMConversation {
    const retval = new LLMConversation();
    retval.aiClient = this.aiClient;
    retval.model = this.model;

    if (
      this.lastReply === undefined ||
      this.lastReply === null ||
      typeof this.lastReply === 'string'
    ) {
      retval.lastReply = this.lastReply;
    } else {
      // For objects and arrays, perform a deep copy to avoid shared references.
      retval.lastReply = JSON.parse(JSON.stringify(this.lastReply));
    }

    retval.assignMessages(this.toDictList());
    return retval;
  }

  /**
   * Submits the conversation history to the OpenAI API. Automatically appends
   * the response to the conversation as an `"assistant"` message and updates
   * `lastReply` to the new content.
   *
   * Optionally appends a message to the conversation, then submits the full
   * history to the OpenAI API and appends the assistant's reply.
   *
   * When `message` is a `Record` with `jsonResponse` key, it is interpreted as a
   * JSON schema format descriptor and used as `jsonResponse`. When it has a
   * `role` key, that role is used for the message. When it has a `content` key,
   * only that content is added to the history.
   *
   * @param message - An optional message to append before submitting. May be
   *   a plain string, a content record, or omitted to submit the existing
   *   history as-is.
   * @param role - The role to use when appending `message`. Defaults to
   *   `"user"`, or the role specified in the message record if available.
   * @param options - Per-call overrides for model, JSON response format, and
   *   shotgun parallelism.
   * @returns The assistant's reply — a `string` for plain-text responses or a
   *   parsed JSON value when `jsonResponse` is set.
   * @throws {Error} If `openaiClient` is not set.
   */
  async submit(
    message?: string | Record<string, unknown>,
    role?: string,
    options: SubmitOptions = {}
  ): Promise<unknown> {
    if (!this.aiClient) {
      throw new Error('AI client is not set. Please provide an AI client.');
    }

    if (message) {
      let messageObj = {
        role: 'user',
        content: '',
      };
      if (typeof message === 'string') {
        messageObj.content = message;
      } else {
        messageObj = { ...messageObj, ...message };
      }

      if (role) {
        messageObj.role = role;
      }
      this.addMessage(messageObj.role, messageObj.content);
    }

    const model =
      options.model ||
      this.model ||
      getModelName(this.LLMProvider || 'openai', 'smart');
    let jsonResponse = options.jsonResponse;

    const llmReply = await llmSubmit(this.toDictList(), this.aiClient, {
      jsonResponse,
      model,
      shotgun: options.shotgun,
    });

    this.addAssistantMessage(llmReply);
    return llmReply;
  }

  /**
   * Appends a message with the given role to the conversation history.
   *
   * Non-string content is normalised: arrays are stored as-is, plain objects
   * are JSON-stringified, and all other values are coerced via `String()`.
   *
   * @param role - The role of the message (e.g. `"user"`, `"assistant"`).
   * @param content - The message content.
   * @returns `this`, for chaining.
   */
  addMessage(role: string, content: unknown): this {
    let normalizedContent: string | unknown[];
    if (typeof content === 'string') {
      normalizedContent = content;
    } else if (Array.isArray(content)) {
      normalizedContent = content;
    } else {
      normalizedContent = JSON.stringify(content, null, 2);
    }

    this.push({ role, content: normalizedContent });
    return this;
  }

  /**
   * Appends a `"user"` message to the conversation history.
   *
   * @param content - The message content.
   * @returns `this`, for chaining.
   */
  addUserMessage(content: unknown): this {
    return this.addMessage('user', content);
  }

  /**
   * Appends an `"assistant"` message to the conversation history.
   * Updates `lastReply` to the new content.
   *
   * @param content - The message content.
   * @returns `this`, for chaining.
   */
  addAssistantMessage(content: unknown): this {
    this.lastReply = content;
    return this.addMessage('assistant', content);
  }

  /**
   * Appends a `"system"` message to the conversation history.
   *
   * @param content - The message content.
   * @returns `this`, for chaining.
   */
  addSystemMessage(content: unknown): this {
    return this.addMessage('system', content);
  }

  /**
   * Appends a `"developer"` message to the conversation history.
   *
   * @param content - The message content.
   * @returns `this`, for chaining.
   */
  addDeveloperMessage(content: unknown): this {
    return this.addMessage('developer', content);
  }

  /**
   * Appends a multimodal message containing both a text prompt and an image
   * to the conversation history.
   *
   * @param role - The role of the message (e.g. `"user"`).
   * @param text - The text prompt accompanying the image.
   * @param imageDataUrl - A data URL (e.g. `data:image/png;base64,...`) or
   *   HTTP URL of the image to include.
   * @returns `this`, for chaining.
   */
  addImage(role: string, text: string, imageDataUrl: string): this {
    const llmProvider = this.LLMProvider;
    if (!llmProvider) {
      throw new Error(
        'LLM provider cannot be identified. Please set an AI client with a supported provider.'
      );
    }
    if (llmProvider === 'openai' || llmProvider === 'deepseek') {
      const gptMsgWithImage = {
        role: role,
        content: [
          {
            type: 'input_text',
            text: text,
          },
          {
            type: 'input_image',
            image_url: imageDataUrl,
            detail: 'high',
          },
        ],
      };
      this.push(gptMsgWithImage);
      return this;
    }

    if (llmProvider === 'anthropic') {
      const trimmedImageDataUrl = imageDataUrl.trim();

      let imageSource:
        | { type: 'url'; url: string }
        | { type: 'base64'; media_type: string; data: string };

      if (/^https?:\/\//i.test(trimmedImageDataUrl)) {
        imageSource = {
          type: 'url',
          url: trimmedImageDataUrl,
        };
      } else {
        const dataUrlMatch = trimmedImageDataUrl.match(
          /^data:(image\/[a-zA-Z0-9.+-]+);base64,([\s\S]+)$/i
        );

        if (!dataUrlMatch) {
          throw new Error(
            'Anthropic image input must be an HTTP(S) URL or a base64 data URL, such as data:image/png;base64,...'
          );
        }

        let mediaType = dataUrlMatch[1].toLowerCase();
        if (mediaType === 'image/jpg') {
          mediaType = 'image/jpeg';
        }

        if (
          !['image/jpeg', 'image/png', 'image/gif', 'image/webp'].includes(
            mediaType
          )
        ) {
          throw new Error(
            `Unsupported Anthropic image media type: ${mediaType}. Supported types: image/jpeg, image/png, image/gif, image/webp.`
          );
        }

        imageSource = {
          type: 'base64',
          media_type: mediaType,
          data: dataUrlMatch[2].replace(/\s+/g, ''),
        };
      }

      const anthropicMsgWithImage = {
        role: role,
        content: [
          {
            type: 'image',
            source: imageSource,
          },
          {
            type: 'text',
            text: text,
          },
        ],
      };
      this.push(anthropicMsgWithImage);
      return this;
    }

    throw new Error(
      `LLM provider ${llmProvider} does not support image messages.`
    );
  }

  /**
   * Appends a message with the given role, then submits the conversation.
   *
   * @param role - The role of the message to add.
   * @param content - The message content.
   * @returns The assistant's reply.
   */
  async submitMessage(role: string, content: unknown): Promise<unknown> {
    this.addMessage(role, content);
    return this.submit();
  }

  /**
   * Appends a `"user"` message, then submits the conversation.
   *
   * @param content - The user message content.
   * @returns The assistant's reply.
   */
  async submitUserMessage(content: unknown): Promise<unknown> {
    this.addUserMessage(content);
    return this.submit();
  }

  /**
   * Appends an `"assistant"` message, then submits the conversation.
   *
   * @param content - The assistant message content.
   * @returns The assistant's reply.
   */
  async submitAssistantMessage(content: unknown): Promise<unknown> {
    this.addAssistantMessage(content);
    return this.submit();
  }

  /**
   * Appends a `"system"` message, then submits the conversation.
   *
   * @param content - The system message content.
   * @returns The assistant's reply.
   */
  async submitSystemMessage(content: unknown): Promise<unknown> {
    this.addSystemMessage(content);
    return this.submit();
  }

  /**
   * Appends a `"developer"` message, then submits the conversation.
   *
   * @param content - The developer message content.
   * @returns The assistant's reply.
   */
  async submitDeveloperMessage(content: unknown): Promise<unknown> {
    this.addDeveloperMessage(content);
    return this.submit();
  }

  /**
   * Appends an image message with the given role, then submits the
   * conversation.
   *
   * @param role - The role of the message (e.g. `"user"`).
   * @param text - The text prompt accompanying the image.
   * @param imageDataUrl - A data URL or HTTP URL of the image.
   * @returns The assistant's reply.
   */
  async submitImage(
    role: string,
    text: string,
    imageDataUrl: string
  ): Promise<unknown> {
    this.addImage(role, text, imageDataUrl);
    return this.submit();
  }

  /**
   * Returns the last message in the conversation history, or `null` if the
   * conversation is empty.
   */
  getLastMessage(): ConversationMessage | null {
    return this.length ? this[this.length - 1] : null;
  }

  /**
   * Returns all messages whose `role` matches the given value.
   *
   * @param role - The role to filter by (e.g. `"user"`, `"assistant"`).
   * @returns An array of matching messages.
   */
  getMessagesByRole(role: string): ConversationMessage[] {
    return this.filter((message) => message.role === role);
  }

  /**
   * Returns the last reply as a string. Returns an empty string if the last
   * reply is not a string (e.g. a parsed JSON value).
   */
  getLastReplyStr(): string {
    return typeof this.lastReply === 'string' ? this.lastReply : '';
  }

  /**
   * Returns the last reply as a deep-cloned plain object. Returns an empty
   * object `{}` if the last reply is not a record.
   */
  getLastReplyDict(): Record<string, unknown> {
    try {
      const retval: any = JSON.parse(JSON.stringify(this.lastReply));

      // Test to make sure that the result is actually a record and not something else
      // like an array or a primitive value.
      if (
        typeof retval === 'object' &&
        retval !== null &&
        !Array.isArray(retval)
      ) {
        return retval;
      } else {
        return {};
      }
    } catch (JSONError) {
      return {};
    }
  }

  /**
   * Returns a single field from the last reply object by name.
   *
   * @param fieldName - The key to look up in the last reply object.
   * @param defaultValue - Value to return when the field is absent or the
   *   last reply is not a record. Defaults to `undefined`.
   * @returns The field value, or `defaultValue` if not found.
   */
  getLastReplyDictField(
    fieldName: string,
    defaultValue: unknown = undefined
  ): unknown {
    try {
      const lastReplyDict = this.getLastReplyDict();
      return lastReplyDict[fieldName] ?? defaultValue;
    } catch {
      return defaultValue;
    }
  }

  /**
   * Returns a deep copy of the conversation history as a plain array of
   * {@link ConversationMessage} objects, suitable for serialisation or passing
   * to {@link gptSubmit} directly.
   */
  toDictList(): ConversationMessage[] {
    return JSON.parse(JSON.stringify([...this]));
  }
}
