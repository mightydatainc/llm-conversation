import {
  currentDatetimeSystemMessage,
  parseFirstJsonValue,
} from './helpers.js';
import {
  AIClientLike,
  getModelName,
  identifyLLMProvider,
  LLM_RETRY_BACKOFF_TIME_SECONDS_DEFAULT,
  LLM_RETRY_LIMIT_DEFAULT,
} from './llmProviders.js';
import { llmSubmitShotgun } from './llmSubmitShotgun.js';

/**
 * Options controlling the behaviour of {@link llmSubmit}.
 *
 * @property model - The model ID to use. If not specified, defaults to the "smart"
 *   tier model for the detected LLM provider (e.g. `gpt-4.1` for OpenAI).
 * @property jsonResponse - When `true`, requests a plain JSON object response.
 *   When a `Record` or JSON string, that value is forwarded as the `text` format
 *   parameter (i.e. a JSON schema). Defaults to `undefined` (plain text).
 * @property systemAnnouncementMessage - An optional system message prepended to
 *   every request, before all other messages.
 * @property retryLimit - Maximum number of retry attempts on recoverable errors.
 *   Defaults to {@link LLM_RETRY_LIMIT_DEFAULT}.
 * @property retryBackoffTimeSeconds - Seconds to wait between retries for
 *   recoverable API errors. Defaults to {@link LLM_RETRY_BACKOFF_TIME_SECONDS_DEFAULT}.
 * @property shotgun - When greater than 1, the request is sent to this many
 *   parallel worker calls and the results are reconciled by a final call.
 * @property warningCallback - Optional callback invoked with a human-readable
 *   warning string on recoverable errors and incomplete responses.
 */
export interface LLMSubmitOptions {
  model?: string;
  jsonResponse?: boolean | Record<string, unknown> | string;
  systemAnnouncementMessage?: string;
  retryLimit?: number;
  retryBackoffTimeSeconds?: number;
  shotgun?: number;
  warningCallback?: (message: string) => void;
}

/**
 * Returns `true` when `error` is an OpenAI SDK error that is worth retrying
 * (e.g. rate-limit or transient server errors). Detection is based on the
 * error's `name` property containing `"OpenAI"` or `"APIError"`.
 *
 * @param error - The caught value to inspect.
 * @returns `true` if the error is a retryable OpenAI API error.
 */
function isRetryableOpenAIError(error: unknown): boolean {
  if (!(error instanceof Error)) {
    return false;
  }

  const name = error.name || '';
  return name.includes('OpenAI') || name.includes('APIError');
}

/**
 * Returns `true` when `error` is an Anthropic SDK error that is worth retrying
 * (e.g. rate-limit or transient server errors). Detection is based on the
 * error's `name` property containing `"Anthropic"` or `"APIError"`.
 *
 * @param error - The caught value to inspect.
 * @returns `true` if the error is a retryable Anthropic API error.
 */
function isRetryableAnthropicError(error: unknown): boolean {
  if (!(error instanceof Error)) {
    return false;
  }

  const name = error.name || '';
  return name.includes('Anthropic') || name.includes('APIError');
}

// Prompt to prohibit unicode characters in model output.
// Currently not used, but may be useful in the future.
const PROMPT_UNICODE_PROHIBITION = `
ABSOLUTELY NO UNICODE ALLOWED. 
Only use typeable keyboard characters. Do not try to circumvent this rule 
with escape sequences, backslashes, or other tricks. Use double dashes (--)
instead of em-dashes or en-dashes; use straight quotes (") and single quotes (')
 instead of curly versions, use hyphens instead of bullets, etc.
`.trim();

/**
 * Updates the token usage statistics for a given model within an AI client.
 * This includes both the total token counts and the breakdown by individual model.
 * Modifies the aiClient object in-situ.
 * @param aiClient The AI client whose token usage is being updated.
 * @param modelName The name of the model for which token usage is being recorded.
 * @param tokenCountInput The number of input tokens to add to the usage statistics.
 * @param tokenCountOutput The number of output tokens to add to the usage statistics.
 */
const updateModelTokenUsage = (
  aiClient: AIClientLike,
  modelName: string,
  tokenCountInput: number,
  tokenCountOutput: number
): undefined => {
  aiClient.token_usage = aiClient.token_usage || {
    all_models: { total: 0, input: 0, output: 0 },
    by_model: {},
  };
  aiClient.token_usage.all_models.total += tokenCountInput + tokenCountOutput;
  aiClient.token_usage.all_models.input += tokenCountInput;
  aiClient.token_usage.all_models.output += tokenCountOutput;

  aiClient.token_usage.by_model[modelName] = aiClient.token_usage.by_model[
    modelName
  ] || { total: 0, input: 0, output: 0 };
  aiClient.token_usage.by_model[modelName].total +=
    tokenCountInput + tokenCountOutput;
  aiClient.token_usage.by_model[modelName].input += tokenCountInput;
  aiClient.token_usage.by_model[modelName].output += tokenCountOutput;
};

/**
 * Submits a conversation to the AI's API and returns the model's reply.
 *
 * @param messages - The conversation history, including any prior assistant
 *   turns. Each element should be a message object with at minimum `role` and
 *   `content` fields.
 * @param aiClient - An {@link AIClientLike} instance used to call the API.
 * @param options - Optional settings controlling model, JSON mode, retries,
 *   and shotgun parallelism.
 * @returns The model's response. A plain `string` when `jsonResponse` is
 *   falsy; otherwise a parsed JSON value.
 * @throws The last encountered error after all retry attempts are exhausted,
 *   or immediately for non-retryable errors.
 */
export const llmSubmit = async (
  messages: unknown[],
  aiClient: AIClientLike,
  options: LLMSubmitOptions = {}
): Promise<string | Record<string, unknown>> => {
  if (options.shotgun && options.shotgun > 1) {
    return await llmSubmitShotgun(messages, aiClient, options, options.shotgun);
  }

  const llmProviderName = identifyLLMProvider(aiClient);

  const model = options.model || getModelName(llmProviderName, 'smart');
  const retryLimit = options.retryLimit ?? LLM_RETRY_LIMIT_DEFAULT;
  const retryBackoffTimeSeconds =
    options.retryBackoffTimeSeconds ?? LLM_RETRY_BACKOFF_TIME_SECONDS_DEFAULT;

  const systemAnnouncement = `${(options.systemAnnouncementMessage || '').trim()}`;

  let failedError: unknown = null;

  // Create a prepared messages payload.
  // We must retain our reference to the original `messages` array for appending the
  // response later. Therefore, we can't simply re-use the messages array.

  // Remove any previous datetime system messages from the conversation, since we'll be
  // prepending a fresh one with the current datetime.
  let preparedMessages = JSON.parse(JSON.stringify(messages));
  preparedMessages = preparedMessages.filter((message: any) => {
    const role = message?.role;
    const content = message?.content;
    return !(role === 'system' && `${content}`.startsWith('!DATETIME:'));
  });

  // Insert a new datetime system message at the beginning of the conversation
  // to provide the model with current temporal context.
  preparedMessages.unshift(currentDatetimeSystemMessage());

  // If a system announcement message is provided, insert it at the start.
  if (systemAnnouncement) {
    preparedMessages.unshift({ role: 'system', content: systemAnnouncement });
  }

  for (let numTry = 0; numTry < retryLimit + 1; numTry += 1) {
    let llmReply = '';

    try {
      if (llmProviderName === 'openai') {
        const openaiClient = aiClient as AIClientLike;
        const payloadBody: any = {
          model,
          input: preparedMessages,
        };
        if (options.jsonResponse) {
          if (typeof options.jsonResponse === 'boolean') {
            // Freeform JSON response requested, with no schema enforcement.
            payloadBody.text = { format: { type: 'json_object' } };
          } else {
            // JSON response requested with a provided JSON schema for format enforcement.
            // It has to be the full text object, with a `format` field and everything.
            payloadBody.text = JSON.parse(JSON.stringify(options.jsonResponse));
          }
        }
        let llmResponse = await openaiClient.responses!.create(payloadBody);
        llmReply = llmResponse.output_text.trim();

        if (options.warningCallback) {
          if (llmResponse.error) {
            options.warningCallback(
              `ERROR: OpenAI API returned an error: ${llmResponse.error}`
            );
          }
          if (llmResponse.incomplete_details) {
            options.warningCallback(
              `ERROR: OpenAI API returned incomplete details: ${llmResponse.incomplete_details}`
            );
          }
        }

        // Feature: Track total token usage.
        if (llmResponse.usage) {
          updateModelTokenUsage(
            aiClient,
            llmResponse.model || model || 'unknown_model',
            llmResponse.usage.input_tokens || 0,
            llmResponse.usage.output_tokens || 0
          );
        }
      } else if (llmProviderName === 'anthropic') {
        const anthropicClient = aiClient as AIClientLike;

        // Anthropic only understands "user" and "assistant".
        // It requires "system" to all be one prompt at the beginning, and it doesn't know what
        // "developer" is at all.
        // We'll build an Anthropic system prompt by popping off beginning messages and
        // concatenating them as long as they're "system" or "developer".
        // Then, we'll go through the remaining messages, and change all "system" and "developer"
        // roles to "user".
        let anthropicSystemPrompt = '';
        while (preparedMessages.length > 0) {
          const firstMsg = preparedMessages[0];
          if (!['system', 'developer'].includes(firstMsg?.role)) {
            // Stop popping once we hit the first non-system/developer message.
            break;
          }
          anthropicSystemPrompt += `${firstMsg.content}\n\n`;
          preparedMessages.shift();
        }
        anthropicSystemPrompt = anthropicSystemPrompt.trim();

        const payloadBody: any = {
          model,
          max_tokens: 16384,
          messages: preparedMessages,
        };
        if (anthropicSystemPrompt) {
          payloadBody.system = anthropicSystemPrompt;
        }

        if (options.jsonResponse) {
          if (typeof options.jsonResponse === 'boolean') {
            // Freeform JSON response requested, with no schema enforcement.
            // We'll have to rely on parseFirstJsonValue to extract the JSON
            // from the model's response. This also means we need to "encourage"
            // the model to start its response with a curly brace, so that
            // parseFirstJsonValue can find it.
            payloadBody.messages.push({
              role: 'user',
              content:
                `Respond with a JSON object. Do not include any text before or after the JSON. ` +
                `The JSON should be the only content in your response, and it must be properly ` +
                `formatted with opening and closing curly braces. Do not put the JSON inside a ` +
                `code block or use any other formatting -- just the raw JSON, starting with an ` +
                `opening curly brace.`,
            });
          } else {
            // JSON response with a provided JSON schema for format enforcement.
            // Anthropic uses output_config.format with the schema.
            // However, it doesn't allow anything other than "schema" in the format.
            payloadBody.output_config = JSON.parse(
              JSON.stringify(options.jsonResponse)
            );
            payloadBody.output_config.format = {
              type: 'json_schema',
              schema: payloadBody.output_config.format.schema,
            };
          }
        }

        const llmResponse = await anthropicClient.messages!.create(payloadBody);

        const textBlocks = llmResponse.content.filter(
          (block: any) => block.type === 'text'
        );
        llmReply = textBlocks
          .map((block: any) => block.text)
          .join('')
          .trim();

        if (
          options.warningCallback &&
          llmResponse.stop_reason === 'max_tokens'
        ) {
          options.warningCallback(
            `ERROR: Anthropic API response was truncated (max_tokens reached)`
          );
        }

        // Feature: Track total token usage.
        if (llmResponse.usage) {
          updateModelTokenUsage(
            aiClient,
            llmResponse.model || model || 'unknown_model',
            llmResponse.usage.input_tokens || 0,
            llmResponse.usage.output_tokens || 0
          );
        }
      } else {
        throw new Error(`Unsupported LLM provider: ${llmProviderName}`);
      }

      if (!options.jsonResponse) {
        return `${llmReply}`;
      }

      return parseFirstJsonValue(llmReply);
    } catch (error) {
      if (error instanceof SyntaxError) {
        failedError = error;
        if (options.warningCallback) {
          options.warningCallback(
            `JSON decode error:\n\n${error}.\n\nRaw text of LLM Reply:\n${llmReply}\n\nRetrying (attempt ${numTry + 1} of ${retryLimit}) immediately...`
          );
        }
        continue;
      }

      if (isRetryableOpenAIError(error) || isRetryableAnthropicError(error)) {
        failedError = error;
        if (options.warningCallback) {
          options.warningCallback(
            `LLM Provider (${llmProviderName}) API error:\n\n${error}.\n\nRetrying (attempt ${numTry + 1} of ${retryLimit}) in ${retryBackoffTimeSeconds} seconds...`
          );
        }
        // Sleep for retryBackoffTimeSeconds before the next retry attempt.
        await new Promise((resolve) =>
          setTimeout(resolve, retryBackoffTimeSeconds * 1000)
        );
        continue;
      }

      throw error;
    }
  }

  if (failedError) {
    throw failedError;
  }

  throw new Error('Unknown error occurred in llmSubmit');
};
