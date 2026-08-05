/**
 * Minimal interface describing an OpenAI client that can submit requests via
 * `responses.create`. Accepting this interface instead of the concrete SDK
 * class keeps callers decoupled from the SDK version and makes testing easier.
 * Note that the `tokenUsage` field is monkeypatched onto the client object
 * and may not be present in the original SDK.
 */
export interface OpenAIClientLike {
  responses: {
    create: (args: any, options?: any) => Promise<any> | any;
  };
  messages?: never;
  tokenUsage?: TokenUsage;
}

/**
 * Minimal interface describing an Anthropic client that can submit requests via
 * `messages.create`. Accepting this interface instead of the concrete SDK
 * class keeps callers decoupled from the SDK version and makes testing easier.
 * Note that the `tokenUsage` field is monkeypatched onto the client object
 * and may not be present in the original SDK.
 */
export interface AnthropicClientLike {
  messages: {
    create: (args: any, options?: any) => Promise<any> | any;
  };
  responses?: never;
  tokenUsage?: TokenUsage;
}

/**
 * Represents the count of tokens used in an LLM request or response.
 */
export interface TokenCounts {
  total: number;
  input: number;
  output: number;
}

/**
 * A structure representing the token usage statistics for an LLM client, including
 * total counts for all models and breakdowns by individual model.
 */
export interface TokenUsage {
  allModels: TokenCounts;
  byModel: Record<string, TokenCounts>;
}

export type AIClientLike = OpenAIClientLike | AnthropicClientLike;

export type LLMProviderName = 'openai' | 'anthropic';
export type LLMModelTier = 'cheap' | 'standard' | 'smart' | 'vision';

/**
 * Identifies the LLM provider based on the presence of specific methods in the client object.
 * @param aiClient An instance of either an OpenAI-like client or an Anthropic-like client.
 * @returns A string indicating the LLM provider: either 'openai' or 'anthropic'.
 */
export const identifyLLMProvider = (
  aiClient: AIClientLike
): LLMProviderName => {
  if ('messages' in aiClient && aiClient.messages != null) {
    return 'anthropic';
  }
  return 'openai';
};

export const GPT_MODEL_CHEAP = 'gpt-4.1-nano';
export const GPT_MODEL_STANDARD = 'gpt-4.1';
export const GPT_MODEL_SMART = 'gpt-4.1';
export const GPT_MODEL_VISION = 'gpt-4.1';

export const CLAUDE_MODEL_CHEAP = 'claude-haiku-4-5-20251001';
export const CLAUDE_MODEL_STANDARD = 'claude-sonnet-4-6';
export const CLAUDE_MODEL_SMART = 'claude-opus-4-6';
export const CLAUDE_MODEL_VISION = 'claude-opus-4-6';

export const LLM_RETRY_LIMIT_DEFAULT = 5;
export const LLM_RETRY_BACKOFF_TIME_SECONDS_DEFAULT = 30;

/**
 * Gets the appropriate model name for a given LLM provider and model tier.
 * @param provider The LLM provider, either 'openai' or 'anthropic'.
 * @param tier The desired model tier, one of 'cheap', 'standard', 'smart', or 'vision'.
 * @returns The model name string corresponding to the specified provider and tier.
 * @throws {Error} If an unsupported provider or tier is specified.
 * @remarks This function centralizes the mapping of provider and tier combinations to
 * specific model names, making it easier to manage and update model selections in one place.
 */
export const getModelName = (
  provider: LLMProviderName,
  tier: LLMModelTier
): string => {
  switch (provider) {
    case 'openai':
      switch (tier) {
        case 'cheap':
          return GPT_MODEL_CHEAP;
        case 'standard':
          return GPT_MODEL_STANDARD;
        case 'smart':
          return GPT_MODEL_SMART;
        case 'vision':
          return GPT_MODEL_VISION;
      }
    case 'anthropic':
      switch (tier) {
        case 'cheap':
          return CLAUDE_MODEL_CHEAP;
        case 'standard':
          return CLAUDE_MODEL_STANDARD;
        case 'smart':
          return CLAUDE_MODEL_SMART;
        case 'vision':
          return CLAUDE_MODEL_VISION;
      }
  }
  throw new Error(`Unsupported provider or tier: ${provider}, ${tier}`);
};
