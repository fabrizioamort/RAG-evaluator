import { LLMProviderInfo } from '@/api/client'

export function isReasoningModelName(model: string) {
    const normalized = model.toLowerCase()
    return Boolean(normalized) && (
        normalized.includes('gpt-5') ||
        normalized.includes('deepseek-v4-flash') ||
        normalized.includes('o1-') ||
        normalized.includes('o3-') ||
        normalized.includes('/o1') ||
        normalized.includes('/o3') ||
        normalized === 'o1' ||
        normalized === 'o3'
    )
}

export function supportsReasoningEffort(
    providers: LLMProviderInfo[],
    providerName: string,
    model: string
) {
    const providerInfo = providers.find(p => p.name === providerName)
    const capability = providerInfo?.model_capabilities?.[model]?.supports_reasoning_effort
    if (capability !== undefined) return capability
    return isReasoningModelName(model)
}

export function defaultModelForProvider(providers: LLMProviderInfo[], providerName: string) {
    return providers.find(p => p.name === providerName)?.models[0] || ''
}
