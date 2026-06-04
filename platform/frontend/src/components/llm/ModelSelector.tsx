import { LLMProviderInfo } from '@/api/client'

interface ModelSelectorProps {
    providers: LLMProviderInfo[]
    provider: string
    model: string
    onProviderChange: (provider: string) => void
    onModelChange: (model: string) => void
    providerLabel?: string
    modelLabel?: string
    modelPlaceholder?: string
    providerClassName?: string
    modelClassName?: string
}

const CUSTOM_VALUE = '__custom__'

export function ModelSelector({
    providers,
    provider,
    model,
    onProviderChange,
    onModelChange,
    providerLabel = 'Provider',
    modelLabel = 'Model',
    modelPlaceholder = 'Custom model',
    providerClassName = '',
    modelClassName = '',
}: ModelSelectorProps) {
    const selectedProviderInfo = providers.find(p => p.name === provider)
    const knownModels = selectedProviderInfo?.models || []
    const isKnownModel = Boolean(model) && knownModels.includes(model)
    const selectValue = isKnownModel ? model : CUSTOM_VALUE

    return (
        <>
            <div className={providerClassName || 'space-y-1.5'}>
                <label className="text-xs font-bold text-muted-foreground uppercase">{providerLabel}</label>
                <select
                    value={provider}
                    onChange={(e) => onProviderChange(e.target.value)}
                    className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                    {providers.map(p => (
                        <option key={p.name} value={p.name}>{p.display_name}</option>
                    ))}
                </select>
            </div>

            <div className={modelClassName || 'space-y-1.5'}>
                <label className="text-xs font-bold text-muted-foreground uppercase">{modelLabel}</label>
                <select
                    value={selectValue}
                    onChange={(e) => {
                        if (e.target.value === CUSTOM_VALUE) {
                            onModelChange(isKnownModel ? '' : model)
                        } else {
                            onModelChange(e.target.value)
                        }
                    }}
                    className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                    {knownModels.map(m => (
                        <option key={m} value={m}>{m}</option>
                    ))}
                    <option value={CUSTOM_VALUE}>Custom</option>
                </select>
                {selectValue === CUSTOM_VALUE && (
                    <input
                        value={model}
                        onChange={(e) => onModelChange(e.target.value)}
                        placeholder={modelPlaceholder}
                        className="mt-2 w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    />
                )}
            </div>
        </>
    )
}
