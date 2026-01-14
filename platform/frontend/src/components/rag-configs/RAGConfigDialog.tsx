import { useState, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { X, Loader2, Save, Info } from 'lucide-react'
import { api, RAGConfig, RAGConfigCreate } from '@/api/client'

interface RAGConfigDialogProps {
    isOpen: boolean
    onClose: () => void
    onSubmit: (data: RAGConfigCreate) => Promise<void>
    config?: RAGConfig // If provided, we are editing
}

export function RAGConfigDialog({ isOpen, onClose, onSubmit, config }: RAGConfigDialogProps) {
    const [name, setName] = useState('')
    const [ragType, setRagType] = useState('vector_semantic')
    const [provider, setProvider] = useState('openai')
    const [model, setModel] = useState('')
    const [parameters, setParameters] = useState<Record<string, unknown>>({})
    const [isSubmitting, setIsSubmitting] = useState(false)

    // Fetch available types and providers
    const { data: typesResponse } = useQuery({
        queryKey: ['rag-types'],
        queryFn: () => api.ragConfigs.getTypes(),
        enabled: isOpen,
    })

    const { data: providersResponse } = useQuery({
        queryKey: ['llm-providers'],
        queryFn: () => api.ragConfigs.getLLMProviders(),
        enabled: isOpen,
    })

    const types = typesResponse?.data || []
    const providers = providersResponse?.data || []
    const selectedTypeInfo = types.find(t => t.name === ragType)
    const selectedProviderInfo = providers.find(p => p.name === provider)

    useEffect(() => {
        if (config) {
            setName(config.name)
            setRagType(config.rag_type)
            setProvider(config.llm_provider)
            setModel(config.llm_model)
            setParameters(config.parameters)
        } else {
            setName('')
            setRagType('vector_semantic')
            setProvider('openai')
            setModel('gpt-4o-mini')
            setParameters({})
        }
    }, [config, isOpen])

    // Update model when provider changes if none selected
    useEffect(() => {
        if (!config && selectedProviderInfo && selectedProviderInfo.models.length > 0) {
            setModel(selectedProviderInfo.models[0])
        }
    }, [provider, providersResponse, config, selectedProviderInfo])

    // Initialize default parameters when ragType changes
    useEffect(() => {
        if (!config && selectedTypeInfo) {
            const defaults: Record<string, unknown> = {}
            selectedTypeInfo.parameters.forEach(p => {
                if (p.default !== undefined) defaults[p.name] = p.default
            })
            setParameters(defaults)
        }
    }, [ragType, typesResponse, config, selectedTypeInfo])

    if (!isOpen) return null

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        if (!name.trim()) return

        setIsSubmitting(true)
        try {
            await onSubmit({
                name: name.trim(),
                rag_type: ragType,
                llm_provider: provider,
                llm_model: model,
                parameters,
            })
            onClose()
        } catch (error) {
            console.error('Failed to save RAG config:', error)
        } finally {
            setIsSubmitting(false)
        }
    }

    const handleParamChange = (name: string, value: unknown) => {
        setParameters(prev => ({ ...prev, [name]: value }))
    }

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <div className="absolute inset-0 bg-background/80 backdrop-blur-sm animate-in fade-in" onClick={onClose} />

            <div className="relative w-full max-w-3xl max-h-[90vh] overflow-y-auto rounded-xl border border-border bg-card p-8 shadow-2xl animate-in zoom-in-95 duration-200">
                <div className="flex items-center justify-between mb-8">
                    <div>
                        <h2 className="text-2xl font-bold tracking-tight">
                            {config ? 'Edit RAG Configuration' : 'Create RAG Configuration'}
                        </h2>
                        <p className="text-sm text-muted-foreground mt-1">
                            Set up the retrieval and generation parameters for your evaluation.
                        </p>
                    </div>
                    <button onClick={onClose} className="rounded-full p-2 text-muted-foreground hover:bg-muted transition-colors">
                        <X className="h-5 w-5" />
                    </button>
                </div>

                <form onSubmit={handleSubmit} className="space-y-8">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                        {/* Basic Info */}
                        <div className="space-y-6">
                            <h3 className="text-sm font-bold uppercase tracking-wider text-primary border-b border-border pb-2">Basic Info</h3>

                            <div className="space-y-2">
                                <label className="text-sm font-semibold">Config Name</label>
                                <input
                                    type="text"
                                    value={name}
                                    onChange={(e) => setName(e.target.value)}
                                    placeholder="e.g., GPT-4o Semantic v1"
                                    className="w-full rounded-lg border border-input bg-background px-4 py-2.5 text-sm focus:ring-2 focus:ring-primary/50 outline-none transition-all"
                                    required
                                />
                            </div>

                            <div className="space-y-2">
                                <label className="text-sm font-semibold">RAG Implementation</label>
                                <select
                                    value={ragType}
                                    onChange={(e) => setRagType(e.target.value)}
                                    className="w-full rounded-lg border border-input bg-background px-4 py-2.5 text-sm focus:ring-2 focus:ring-primary/50 outline-none transition-all"
                                >
                                    {types.map(t => (
                                        <option key={t.name} value={t.name}>{t.display_name}</option>
                                    ))}
                                </select>
                                {selectedTypeInfo && (
                                    <p className="text-[11px] text-muted-foreground flex items-center gap-1 mt-1">
                                        <Info className="h-3 w-3" />
                                        {selectedTypeInfo.description}
                                    </p>
                                )}
                            </div>

                            <div className="space-y-4 pt-4">
                                <h3 className="text-sm font-bold uppercase tracking-wider text-primary border-b border-border pb-2">LLM Settings</h3>

                                <div className="space-y-2">
                                    <label className="text-sm font-semibold">Provider</label>
                                    <select
                                        value={provider}
                                        onChange={(e) => setProvider(e.target.value)}
                                        className="w-full rounded-lg border border-input bg-background px-4 py-2.5 text-sm focus:ring-2 focus:ring-primary/50 outline-none transition-all"
                                    >
                                        {providers.map(p => (
                                            <option key={p.name} value={p.name}>{p.display_name}</option>
                                        ))}
                                    </select>
                                </div>

                                <div className="space-y-2">
                                    <label className="text-sm font-semibold">Model</label>
                                    <select
                                        value={model}
                                        onChange={(e) => setModel(e.target.value)}
                                        className="w-full rounded-lg border border-input bg-background px-4 py-2.5 text-sm focus:ring-2 focus:ring-primary/50 outline-none transition-all"
                                    >
                                        {selectedProviderInfo?.models.map(m => (
                                            <option key={m} value={m}>{m}</option>
                                        ))}
                                    </select>
                                </div>
                            </div>
                        </div>

                        {/* RAG Parameters */}
                        <div className="space-y-6">
                            <h3 className="text-sm font-bold uppercase tracking-wider text-primary border-b border-border pb-2">RAG Parameters</h3>

                            <div className="space-y-4">
                                {selectedTypeInfo?.parameters.length === 0 && (
                                    <p className="text-sm text-muted-foreground italic">No parameters available for this type.</p>
                                )}
                                {selectedTypeInfo?.parameters.map(param => (
                                    <div key={param.name} className="space-y-2">
                                        <div className="flex items-center justify-between">
                                            <label className="text-sm font-semibold capitalize">{param.name.replace('_', ' ')}</label>
                                            {param.required && <span className="text-[10px] font-bold text-destructive uppercase tracking-widest">Required</span>}
                                        </div>

                                        {param.type === 'string' && param.choices ? (
                                            <select
                                                value={(parameters[param.name] as string) ?? param.default}
                                                onChange={(e) => handleParamChange(param.name, e.target.value)}
                                                className="w-full rounded-lg border border-input bg-background px-4 py-2 text-sm focus:ring-2 focus:ring-primary/50 outline-none transition-all"
                                            >
                                                {param.choices.map(c => <option key={c} value={c}>{c}</option>)}
                                            </select>
                                        ) : param.type === 'boolean' ? (
                                            <div className="flex items-center gap-2">
                                                <input
                                                    type="checkbox"
                                                    checked={!!((parameters[param.name] as boolean) ?? param.default)}
                                                    onChange={(e) => handleParamChange(param.name, e.target.checked)}
                                                    className="h-4 w-4 rounded border-input text-primary focus:ring-primary/50"
                                                />
                                                <span className="text-sm text-muted-foreground">Enabled</span>
                                            </div>
                                        ) : (
                                            <input
                                                type={param.type === 'integer' || param.type === 'float' ? 'number' : 'text'}
                                                value={(parameters[param.name] as string | number) ?? param.default ?? ''}
                                                onChange={(e) => handleParamChange(param.name, param.type === 'integer' || param.type === 'float' ? Number(e.target.value) : e.target.value)}
                                                step={param.type === 'float' ? '0.1' : '1'}
                                                min={param.min_value}
                                                max={param.max_value}
                                                className="w-full rounded-lg border border-input bg-background px-4 py-2 text-sm focus:ring-2 focus:ring-primary/50 outline-none transition-all"
                                            />
                                        )}
                                        <p className="text-[11px] text-muted-foreground">{param.description}</p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>

                    <div className="flex justify-end gap-3 pt-8 border-t border-border">
                        <button type="button" onClick={onClose} className="rounded-lg px-6 py-2.5 text-sm font-semibold hover:bg-muted transition-colors">
                            Cancel
                        </button>
                        <button
                            type="submit"
                            disabled={isSubmitting || !name.trim()}
                            className="flex items-center gap-2 rounded-lg bg-primary px-8 py-2.5 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-all shadow-md active:scale-95 disabled:opacity-50"
                        >
                            {isSubmitting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
                            {config ? 'Update Configuration' : 'Create Configuration'}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    )
}
