import React, { useState, useEffect, useCallback, useMemo } from 'react'
import { Play, Loader2, Database, FileText, ChevronRight, ChevronLeft, LucideIcon, Layers, Calendar, CheckSquare, Square, Info } from 'lucide-react'
import { api, KnowledgeBase, TestSet, KnowledgeBaseIndex, EvaluationCreate, RAGTypeInfo, RAGTypeParameter, LLMProviderInfo } from '@/api/client'
import { cn } from '@/lib/utils'
import { DialogShell } from '@/components/ui/DialogShell'
import { ModelSelector } from '@/components/llm/ModelSelector'
import { defaultModelForProvider, supportsReasoningEffort } from '@/lib/llm-models'

interface StartEvaluationWizardProps {
    projectId: string
    isOpen: boolean
    onClose: () => void
    onStarted: (evaluationId: string) => void
    initialKnowledgeBaseId?: string
    initialIndexId?: string
    initialTestSetId?: string
}

type Step = 'testset' | 'kb' | 'index' | 'overrides' | 'metrics' | 'review'

const AVAILABLE_METRICS = [
    { id: 'faithfulness', name: 'Faithfulness', description: 'Consistency of the answer with the retrieved context.' },
    { id: 'relevancy', name: 'Answer Relevancy', description: 'Relevance of the answer to the user question.' },
    { id: 'precision', name: 'Contextual Precision', description: 'Quality of the top-ranked retrieved documents.' },
    { id: 'recall', name: 'Contextual Recall', description: 'Whether context contains answer to the question.' },
    { id: 'g_eval', name: 'G-Eval (Correctness)', description: 'LLM-based evaluation of factual correctness vs expected answer.' },
]

function timeAgo(dateString: string) {
    const date = new Date(dateString)
    const now = new Date()
    const seconds = Math.floor((now.getTime() - date.getTime()) / 1000)

    if (seconds < 60) return 'just now'
    const minutes = Math.floor(seconds / 60)
    if (minutes < 60) return `${minutes}m ago`
    const hours = Math.floor(minutes / 60)
    if (hours < 24) return `${hours}h ago`
    const days = Math.floor(hours / 24)
    return `${days}d ago`
}

export function StartEvaluationWizard({
    projectId,
    isOpen,
    onClose,
    onStarted,
    initialKnowledgeBaseId,
    initialIndexId,
    initialTestSetId,
}: StartEvaluationWizardProps) {
    const [step, setStep] = useState<Step>('testset')
    const [kbs, setKbs] = useState<KnowledgeBase[]>([])
    const [testSets, setTestSets] = useState<TestSet[]>([])
    const [indexes, setIndexes] = useState<KnowledgeBaseIndex[]>([])
    const [ragTypes, setRagTypes] = useState<RAGTypeInfo[]>([])

    const [selectedKb, setSelectedKb] = useState<string>('')
    const [selectedTestSet, setSelectedTestSet] = useState<string>('')
    const [selectedIndex, setSelectedIndex] = useState<string>('')
    const [selectedMetrics, setSelectedMetrics] = useState<string[]>(AVAILABLE_METRICS.map(m => m.id))
    const [evaluationName, setEvaluationName] = useState<string>('')
    const [includeReason, setIncludeReason] = useState(true)
    const [providers, setProviders] = useState<LLMProviderInfo[]>([])
    const [queryModel, setQueryModel] = useState('')
    const [queryProvider, setQueryProvider] = useState('')
    const [queryReasoningEffort, setQueryReasoningEffort] = useState('')
    const [judgeModel, setJudgeModel] = useState('')
    const [judgeProvider, setJudgeProvider] = useState('')
    const [queryTopK, setQueryTopK] = useState(5)
    const [queryParams, setQueryParams] = useState<Record<string, unknown>>({})

    const [isLoading, setIsLoading] = useState(false)
    const [isLoadingIndexes, setIsLoadingIndexes] = useState(false)
    const [isStarting, setIsStarting] = useState(false)

    const loadData = useCallback(async () => {
        setIsLoading(true)
        try {
            const [kbRes, tsRes, ragTypeRes, providerRes] = await Promise.all([
                api.knowledgeBases.list(projectId),
                api.testSets.list(projectId),
                api.ragConfigs.getTypes(),
                api.ragConfigs.getLLMProviders(),
            ])
            setKbs(kbRes.data.items)
            setTestSets(tsRes.data.items)
            setRagTypes(ragTypeRes.data)
            setProviders(providerRes.data)
        } catch (error) {
            console.error('Failed to load evaluation requirements:', error)
        } finally {
            setIsLoading(false)
        }
    }, [projectId])

    useEffect(() => {
        if (isOpen && projectId) {
            loadData()
            setStep('testset') // Reset step
            setSelectedKb(initialKnowledgeBaseId || '')
            setSelectedTestSet(initialTestSetId || '')
            setSelectedIndex(initialIndexId || '')
            setSelectedMetrics(AVAILABLE_METRICS.map(m => m.id))
            setEvaluationName('')
            setIncludeReason(true)
            setQueryModel('')
            setQueryProvider('')
            setQueryReasoningEffort('')
            setJudgeModel('')
            setJudgeProvider('')
            setQueryTopK(5)
            setQueryParams({})
            setIndexes([])
        }
    }, [isOpen, projectId, loadData, initialKnowledgeBaseId, initialIndexId, initialTestSetId])

    // Load indexes when KB is selected
    useEffect(() => {
        if (selectedKb) {
            const loadIndexes = async () => {
                setIsLoadingIndexes(true)
                try {
                    const res = await api.indexes.list({ kb_id: selectedKb, status: 'ready' })
                    setIndexes(res.data.items)
                } catch (e) {
                    console.error('Failed to load indexes', e)
                } finally {
                    setIsLoadingIndexes(false)
                }
            }
            loadIndexes()
        } else {
            setIndexes([])
        }
    }, [selectedKb])

    const selectedIndexInfo = indexes.find(i => i.id === selectedIndex)
    const indexSnapshot = (selectedIndexInfo?.config_snapshot ?? {}) as {
        rag_type?: string
        llm_model?: string
        llm_reasoning_effort?: string | null
        parameters?: Record<string, unknown>
    }
    const selectedRagTypeInfo = ragTypes.find(t => t.name === indexSnapshot.rag_type)
    const buildParamDefs = useMemo(
        () => selectedRagTypeInfo?.parameters.filter(p => p.phase === 'build' && !p.platform_managed) || [],
        [selectedRagTypeInfo]
    )
    const queryParamDefs = useMemo(
        () => selectedRagTypeInfo?.parameters.filter(p => p.phase === 'query' && !p.platform_managed) || [],
        [selectedRagTypeInfo]
    )

    useEffect(() => {
        if (!selectedIndexInfo) return

        const snapshot = (selectedIndexInfo.config_snapshot ?? {}) as {
            llm_model?: string
            llm_provider?: string
            llm_reasoning_effort?: string | null
            parameters?: Record<string, unknown>
        }
        const defaultModel = snapshot.llm_model || 'gpt-4o-mini'
        const defaultProvider = snapshot.llm_provider || 'openai'
        const defaults: Record<string, unknown> = {}
        queryParamDefs.forEach(param => {
            const snapshotValue = snapshot.parameters?.[param.name]
            if (snapshotValue !== undefined) defaults[param.name] = snapshotValue
            else if (param.default !== undefined) defaults[param.name] = param.default
        })

        setQueryModel(defaultModel)
        setQueryProvider(defaultProvider)
        setQueryReasoningEffort(snapshot.llm_reasoning_effort || '')
        setJudgeModel(defaultModel)
        setJudgeProvider(defaultProvider)
        setQueryTopK(5)
        setQueryParams(defaults)
    }, [selectedIndexInfo, queryParamDefs])

    const queryModelSupportsReasoningEffort = supportsReasoningEffort(
        providers,
        queryProvider,
        queryModel
    )

    useEffect(() => {
        if (providers.length > 0 && !queryModelSupportsReasoningEffort) {
            setQueryReasoningEffort('')
        }
    }, [providers.length, queryModelSupportsReasoningEffort])

    const handleQueryProviderChange = (nextProvider: string) => {
        setQueryProvider(nextProvider)
        const nextModel = defaultModelForProvider(providers, nextProvider)
        if (nextModel) setQueryModel(nextModel)
        setQueryReasoningEffort('')
    }

    const handleJudgeProviderChange = (nextProvider: string) => {
        setJudgeProvider(nextProvider)
        const nextModel = defaultModelForProvider(providers, nextProvider)
        if (nextModel) setJudgeModel(nextModel)
    }

    const handleStart = async () => {
        setIsStarting(true)
        try {
            const data: EvaluationCreate = {
                name: evaluationName || undefined,
                test_set_id: selectedTestSet,
                knowledge_base_index_id: selectedIndex,
                metric_names: selectedMetrics,
                include_reason: includeReason,
                query_overrides: {
                    llm_model: queryModel || undefined,
                    llm_provider: queryProvider || undefined,
                    llm_reasoning_effort: queryModelSupportsReasoningEffort ? queryReasoningEffort || undefined : undefined,
                    top_k: queryTopK,
                    parameters: queryParams,
                },
                eval_judge_model: judgeModel || undefined,
                eval_judge_provider: judgeProvider || undefined,
            }
            const res = await api.evaluations.create(data)
            onStarted(res.data.id)
            onClose()
        } catch (error) {
            console.error('Failed to start evaluation:', error)
            alert('Failed to start evaluation: ' + (error as Error).message)
        } finally {
            setIsStarting(false)
        }
    }

    if (!isOpen) return null

    const isIndexLocked = Boolean(initialIndexId)
    const steps: { id: Step; label: string; icon: LucideIcon }[] = [
        { id: 'testset', label: 'Test Set', icon: FileText },
        ...(!isIndexLocked ? [
            { id: 'kb' as Step, label: 'Knowledge Base', icon: Database },
            { id: 'index' as Step, label: 'Index', icon: Layers },
        ] : []),
        { id: 'overrides', label: 'Query', icon: Info },
        { id: 'metrics', label: 'Metrics', icon: CheckSquare },
        { id: 'review', label: 'Review', icon: Play }
    ]

    const currentStepIndex = steps.findIndex(s => s.id === step)
    const previousStep = currentStepIndex > 0 ? steps[currentStepIndex - 1]?.id : undefined
    const nextStep = currentStepIndex >= 0 ? steps[currentStepIndex + 1]?.id : undefined

    const getParamValue = (param: RAGTypeParameter) => {
        return indexSnapshot.parameters?.[param.name] ?? param.default ?? ''
    }

    const setQueryParamValue = (param: RAGTypeParameter, value: unknown) => {
        setQueryParams(prev => ({ ...prev, [param.name]: value }))
    }

    const renderQueryParam = (param: RAGTypeParameter) => (
        <div key={param.name} className="space-y-1.5">
            <label className="text-xs font-bold text-muted-foreground uppercase">{param.name.replace(/_/g, ' ')}</label>
            {param.type === 'boolean' ? (
                <input
                    type="checkbox"
                    checked={Boolean(queryParams[param.name] ?? param.default)}
                    onChange={(e) => setQueryParamValue(param, e.target.checked)}
                    className="h-4 w-4 rounded border-input text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                />
            ) : param.type === 'string' && param.choices ? (
                <select
                    value={(queryParams[param.name] as string) ?? param.default ?? ''}
                    onChange={(e) => setQueryParamValue(param, e.target.value)}
                    className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                    {param.choices.map(choice => <option key={choice} value={choice}>{choice}</option>)}
                </select>
            ) : (
                <input
                    type={param.type === 'integer' || param.type === 'float' ? 'number' : 'text'}
                    value={(queryParams[param.name] as string | number) ?? param.default ?? ''}
                    min={param.min_value}
                    max={param.max_value}
                    step={param.type === 'float' ? '0.1' : '1'}
                    onChange={(e) => setQueryParamValue(param, param.type === 'integer' || param.type === 'float' ? Number(e.target.value) : e.target.value)}
                    className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                />
            )}
            <p className="text-[11px] text-muted-foreground">{param.description}</p>
        </div>
    )

    return (
        <DialogShell
            isOpen={isOpen}
            title="Launch Evaluation"
            icon={<Play className="h-5 w-5 text-primary fill-primary" />}
            onClose={onClose}
            size="xl"
            closeDisabled={isStarting}
            bodyClassName="p-0"
            footer={(
                <div className="flex items-center justify-between">
                    <button
                        onClick={() => {
                            if (previousStep) setStep(previousStep)
                        }}
                        disabled={!previousStep || isStarting}
                        className="flex items-center gap-2 px-4 py-2 text-sm font-medium hover:bg-accent rounded-lg transition-colors disabled:opacity-30"
                    >
                        <ChevronLeft className="h-4 w-4" /> Back
                    </button>

                    <div className="flex gap-3">
                        <button
                            onClick={onClose}
                            disabled={isStarting}
                            className="px-6 py-2 text-sm font-medium hover:bg-accent rounded-lg transition-colors disabled:opacity-50"
                        >
                            Cancel
                        </button>

                        {step === 'review' ? (
                            <button
                                onClick={handleStart}
                                disabled={isStarting}
                                className="flex items-center gap-2 rounded-lg bg-primary px-8 py-2 text-sm font-bold text-primary-foreground hover:bg-primary/90 transition-all shadow-lg shadow-primary/20 active:scale-95 disabled:opacity-50"
                            >
                                {isStarting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4 fill-current" />}
                                {isStarting ? 'Launching...' : 'Start Now'}
                            </button>
                        ) : (
                            <button
                                onClick={() => {
                                    if (nextStep) setStep(nextStep)
                                }}
                                disabled={
                                    !nextStep ||
                                    (step === 'testset' && !selectedTestSet) ||
                                    (step === 'kb' && !selectedKb) ||
                                    (step === 'index' && !selectedIndex) ||
                                    (step === 'overrides' && (!queryModel || !judgeModel || queryTopK < 1))
                                }
                                className="flex items-center gap-2 rounded-lg bg-primary px-8 py-2 text-sm font-bold text-primary-foreground hover:bg-primary/90 transition-all disabled:opacity-50"
                            >
                                Continue <ChevronRight className="h-4 w-4" />
                            </button>
                        )}
                    </div>
                </div>
            )}
        >
            {/* Stepper */}
                <div className="flex items-center justify-center bg-muted/30 px-6 py-4 border-b border-border">
                    {steps.map((s, i) => (
                        <React.Fragment key={s.id}>
                            <div className={cn(
                                "flex flex-col items-center gap-1",
                                step === s.id ? "text-primary" : "text-muted-foreground"
                            )}>
                                <div className={cn(
                                    "flex h-8 w-8 items-center justify-center rounded-full border-2 text-xs font-bold transition-all",
                                    step === s.id ? "border-primary bg-primary text-primary-foreground" : "border-muted-foreground/30 bg-background"
                                )}>
                                    {i + 1}
                                </div>
                                <span className="text-[10px] font-bold uppercase tracking-wider">{s.label}</span>
                            </div>
                            {i < steps.length - 1 && (
                                <div className="mx-4 h-px w-12 bg-border" />
                            )}
                        </React.Fragment>
                    ))}
                </div>

                {/* Content */}
                <div className="p-8 min-h-[300px]">
                    {isLoading ? (
                        <div className="flex h-[300px] flex-col items-center justify-center gap-4">
                            <Loader2 className="h-10 w-10 animate-spin text-primary" />
                            <p className="text-muted-foreground font-medium">Preparing wizard...</p>
                        </div>
                    ) : (
                        <>
                            {step === 'testset' && (
                                <div className="space-y-4 animate-in fade-in slide-in-from-right-4 duration-300">
                                    <div className="mb-4">
                                        <h3 className="text-lg font-bold">Select Test Set</h3>
                                        <p className="text-sm text-muted-foreground">Choose the questions to run against the RAG.</p>
                                    </div>
                                    <div className="grid gap-3 max-h-[300px] overflow-y-auto pr-2">
                                        {testSets.map(ts => (
                                            <button
                                                key={ts.id}
                                                onClick={() => setSelectedTestSet(ts.id)}
                                                className={cn(
                                                    "flex items-center justify-between rounded-xl border p-4 text-left transition-all",
                                                    selectedTestSet === ts.id
                                                        ? "border-primary bg-primary/5 ring-1 ring-primary"
                                                        : "border-border hover:border-primary/50 hover:bg-accent"
                                                )}
                                            >
                                                <div>
                                                    <p className="font-bold">{ts.name}</p>
                                                    <p className="text-xs text-muted-foreground mt-1">{ts.test_case_count} test cases</p>
                                                </div>
                                                {selectedTestSet === ts.id && <div className="h-2 w-2 rounded-full bg-primary" />}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {step === 'kb' && (
                                <div className="space-y-4 animate-in fade-in slide-in-from-right-4 duration-300">
                                    <div className="mb-4">
                                        <h3 className="text-lg font-bold">Select Knowledge Base</h3>
                                        <p className="text-sm text-muted-foreground">Choose the knowledge base containing your documents.</p>
                                    </div>
                                    <div className="grid gap-3 max-h-[300px] overflow-y-auto pr-2">
                                        {kbs.map(kb => (
                                            <button
                                                key={kb.id}
                                                onClick={() => setSelectedKb(kb.id)}
                                                className={cn(
                                                    "flex items-center justify-between rounded-xl border p-4 text-left transition-all",
                                                    selectedKb === kb.id
                                                        ? "border-primary bg-primary/5 ring-1 ring-primary"
                                                        : "border-border hover:border-primary/50 hover:bg-accent"
                                                )}
                                            >
                                                <div>
                                                    <p className="font-bold">{kb.name}</p>
                                                    <p className="text-xs text-muted-foreground mt-1">Version {kb.current_version} • {kb.document_count} docs</p>
                                                </div>
                                                {selectedKb === kb.id && <div className="h-2 w-2 rounded-full bg-primary" />}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {step === 'index' && (
                                <div className="space-y-4 animate-in fade-in slide-in-from-right-4 duration-300">
                                    <div className="mb-4">
                                        <h3 className="text-lg font-bold">Select Index</h3>
                                        <p className="text-sm text-muted-foreground">Choose the specific index (RAG configuration) to evaluate.</p>
                                    </div>

                                    {isLoadingIndexes ? (
                                        <div className="flex justify-center py-8">
                                            <Loader2 className="h-8 w-8 animate-spin text-primary" />
                                        </div>
                                    ) : indexes.length === 0 ? (
                                        <div className="text-center py-8 border border-dashed rounded-lg bg-muted/20">
                                            <p className="text-muted-foreground">No ready indexes found for this Knowledge Base.</p>
                                            <p className="text-xs text-muted-foreground mt-1">Go to the Knowledge Base to create an index.</p>
                                        </div>
                                    ) : (
                                        <div className="grid gap-3 max-h-[300px] overflow-y-auto pr-2">
                                            {indexes.map(idx => (
                                                <button
                                                    key={idx.id}
                                                    onClick={() => {
                                                        setSelectedIndex(idx.id)
                                                        // Suggest a name
                                                        const timestamp = new Date().toLocaleString([], {
                                                            month: 'short',
                                                            day: 'numeric',
                                                            hour: '2-digit',
                                                            minute: '2-digit'
                                                        });
                                                        setEvaluationName(`${idx.name} - ${timestamp}`);
                                                    }}
                                                    className={cn(
                                                        "flex items-center justify-between rounded-xl border p-4 text-left transition-all",
                                                        selectedIndex === idx.id
                                                            ? "border-primary bg-primary/5 ring-1 ring-primary"
                                                            : "border-border hover:border-primary/50 hover:bg-accent"
                                                    )}
                                                >
                                                    <div>
                                                        <p className="font-bold">{idx.name}</p>
                                                        <div className="flex items-center gap-2 mt-1">
                                                            <span className="text-xs bg-muted px-1.5 py-0.5 rounded text-muted-foreground">
                                                                {idx.storage_type}
                                                            </span>
                                                            <span className="text-xs text-muted-foreground flex items-center">
                                                                <Calendar className="h-3 w-3 mr-1" />
                                                                {timeAgo(idx.created_at)}
                                                            </span>
                                                        </div>
                                                    </div>
                                                    {selectedIndex === idx.id && <div className="h-2 w-2 rounded-full bg-primary" />}
                                                </button>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            )}

                            {step === 'overrides' && (
                                <div className="space-y-5 animate-in fade-in slide-in-from-right-4 duration-300">
                                    <div>
                                        <h3 className="text-lg font-bold">Query Settings</h3>
                                        <p className="text-sm text-muted-foreground">Adjust runtime settings for this evaluation without rebuilding the index.</p>
                                    </div>

                                    <div className="grid gap-4 sm:grid-cols-3">
                                        <ModelSelector
                                            providers={providers}
                                            provider={queryProvider}
                                            model={queryModel}
                                            onProviderChange={handleQueryProviderChange}
                                            onModelChange={setQueryModel}
                                            providerLabel="Generation Provider"
                                            modelLabel="RAG Model"
                                            modelPlaceholder="Enter a RAG model"
                                        />
                                        <div className="space-y-1.5">
                                            <label className="text-xs font-bold text-muted-foreground uppercase">Top K</label>
                                            <input
                                                type="number"
                                                min={1}
                                                max={50}
                                                value={queryTopK}
                                                onChange={(e) => setQueryTopK(Number(e.target.value))}
                                                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                                            />
                                        </div>
                                        {queryModelSupportsReasoningEffort && (
                                            <div className="space-y-1.5 sm:col-span-3">
                                                <label className="text-xs font-bold text-muted-foreground uppercase">RAG Reasoning Effort</label>
                                                <select
                                                    value={queryReasoningEffort}
                                                    onChange={(e) => setQueryReasoningEffort(e.target.value)}
                                                    className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                                                >
                                                    <option value="">Default (model decides)</option>
                                                    <option value="low">Low - faster, less thorough</option>
                                                    <option value="medium">Medium - balanced</option>
                                                    <option value="high">High - slower, most thorough</option>
                                                </select>
                                            </div>
                                        )}
                                        <ModelSelector
                                            providers={providers}
                                            provider={judgeProvider}
                                            model={judgeModel}
                                            onProviderChange={handleJudgeProviderChange}
                                            onModelChange={setJudgeModel}
                                            providerLabel="Judge Provider"
                                            modelLabel="Judge Model"
                                            modelPlaceholder="Enter a judge model"
                                            modelClassName="space-y-1.5 sm:col-span-2"
                                        />
                                    </div>
                                    <p className="text-[11px] text-muted-foreground">
                                        Generation and judge can use different providers. For RLM, orchestrator/worker models appear below as query parameters.
                                    </p>

                                    {queryParamDefs.length > 0 && (
                                        <div className="grid gap-4 rounded-lg border border-border p-4">
                                            {queryParamDefs.map(renderQueryParam)}
                                        </div>
                                    )}

                                    {buildParamDefs.length > 0 && (
                                        <div className="rounded-lg border border-border bg-muted/20 p-4">
                                            <p className="mb-3 text-xs font-bold uppercase text-muted-foreground">Frozen build parameters</p>
                                            <div className="grid gap-2 sm:grid-cols-2">
                                                {buildParamDefs.map(param => (
                                                    <div key={param.name} className="rounded-md bg-background p-3">
                                                        <p className="text-[10px] font-bold uppercase text-muted-foreground">{param.name.replace(/_/g, ' ')}</p>
                                                        <p className="mt-1 truncate text-sm font-medium" title={String(getParamValue(param))}>
                                                            {String(getParamValue(param))}
                                                        </p>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            )}

                            {step === 'metrics' && (
                                <div className="space-y-4 animate-in fade-in slide-in-from-right-4 duration-300">
                                    <div className="mb-4">
                                        <h3 className="text-lg font-bold">Select Metrics</h3>
                                        <p className="text-sm text-muted-foreground">Choose which evaluations to perform on the RAG outputs.</p>
                                    </div>

                                    <div className="grid gap-3">
                                        {AVAILABLE_METRICS.map(metric => {
                                            const isSelected = selectedMetrics.includes(metric.id);
                                            return (
                                                <button
                                                    key={metric.id}
                                                    onClick={() => {
                                                        if (isSelected) {
                                                            setSelectedMetrics(selectedMetrics.filter(id => id !== metric.id));
                                                        } else {
                                                            setSelectedMetrics([...selectedMetrics, metric.id]);
                                                        }
                                                    }}
                                                    className={cn(
                                                        "flex items-start gap-4 rounded-xl border p-4 text-left transition-all",
                                                        isSelected
                                                            ? "border-primary bg-primary/5 ring-1 ring-primary"
                                                            : "border-border hover:border-primary/50 hover:bg-accent"
                                                    )}
                                                >
                                                    <div className={cn(
                                                        "mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded border transition-colors",
                                                        isSelected ? "bg-primary border-primary text-primary-foreground" : "border-muted-foreground/30"
                                                    )}>
                                                        {isSelected ? <CheckSquare className="h-4 w-4" /> : <Square className="h-4 w-4" />}
                                                    </div>
                                                    <div className="flex-1">
                                                        <div className="flex items-center gap-2">
                                                            <p className="font-bold">{metric.name}</p>
                                                            {metric.id === 'g_eval' && (
                                                                <span className="text-[10px] bg-primary/20 text-primary px-1.5 py-0.5 rounded font-bold uppercase tracking-wider">New</span>
                                                            )}
                                                        </div>
                                                        <p className="text-xs text-muted-foreground mt-0.5">{metric.description}</p>
                                                    </div>
                                                </button>
                                            );
                                        })}
                                    </div>

                                    <div className="mt-4 p-3 rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-100 dark:border-blue-900/30 flex gap-3">
                                        <Info className="h-5 w-5 text-blue-500 shrink-0 mt-0.5" />
                                        <p className="text-xs text-blue-700 dark:text-blue-300 leading-relaxed">
                                            Metrics use the selected judge model ({judgeModel || 'matching the RAG model'}). Selecting more metrics increases total evaluation cost and duration.
                                        </p>
                                    </div>

                                    <label className="mt-4 flex items-start gap-3 rounded-lg border border-border bg-muted/20 px-4 py-3 text-sm">
                                        <input
                                            type="checkbox"
                                            checked={includeReason}
                                            onChange={(e) => setIncludeReason(e.target.checked)}
                                            className="mt-1 h-4 w-4 rounded border-input text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                                        />
                                        <span>
                                            <span className="block font-semibold">Include metric reasoning</span>
                                            <span className="block text-xs text-muted-foreground">
                                                Disabling reduces token usage, but explanations will be omitted.
                                            </span>
                                        </span>
                                    </label>
                                </div>
                            )}

                            {step === 'review' && (
                                <div className="space-y-6 animate-in fade-in slide-in-from-right-4 duration-300">
                                    <div className="mb-4 text-center">
                                        <div className="inline-flex h-16 w-16 items-center justify-center rounded-full bg-primary/10 text-primary mb-4">
                                            <Play className="h-8 w-8 fill-primary" />
                                        </div>
                                        <h3 className="text-2xl font-black">Ready to launch?</h3>
                                        <p className="text-muted-foreground">Verify your configuration before starting.</p>
                                    </div>

                                    <div className="space-y-4">
                                        <div>
                                            <label className="text-xs font-bold text-muted-foreground uppercase">Evaluation Name</label>
                                            <input
                                                type="text"
                                                value={evaluationName}
                                                onChange={(e) => setEvaluationName(e.target.value)}
                                                placeholder="Give this evaluation a name..."
                                                className="mt-1 flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                                            />
                                        </div>

                                        <div className="rounded-xl bg-accent/30 border border-border overflow-hidden">
                                            <div className="grid gap-px bg-border sm:grid-cols-3">
                                                <div className="bg-card p-4">
                                                    <p className="text-[10px] font-bold uppercase text-muted-foreground mb-1">Test Set</p>
                                                    <p className="text-sm font-semibold truncate">{testSets.find(t => t.id === selectedTestSet)?.name}</p>
                                                </div>
                                                <div className="bg-card p-4">
                                                    <p className="text-[10px] font-bold uppercase text-muted-foreground mb-1">Knowledge Base</p>
                                                    <p className="text-sm font-semibold truncate">{kbs.find(k => k.id === selectedKb)?.name}</p>
                                                </div>
                                                <div className="bg-card p-4">
                                                    <p className="text-[10px] font-bold uppercase text-muted-foreground mb-1">Index</p>
                                                    <p className="text-sm font-semibold truncate">{indexes.find(i => i.id === selectedIndex)?.name}</p>
                                                </div>
                                                <div className="bg-card p-4 sm:col-span-3 border-t border-border">
                                                    <p className="text-[10px] font-bold uppercase text-muted-foreground mb-1">Query Settings</p>
                                                    <div className="grid gap-2 text-[11px] text-muted-foreground sm:grid-cols-3">
                                                        <span className="truncate">RAG: <strong className="text-foreground">{queryModel}</strong></span>
                                                        <span>Top K: <strong className="text-foreground">{queryTopK}</strong></span>
                                                        <span className="truncate">Judge: <strong className="text-foreground">{judgeModel}</strong></span>
                                                        {queryModelSupportsReasoningEffort && (
                                                            <span className="truncate">RAG Effort: <strong className="text-foreground">{queryReasoningEffort || 'Default'}</strong></span>
                                                        )}
                                                    </div>
                                                </div>
                                                <div className="bg-card p-4 sm:col-span-3 border-t border-border">
                                                    <p className="text-[10px] font-bold uppercase text-muted-foreground mb-1">Selected Metrics</p>
                                                    <div className="flex flex-wrap gap-1.5 mt-1">
                                                        {selectedMetrics.map(id => (
                                                            <span key={id} className="text-[10px] bg-muted px-2 py-0.5 rounded font-medium text-muted-foreground">
                                                                {AVAILABLE_METRICS.find(m => m.id === id)?.name}
                                                            </span>
                                                        ))}
                                                        {selectedMetrics.length === 0 && (
                                                            <span className="text-[10px] text-destructive font-medium italic">No metrics selected! (Evaluation will only track performance)</span>
                                                        )}
                                                    </div>
                                                    <div className="mt-2 text-[10px] font-medium text-muted-foreground">
                                                        Reasoning: {includeReason ? 'Enabled' : 'Disabled'}
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </>
                    )}
                </div>
        </DialogShell>
    )
}
