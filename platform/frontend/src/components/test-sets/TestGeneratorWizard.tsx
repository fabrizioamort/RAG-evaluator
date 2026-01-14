import React, { useState, useEffect } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import {
    X,
    Sparkles,
    Loader2,
    Database,
    Settings,
    ChevronRight,
    ChevronLeft,
    AlertCircle,
    CheckCircle2,
} from 'lucide-react'
import { api, TestGenerationConfig } from '@/api/client'
import { cn } from '@/lib/utils'

interface TestGeneratorWizardProps {
    testSetId: string
    projectId: string
    isOpen: boolean
    onClose: () => void
    onStarted: (jobId: string) => void
}

type Step = 'kb' | 'config' | 'templates' | 'review'

const DEFAULT_DIFFICULTY_DISTRIBUTION = {
    easy: 0.3,
    medium: 0.5,
    hard: 0.2,
}

export function TestGeneratorWizard({
    testSetId,
    projectId,
    isOpen,
    onClose,
    onStarted,
}: TestGeneratorWizardProps) {
    const [step, setStep] = useState<Step>('kb')

    // Form state
    const [selectedKb, setSelectedKb] = useState<string>('')
    const [targetCount, setTargetCount] = useState(20)
    const [questionsPerChunk, setQuestionsPerChunk] = useState(2)
    const [llmModel, setLlmModel] = useState('gpt-4o-mini')
    const [skipSemanticCheck, setSkipSemanticCheck] = useState(false)
    const [difficultyDistribution, setDifficultyDistribution] = useState(DEFAULT_DIFFICULTY_DISTRIBUTION)
    const [selectedTemplates, setSelectedTemplates] = useState<string[]>([])

    // Fetch knowledge bases
    const { data: kbsData, isLoading: isLoadingKbs } = useQuery({
        queryKey: ['knowledge-bases', projectId],
        queryFn: () => api.knowledgeBases.list(projectId),
        enabled: isOpen && !!projectId,
    })

    // Fetch templates
    const { data: templatesData, isLoading: isLoadingTemplates } = useQuery({
        queryKey: ['test-templates'],
        queryFn: () => api.testTemplates.list({ limit: 100 }),
        enabled: isOpen,
    })

    // Start generation mutation
    const startGenerationMutation = useMutation({
        mutationFn: (config: TestGenerationConfig) => api.testSets.generate(testSetId, config),
        onSuccess: (res) => {
            onStarted(res.data.id)
            onClose()
        },
    })

    // Reset state when opened
    useEffect(() => {
        if (isOpen) {
            setStep('kb')
            setSelectedKb('')
            setTargetCount(20)
            setQuestionsPerChunk(2)
            setLlmModel('gpt-4o-mini')
            setSkipSemanticCheck(false)
            setDifficultyDistribution(DEFAULT_DIFFICULTY_DISTRIBUTION)
            setSelectedTemplates([])
        }
    }, [isOpen])

    const handleStart = async () => {
        const config: TestGenerationConfig = {
            knowledge_base_id: selectedKb,
            target_count: targetCount,
            questions_per_chunk: questionsPerChunk,
            llm_model: llmModel,
            skip_semantic_check: skipSemanticCheck,
            difficulty_distribution: difficultyDistribution,
            template_ids: selectedTemplates.length > 0 ? selectedTemplates : undefined,
        }
        await startGenerationMutation.mutateAsync(config)
    }

    const handleDifficultyChange = (level: 'easy' | 'medium' | 'hard', value: number) => {
        // Ensure values are between 0 and 1 and sum to 1
        const newValue = Math.max(0, Math.min(1, value))
        const remaining = 1 - newValue
        const otherLevels = Object.keys(difficultyDistribution).filter(k => k !== level) as ('easy' | 'medium' | 'hard')[]

        // Distribute remaining proportionally
        const currentOtherSum = otherLevels.reduce((sum, l) => sum + difficultyDistribution[l], 0)
        const newDistribution = { ...difficultyDistribution, [level]: newValue }

        if (currentOtherSum > 0) {
            otherLevels.forEach(l => {
                newDistribution[l] = (difficultyDistribution[l] / currentOtherSum) * remaining
            })
        } else {
            otherLevels.forEach((l) => {
                newDistribution[l] = remaining / otherLevels.length
            })
        }

        setDifficultyDistribution(newDistribution)
    }

    const toggleTemplate = (templateId: string) => {
        setSelectedTemplates(prev =>
            prev.includes(templateId)
                ? prev.filter(id => id !== templateId)
                : [...prev, templateId]
        )
    }

    if (!isOpen) return null

    const kbs = kbsData?.data.items || []
    const templates = templatesData?.data.items || []
    const selectedKbData = kbs.find(kb => kb.id === selectedKb)
    const isLoading = isLoadingKbs || isLoadingTemplates

    const steps: { id: Step; label: string; icon: React.ElementType }[] = [
        { id: 'kb', label: 'Source', icon: Database },
        { id: 'config', label: 'Settings', icon: Settings },
        { id: 'templates', label: 'Templates', icon: Sparkles },
        { id: 'review', label: 'Review', icon: CheckCircle2 },
    ]

    const canProceed = () => {
        switch (step) {
            case 'kb':
                return !!selectedKb
            case 'config':
                return targetCount > 0 && targetCount <= 500
            case 'templates':
                return true // Templates are optional
            case 'review':
                return true
            default:
                return false
        }
    }

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <div
                className="absolute inset-0 bg-background/80 backdrop-blur-sm animate-in fade-in duration-200"
                onClick={onClose}
            />
            <div className="relative w-full max-w-2xl rounded-xl border border-border bg-card shadow-2xl animate-in zoom-in-95 duration-200">
                {/* Header */}
                <div className="flex items-center justify-between border-b border-border p-6">
                    <h2 className="text-xl font-bold flex items-center gap-2">
                        <Sparkles className="h-5 w-5 text-primary" />
                        Generate Test Cases
                    </h2>
                    <button
                        onClick={onClose}
                        className="rounded-md p-1 hover:bg-muted transition-colors"
                    >
                        <X className="h-5 w-5" />
                    </button>
                </div>

                {/* Stepper */}
                <div className="flex items-center justify-center bg-muted/30 px-6 py-4 border-b border-border">
                    {steps.map((s, i) => (
                        <React.Fragment key={s.id}>
                            <div
                                className={cn(
                                    'flex flex-col items-center gap-1',
                                    step === s.id ? 'text-primary' : 'text-muted-foreground'
                                )}
                            >
                                <div
                                    className={cn(
                                        'flex h-8 w-8 items-center justify-center rounded-full border-2 text-xs font-bold transition-all',
                                        step === s.id
                                            ? 'border-primary bg-primary text-primary-foreground'
                                            : 'border-muted-foreground/30 bg-background'
                                    )}
                                >
                                    {i + 1}
                                </div>
                                <span className="text-[10px] font-bold uppercase tracking-wider">
                                    {s.label}
                                </span>
                            </div>
                            {i < steps.length - 1 && <div className="mx-4 h-px w-12 bg-border" />}
                        </React.Fragment>
                    ))}
                </div>

                {/* Content */}
                <div className="p-8 min-h-[350px]">
                    {isLoading ? (
                        <div className="flex h-[300px] flex-col items-center justify-center gap-4">
                            <Loader2 className="h-10 w-10 animate-spin text-primary" />
                            <p className="text-muted-foreground font-medium">Loading...</p>
                        </div>
                    ) : (
                        <>
                            {/* Step 1: Select Knowledge Base */}
                            {step === 'kb' && (
                                <div className="space-y-4 animate-in fade-in slide-in-from-right-4 duration-300">
                                    <div className="mb-4">
                                        <h3 className="text-lg font-bold">Select Knowledge Base</h3>
                                        <p className="text-sm text-muted-foreground">
                                            Choose the document source for generating test questions.
                                        </p>
                                    </div>
                                    {kbs.length === 0 ? (
                                        <div className="flex flex-col items-center justify-center py-12 text-center">
                                            <AlertCircle className="h-12 w-12 text-muted-foreground/50 mb-4" />
                                            <p className="text-muted-foreground">
                                                No knowledge bases available. Create one first.
                                            </p>
                                        </div>
                                    ) : (
                                        <div className="grid gap-3 max-h-[280px] overflow-y-auto pr-2">
                                            {kbs.filter(kb => kb.status === 'ready').map(kb => (
                                                <button
                                                    key={kb.id}
                                                    onClick={() => setSelectedKb(kb.id)}
                                                    className={cn(
                                                        'flex items-center justify-between rounded-xl border p-4 text-left transition-all',
                                                        selectedKb === kb.id
                                                            ? 'border-primary bg-primary/5 ring-1 ring-primary'
                                                            : 'border-border hover:border-primary/50 hover:bg-accent'
                                                    )}
                                                >
                                                    <div>
                                                        <p className="font-bold">{kb.name}</p>
                                                        <p className="text-xs text-muted-foreground mt-1">
                                                            Version {kb.current_version} • {kb.document_count} documents
                                                        </p>
                                                    </div>
                                                    {selectedKb === kb.id && (
                                                        <div className="h-2 w-2 rounded-full bg-primary" />
                                                    )}
                                                </button>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* Step 2: Configuration */}
                            {step === 'config' && (
                                <div className="space-y-6 animate-in fade-in slide-in-from-right-4 duration-300">
                                    <div className="mb-4">
                                        <h3 className="text-lg font-bold">Generation Settings</h3>
                                        <p className="text-sm text-muted-foreground">
                                            Configure how test cases should be generated.
                                        </p>
                                    </div>

                                    <div className="grid gap-6">
                                        {/* Target Count */}
                                        <div className="space-y-2">
                                            <label className="text-sm font-semibold">
                                                Number of Questions
                                            </label>
                                            <input
                                                type="number"
                                                min={1}
                                                max={500}
                                                value={targetCount}
                                                onChange={(e) => setTargetCount(parseInt(e.target.value) || 1)}
                                                className="w-full rounded-lg border border-input bg-background px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
                                            />
                                            <p className="text-xs text-muted-foreground">
                                                Between 1 and 500 questions
                                            </p>
                                        </div>

                                        {/* Questions per Chunk */}
                                        <div className="space-y-2">
                                            <label className="text-sm font-semibold">
                                                Questions per Document Chunk
                                            </label>
                                            <input
                                                type="number"
                                                min={1}
                                                max={10}
                                                value={questionsPerChunk}
                                                onChange={(e) => setQuestionsPerChunk(parseInt(e.target.value) || 1)}
                                                className="w-full rounded-lg border border-input bg-background px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
                                            />
                                        </div>

                                        {/* LLM Model */}
                                        <div className="space-y-2">
                                            <label className="text-sm font-semibold">LLM Model</label>
                                            <select
                                                value={llmModel}
                                                onChange={(e) => setLlmModel(e.target.value)}
                                                className="w-full rounded-lg border border-input bg-background px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
                                            >
                                                <option value="gpt-5.1">GPT-5.1</option>
                                                <option value="gpt-5-mini">GPT-5 Mini</option>
                                                <option value="gpt-5-nano">GPT-5 Nano</option>
                                                <option value="gpt-4o">GPT-4o</option>
                                                <option value="gpt-4-turbo">GPT-4 Turbo</option>
                                                <option value="claude-3-haiku-20240307">Claude 3 Haiku</option>
                                                <option value="claude-3-sonnet-20240229">Claude 3 Sonnet</option>
                                            </select>
                                        </div>

                                        {/* Difficulty Distribution */}
                                        <div className="space-y-3">
                                            <label className="text-sm font-semibold">
                                                Difficulty Distribution
                                            </label>
                                            <div className="grid grid-cols-3 gap-4">
                                                {(['easy', 'medium', 'hard'] as const).map(level => (
                                                    <div key={level} className="space-y-1">
                                                        <div className="flex items-center justify-between">
                                                            <span className={cn(
                                                                'text-xs font-bold uppercase',
                                                                level === 'easy' && 'text-green-600',
                                                                level === 'medium' && 'text-amber-600',
                                                                level === 'hard' && 'text-red-600'
                                                            )}>
                                                                {level}
                                                            </span>
                                                            <span className="text-xs text-muted-foreground">
                                                                {Math.round(difficultyDistribution[level] * 100)}%
                                                            </span>
                                                        </div>
                                                        <input
                                                            type="range"
                                                            min={0}
                                                            max={100}
                                                            value={Math.round(difficultyDistribution[level] * 100)}
                                                            onChange={(e) => handleDifficultyChange(level, parseInt(e.target.value) / 100)}
                                                            className="w-full h-2 rounded-full appearance-none bg-muted cursor-pointer"
                                                        />
                                                    </div>
                                                ))}
                                            </div>
                                        </div>

                                        {/* Skip Semantic Check */}
                                        <label className="flex items-center gap-3 cursor-pointer">
                                            <input
                                                type="checkbox"
                                                checked={skipSemanticCheck}
                                                onChange={(e) => setSkipSemanticCheck(e.target.checked)}
                                                className="h-4 w-4 rounded border-input text-primary focus:ring-primary/50"
                                            />
                                            <div>
                                                <span className="text-sm font-medium">
                                                    Skip semantic duplicate check
                                                </span>
                                                <p className="text-xs text-muted-foreground">
                                                    Faster but may produce similar questions
                                                </p>
                                            </div>
                                        </label>
                                    </div>
                                </div>
                            )}

                            {/* Step 3: Templates */}
                            {step === 'templates' && (
                                <div className="space-y-4 animate-in fade-in slide-in-from-right-4 duration-300">
                                    <div className="mb-4">
                                        <h3 className="text-lg font-bold">Select Templates (Optional)</h3>
                                        <p className="text-sm text-muted-foreground">
                                            Choose question templates to guide generation. Leave empty for auto-selection.
                                        </p>
                                    </div>
                                    <div className="grid gap-3 max-h-[280px] overflow-y-auto pr-2">
                                        {templates.map(template => (
                                            <button
                                                key={template.id}
                                                onClick={() => toggleTemplate(template.id)}
                                                className={cn(
                                                    'flex items-start justify-between rounded-xl border p-4 text-left transition-all',
                                                    selectedTemplates.includes(template.id)
                                                        ? 'border-primary bg-primary/5 ring-1 ring-primary'
                                                        : 'border-border hover:border-primary/50 hover:bg-accent'
                                                )}
                                            >
                                                <div className="flex-1">
                                                    <div className="flex items-center gap-2">
                                                        <p className="font-bold">{template.name}</p>
                                                        {template.is_builtin && (
                                                            <span className="rounded-full bg-muted px-2 py-0.5 text-[10px] font-bold uppercase text-muted-foreground">
                                                                Builtin
                                                            </span>
                                                        )}
                                                        <span className={cn(
                                                            'rounded-full px-2 py-0.5 text-[10px] font-bold uppercase',
                                                            template.complexity_level === 'easy' && 'bg-green-100 text-green-700',
                                                            template.complexity_level === 'medium' && 'bg-amber-100 text-amber-700',
                                                            template.complexity_level === 'hard' && 'bg-red-100 text-red-700'
                                                        )}>
                                                            {template.complexity_level}
                                                        </span>
                                                    </div>
                                                    <p className="text-xs text-muted-foreground mt-1">
                                                        {template.description || template.question_template}
                                                    </p>
                                                    <p className="text-xs text-muted-foreground/70 mt-0.5 italic">
                                                        Category: {template.category}
                                                    </p>
                                                </div>
                                                {selectedTemplates.includes(template.id) && (
                                                    <CheckCircle2 className="h-5 w-5 text-primary flex-shrink-0 ml-2" />
                                                )}
                                            </button>
                                        ))}
                                    </div>
                                    {selectedTemplates.length > 0 && (
                                        <p className="text-sm text-primary font-medium">
                                            {selectedTemplates.length} template{selectedTemplates.length > 1 ? 's' : ''} selected
                                        </p>
                                    )}
                                </div>
                            )}

                            {/* Step 4: Review */}
                            {step === 'review' && (
                                <div className="space-y-6 animate-in fade-in slide-in-from-right-4 duration-300">
                                    <div className="mb-4 text-center">
                                        <div className="inline-flex h-16 w-16 items-center justify-center rounded-full bg-primary/10 text-primary mb-4">
                                            <Sparkles className="h-8 w-8" />
                                        </div>
                                        <h3 className="text-2xl font-black">Ready to generate?</h3>
                                        <p className="text-muted-foreground">
                                            Review your configuration before starting.
                                        </p>
                                    </div>

                                    <div className="rounded-xl bg-accent/30 border border-border overflow-hidden">
                                        <div className="divide-y divide-border">
                                            <div className="flex justify-between items-center p-4">
                                                <span className="text-sm text-muted-foreground">Knowledge Base</span>
                                                <span className="text-sm font-semibold">{selectedKbData?.name}</span>
                                            </div>
                                            <div className="flex justify-between items-center p-4">
                                                <span className="text-sm text-muted-foreground">Target Questions</span>
                                                <span className="text-sm font-semibold">{targetCount}</span>
                                            </div>
                                            <div className="flex justify-between items-center p-4">
                                                <span className="text-sm text-muted-foreground">LLM Model</span>
                                                <span className="text-sm font-semibold">{llmModel}</span>
                                            </div>
                                            <div className="flex justify-between items-center p-4">
                                                <span className="text-sm text-muted-foreground">Difficulty Mix</span>
                                                <span className="text-sm font-semibold">
                                                    {Math.round(difficultyDistribution.easy * 100)}% / {Math.round(difficultyDistribution.medium * 100)}% / {Math.round(difficultyDistribution.hard * 100)}%
                                                </span>
                                            </div>
                                            <div className="flex justify-between items-center p-4">
                                                <span className="text-sm text-muted-foreground">Templates</span>
                                                <span className="text-sm font-semibold">
                                                    {selectedTemplates.length > 0 ? `${selectedTemplates.length} selected` : 'Auto'}
                                                </span>
                                            </div>
                                        </div>
                                    </div>

                                    {startGenerationMutation.isError && (
                                        <div className="flex items-center gap-2 rounded-lg bg-destructive/10 p-4 text-destructive">
                                            <AlertCircle className="h-5 w-5 flex-shrink-0" />
                                            <p className="text-sm">
                                                Failed to start generation. Please try again.
                                            </p>
                                        </div>
                                    )}
                                </div>
                            )}
                        </>
                    )}
                </div>

                {/* Footer */}
                <div className="flex items-center justify-between border-t border-border p-6 bg-muted/20 rounded-b-xl">
                    <button
                        onClick={() => {
                            if (step === 'review') setStep('templates')
                            else if (step === 'templates') setStep('config')
                            else if (step === 'config') setStep('kb')
                        }}
                        disabled={step === 'kb' || startGenerationMutation.isPending}
                        className="flex items-center gap-2 px-4 py-2 text-sm font-medium hover:bg-accent rounded-lg transition-colors disabled:opacity-30"
                    >
                        <ChevronLeft className="h-4 w-4" /> Back
                    </button>

                    <div className="flex gap-3">
                        <button
                            onClick={onClose}
                            disabled={startGenerationMutation.isPending}
                            className="px-6 py-2 text-sm font-medium hover:bg-accent rounded-lg transition-colors disabled:opacity-50"
                        >
                            Cancel
                        </button>

                        {step === 'review' ? (
                            <button
                                onClick={handleStart}
                                disabled={startGenerationMutation.isPending}
                                className="flex items-center gap-2 rounded-lg bg-primary px-8 py-2 text-sm font-bold text-primary-foreground hover:bg-primary/90 transition-all shadow-lg shadow-primary/20 active:scale-95 disabled:opacity-50"
                            >
                                {startGenerationMutation.isPending ? (
                                    <Loader2 className="h-4 w-4 animate-spin" />
                                ) : (
                                    <Sparkles className="h-4 w-4" />
                                )}
                                {startGenerationMutation.isPending ? 'Starting...' : 'Generate'}
                            </button>
                        ) : (
                            <button
                                onClick={() => {
                                    if (step === 'kb') setStep('config')
                                    else if (step === 'config') setStep('templates')
                                    else if (step === 'templates') setStep('review')
                                }}
                                disabled={!canProceed()}
                                className="flex items-center gap-2 rounded-lg bg-primary px-8 py-2 text-sm font-bold text-primary-foreground hover:bg-primary/90 transition-all disabled:opacity-50"
                            >
                                Continue <ChevronRight className="h-4 w-4" />
                            </button>
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}
