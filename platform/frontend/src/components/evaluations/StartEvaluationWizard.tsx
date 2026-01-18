import React, { useState, useEffect, useCallback } from 'react'
import { X, Play, Loader2, Database, FileText, ChevronRight, ChevronLeft, LucideIcon, Layers, Calendar } from 'lucide-react'
import { api, KnowledgeBase, TestSet, KnowledgeBaseIndex, EvaluationCreate } from '@/api/client'
import { cn } from '@/lib/utils'

interface StartEvaluationWizardProps {
    projectId: string
    isOpen: boolean
    onClose: () => void
    onStarted: (evaluationId: string) => void
}

type Step = 'testset' | 'kb' | 'index' | 'review'

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

export function StartEvaluationWizard({ projectId, isOpen, onClose, onStarted }: StartEvaluationWizardProps) {
    const [step, setStep] = useState<Step>('testset')
    const [kbs, setKbs] = useState<KnowledgeBase[]>([])
    const [testSets, setTestSets] = useState<TestSet[]>([])
    const [indexes, setIndexes] = useState<KnowledgeBaseIndex[]>([])

    const [selectedKb, setSelectedKb] = useState<string>('')
    const [selectedTestSet, setSelectedTestSet] = useState<string>('')
    const [selectedIndex, setSelectedIndex] = useState<string>('')
    const [evaluationName, setEvaluationName] = useState<string>('')

    const [isLoading, setIsLoading] = useState(false)
    const [isLoadingIndexes, setIsLoadingIndexes] = useState(false)
    const [isStarting, setIsStarting] = useState(false)

    const loadData = useCallback(async () => {
        setIsLoading(true)
        try {
            const [kbRes, tsRes] = await Promise.all([
                api.knowledgeBases.list(projectId),
                api.testSets.list(projectId)
            ])
            setKbs(kbRes.data.items)
            setTestSets(tsRes.data.items)
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
            setSelectedKb('')
            setSelectedTestSet('')
            setSelectedIndex('')
            setEvaluationName('')
            setIndexes([])
        }
    }, [isOpen, projectId, loadData])

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

    const handleStart = async () => {
        setIsStarting(true)
        try {
            const data: EvaluationCreate = {
                name: evaluationName || undefined,
                test_set_id: selectedTestSet,
                knowledge_base_index_id: selectedIndex
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

    const steps: { id: Step; label: string; icon: LucideIcon }[] = [
        { id: 'testset', label: 'Test Set', icon: FileText },
        { id: 'kb', label: 'Knowledge Base', icon: Database },
        { id: 'index', label: 'Index', icon: Layers },
        { id: 'review', label: 'Review', icon: Play }
    ]

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
                        <Play className="h-5 w-5 text-primary fill-primary" />
                        Launch Evaluation
                    </h2>
                    <button onClick={onClose} className="rounded-md p-1 hover:bg-muted transition-colors">
                        <X className="h-5 w-5" />
                    </button>
                </div>

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
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </>
                    )}
                </div>


                {/* Footer */}
                <div className="flex items-center justify-between border-t border-border p-6 bg-muted/20 rounded-b-xl">
                    <button
                        onClick={() => {
                            if (step === 'review') setStep('index')
                            else if (step === 'index') setStep('kb')
                            else if (step === 'kb') setStep('testset')
                        }}
                        disabled={step === 'testset' || isStarting}
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
                                    if (step === 'testset') setStep('kb')
                                    else if (step === 'kb') setStep('index')
                                    else if (step === 'index') setStep('review')
                                }}
                                disabled={
                                    (step === 'testset' && !selectedTestSet) ||
                                    (step === 'kb' && !selectedKb) ||
                                    (step === 'index' && !selectedIndex)
                                }
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