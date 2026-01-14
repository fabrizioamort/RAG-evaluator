import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
    Loader2,
    Search,
    ChevronDown,
    CheckCircle2,
    MessageSquare,
    Database,
    Fingerprint,
    Info,
    Clock
} from 'lucide-react'
import { api, EvaluationResult } from '../../api/client'
import { cn } from '@/lib/utils'
import { MetricExplainability } from './MetricExplainability'
import { RetrievalTraceViewer } from './RetrievalTraceViewer'

interface EvaluationResultsProps {
    evaluationId: string
    onBack: () => void
}

export function EvaluationResults({ evaluationId, onBack }: EvaluationResultsProps) {
    const [page] = useState(1)
    const [search, setSearch] = useState('')
    const [expandedResultId, setExpandedResultId] = useState<string | null>(null)
    const [activeDetailTab, setActiveDetailTab] = useState<'overview' | 'trace'>('overview')

    const { data: evaluation } = useQuery({
        queryKey: ['evaluation', evaluationId],
        queryFn: () => api.evaluations.get(evaluationId),
    })

    const { data: results, isLoading } = useQuery({
        queryKey: ['evaluation-results', evaluationId, page, search],
        queryFn: () => api.evaluations.getResults(evaluationId, { limit: 50, offset: (page - 1) * 50 }),
    })

    if (isLoading) {
        return (
            <div className="flex justify-center py-20">
                <Loader2 className="h-8 w-8 animate-spin text-primary/50" />
            </div>
        )
    }

    const items = results?.data?.items || []

    return (
        <div className="space-y-6 animate-in fade-in slide-in-from-right-4 duration-500">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div className="space-y-1">
                    <h2 className="text-2xl font-bold tracking-tight">Evaluation Results</h2>
                    <p className="text-muted-foreground">
                        Detailed analysis of {evaluation?.data.result_count || 0} test cases.
                    </p>
                </div>
                <button
                    onClick={onBack}
                    className="rounded-lg border border-border px-4 py-2 text-sm font-medium hover:bg-accent transition-colors"
                >
                    Back to List
                </button>
            </div>

            {/* Metrics Summary Cards */}
            {evaluation?.data.summary_metrics && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {[
                        { label: 'Faithfulness', value: evaluation.data.summary_metrics.faithfulness_avg, color: 'text-blue-500', bg: 'bg-blue-500/10', border: 'border-blue-500/20' },
                        { label: 'Relevancy', value: evaluation.data.summary_metrics.relevancy_avg, color: 'text-green-500', bg: 'bg-green-500/10', border: 'border-green-500/20' },
                        { label: 'Precision', value: evaluation.data.summary_metrics.precision_avg, color: 'text-purple-500', bg: 'bg-purple-500/10', border: 'border-purple-500/20' },
                        { label: 'Recall', value: evaluation.data.summary_metrics.recall_avg, color: 'text-orange-500', bg: 'bg-orange-500/10', border: 'border-orange-500/20' }
                    ].map((m) => (
                        <div key={m.label} className={cn("rounded-xl border p-4", m.bg, m.border)}>
                            <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">{m.label}</p>
                            <p className={`text - 2xl font - black mt - 1 ${m.color} `}>
                                {m.value?.toFixed(2) || 'N/A'}
                            </p>
                        </div>
                    ))}
                </div>
            )}

            {/* Filters */}
            <div className="flex items-center gap-4">
                <div className="relative flex-1 max-w-sm">
                    <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                    <input
                        type="text"
                        placeholder="Search questions..."
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                        className="h-10 w-full rounded-lg border border-input bg-background pl-9 pr-4 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    />
                </div>
            </div>

            {/* Results List */}
            <div className="space-y-4">
                {items.map((result: EvaluationResult) => (
                    <div
                        key={result.id}
                        className="rounded-xl border border-border bg-card overflow-hidden transition-all"
                    >
                        {/* Result Header Row */}
                        <div
                            className="flex items-center gap-4 p-4 hover:bg-accent/50 cursor-pointer"
                            onClick={() => {
                                setExpandedResultId(expandedResultId === result.id ? null : result.id)
                                setActiveDetailTab('overview')
                            }}
                        >
                            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10 text-primary font-bold text-xs">
                                #{result.test_case_id?.slice(0, 4)}
                            </div>

                            <div className="flex-1 min-w-0">
                                <p className="font-medium truncate">{result.question || "Unknown Question"}</p>
                                <div className="flex items-center gap-4 mt-1 text-xs text-muted-foreground">
                                    <span className="flex items-center gap-1">
                                        <Database className="h-3 w-3" />
                                        {result.retrieved_context_artifact_id ? "Context available" : "No context"}
                                    </span>
                                    <span className="flex items-center gap-1">
                                        <Clock className="h-3 w-3" />
                                        {result.latency_seconds?.toFixed(2)}s
                                    </span>
                                    <span className="flex items-center gap-1">
                                        <Fingerprint className="h-3 w-3" />
                                        {(result.prompt_tokens || 0) + (result.completion_tokens || 0)} tokens
                                    </span>
                                </div>
                            </div>

                            {/* Mini Score Badges */}
                            <div className="flex items-center gap-2">
                                <ScoreBadge label="F" value={result.faithfulness_score} />
                                <ScoreBadge label="R" value={result.relevancy_score} />
                                <ScoreBadge label="P" value={result.precision_score} />
                                <ScoreBadge label="C" value={result.recall_score} />
                            </div>

                            <ChevronDown
                                className={cn(
                                    "h-5 w-5 text-muted-foreground transition-transform",
                                    expandedResultId === result.id ? "rotate-180" : ""
                                )}
                            />
                        </div>

                        {/* Expanded Detail View */}
                        {expandedResultId === result.id && (
                            <div className="border-t border-border bg-muted/30 p-6 space-y-6 animate-in slide-in-from-top-2">
                                {/* Detail Tabs */}
                                <div className="flex gap-6 border-b border-border/50 mb-4">
                                    <button
                                        onClick={() => setActiveDetailTab('overview')}
                                        className={cn(
                                            "pb-2 text-sm font-bold border-b-2 transition-all",
                                            activeDetailTab === 'overview' ? "border-primary text-primary" : "border-transparent text-muted-foreground hover:text-foreground"
                                        )}
                                    >
                                        Overview & Metrics
                                    </button>
                                    <button
                                        onClick={() => setActiveDetailTab('trace')}
                                        className={cn(
                                            "pb-2 text-sm font-bold border-b-2 transition-all",
                                            activeDetailTab === 'trace' ? "border-primary text-primary" : "border-transparent text-muted-foreground hover:text-foreground"
                                        )}
                                    >
                                        Retrieval Trace
                                    </button>
                                </div>

                                {activeDetailTab === 'overview' ? (
                                    <div className="grid md:grid-cols-2 gap-6">
                                        {/* Left Column: Q&A */}
                                        <div className="space-y-4">
                                            <div>
                                                <h4 className="flex items-center gap-2 text-sm font-semibold text-primary mb-2">
                                                    <Info className="h-4 w-4" />
                                                    Question
                                                </h4>
                                                <div className="rounded-lg bg-background p-3 border border-border text-sm">
                                                    {result.question}
                                                </div>
                                            </div>
                                            <div>
                                                <h4 className="flex items-center gap-2 text-sm font-semibold text-green-600 mb-2">
                                                    <CheckCircle2 className="h-4 w-4" />
                                                    Expected Answer
                                                </h4>
                                                <div className="rounded-lg bg-background p-3 border border-border text-sm">
                                                    {result.expected_answer || <span className="text-muted-foreground italic">Not specified</span>}
                                                </div>
                                            </div>
                                            <div>
                                                <h4 className="flex items-center gap-2 text-sm font-semibold text-blue-600 mb-2">
                                                    <MessageSquare className="h-4 w-4" />
                                                    Generated Answer
                                                </h4>
                                                <div className="rounded-lg bg-background p-3 border border-border text-sm">
                                                    {result.generated_answer}
                                                </div>
                                            </div>
                                        </div>

                                        {/* Right Column: Reasoning & Metrics */}
                                        <div className="space-y-4">
                                            <h4 className="text-sm font-semibold mb-2">Metric Analysis</h4>
                                            <div className="space-y-3">
                                                <MetricExplainability
                                                    label="Faithfulness"
                                                    score={result.faithfulness_score}
                                                    reason={result.faithfulness_reason}
                                                />
                                                <MetricExplainability
                                                    label="Answer Relevancy"
                                                    score={result.relevancy_score}
                                                    reason={result.relevancy_reason}
                                                />
                                                <MetricExplainability
                                                    label="Contextual Precision"
                                                    score={result.precision_score}
                                                    reason={result.precision_reason}
                                                />
                                                <MetricExplainability
                                                    label="Contextual Recall"
                                                    score={result.recall_score}
                                                    reason={result.recall_reason}
                                                />
                                            </div>
                                        </div>
                                    </div>
                                ) : (
                                    <RetrievalTraceViewer evaluationId={evaluationId} resultId={result.id} />
                                )}
                            </div>
                        )}
                    </div>
                ))}

                {items.length === 0 && (
                    <div className="py-20 text-center text-muted-foreground">
                        No results found matching your search.
                    </div>
                )}
            </div>
        </div>
    )
}

function ScoreBadge({ label, value }: { label: string, value: number | null }) {
    if (value === null || value === undefined) return null;

    // Color scale
    const colorClass = value >= 0.7 ? "bg-green-500/10 text-green-600 border-green-500/20" :
        value >= 0.4 ? "bg-yellow-500/10 text-yellow-600 border-yellow-500/20" :
            "bg-red-500/10 text-red-600 border-red-500/20";

    return (
        <div className={cn("flex items-center gap-1 rounded px-1.5 py-0.5 border text-[10px] font-bold", colorClass)}>
            <span>{label}</span>
            <span>{value.toFixed(2)}</span>
        </div>
    )
}
