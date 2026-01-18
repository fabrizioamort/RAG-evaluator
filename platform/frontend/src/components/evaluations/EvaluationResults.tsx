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
    Clock,
    Target
} from 'lucide-react'
import { api, EvaluationResult } from '../../api/client'
import { cn } from '@/lib/utils'
import { MetricExplainability } from './MetricExplainability'
import { RetrievalTraceViewer } from './RetrievalTraceViewer'
import { ManifestViewer } from './ManifestViewer'
import { BaselineComparison } from './BaselineComparison'
import { DifficultyChart } from './DifficultyChart'

interface EvaluationResultsProps {
    evaluationId: string
    onBack: () => void
}

const formatScore = (val: number | string | null | undefined, decimals: number = 2) => {
    if (val === null || val === undefined) return 'N/A';
    const num = typeof val === 'number' ? val : parseFloat(val);
    return isNaN(num) ? 'N/A' : num.toFixed(decimals);
};

export function EvaluationResults({ evaluationId, onBack }: EvaluationResultsProps) {
    const [page] = useState(1)
    const [search, setSearch] = useState('')
    const [expandedResultId, setExpandedResultId] = useState<string | null>(null)
    const [activeDetailTab, setActiveDetailTab] = useState<'overview' | 'trace'>('overview')
    const [activeTopTab, setActiveTopTab] = useState<'results' | 'manifest'>('results')
    const [isSettingBaseline, setIsSettingBaseline] = useState(false)
    const [baselineReason, setBaselineReason] = useState('')

    const { data: evaluation } = useQuery({
        queryKey: ['evaluation', evaluationId],
        queryFn: () => api.evaluations.get(evaluationId),
    })

    const { data: results, isLoading } = useQuery({
        queryKey: ['evaluation-results', evaluationId, page, search],
        queryFn: () => api.evaluations.getResults(evaluationId, { limit: 50, offset: (page - 1) * 50 }),
    })

    const { data: baseline, refetch: refetchBaseline } = useQuery({
        queryKey: ['project-baseline', evaluation?.data?.project_id],
        queryFn: () => api.projects.getBaseline(evaluation?.data?.project_id || ''),
        enabled: !!evaluation?.data?.project_id,
        retry: false,
    })

    const handleSetBaseline = async () => {
        if (!baselineReason) return
        try {
            await api.evaluations.setBaseline(evaluationId, baselineReason)
            setIsSettingBaseline(false)
            setBaselineReason('')
            refetchBaseline()
        } catch (error) {
            console.error('Failed to set baseline:', error)
        }
    }

    if (isLoading || !evaluation) {
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
                    <h2 className="text-2xl font-bold tracking-tight">
                        {evaluation?.data.name || "Evaluation Results"}
                    </h2>
                    <p className="text-muted-foreground">
                        Detailed analysis of {evaluation?.data.result_count || 0} test cases.
                    </p>
                </div>
                <div className="flex items-center gap-3">
                    {evaluation?.data.status === 'completed' && !evaluation.data.is_baseline && (
                        <button
                            onClick={() => setIsSettingBaseline(true)}
                            className="flex items-center gap-2 rounded-lg bg-primary/10 px-4 py-2 text-sm font-bold text-primary hover:bg-primary/20 transition-all border border-primary/20"
                        >
                            <Target className="h-4 w-4" />
                            Set as Baseline
                        </button>
                    )}
                    <button
                        onClick={onBack}
                        className="rounded-lg border border-border px-4 py-2 text-sm font-medium hover:bg-accent transition-colors"
                    >
                        Back to List
                    </button>
                </div>
            </div>

            {/* Baseline Settings Overlay/Dialog */}
            {isSettingBaseline && (
                <div className="rounded-xl border border-primary/30 bg-primary/5 p-6 animate-in zoom-in-95">
                    <h3 className="text-sm font-black uppercase tracking-widest text-primary mb-4">Set this evaluation as baseline</h3>
                    <div className="space-y-4">
                        <div>
                            <label className="text-xs font-bold text-muted-foreground uppercase">Reason for baseline</label>
                            <input
                                type="text"
                                value={baselineReason}
                                onChange={(e) => setBaselineReason(e.target.value)}
                                placeholder="e.g. Best performance so far with GPT-4"
                                className="mt-1 flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                            />
                        </div>
                        <div className="flex gap-2">
                            <button
                                onClick={handleSetBaseline}
                                className="rounded-md bg-primary px-4 py-2 text-xs font-bold text-primary-foreground hover:bg-primary/90"
                            >
                                Confirm Baseline
                            </button>
                            <button
                                onClick={() => setIsSettingBaseline(false)}
                                className="rounded-md border border-border px-4 py-2 text-xs font-bold hover:bg-accent"
                            >
                                Cancel
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Top Tabs */}
            <div className="flex gap-8 border-b border-border">
                <button
                    onClick={() => setActiveTopTab('results')}
                    className={cn(
                        "pb-4 text-sm font-bold border-b-2 transition-all",
                        activeTopTab === 'results' ? "border-primary text-primary" : "border-transparent text-muted-foreground hover:text-foreground"
                    )}
                >
                    Evaluation Results
                </button>
                <button
                    onClick={() => setActiveTopTab('manifest')}
                    className={cn(
                        "pb-4 text-sm font-bold border-b-2 transition-all",
                        activeTopTab === 'manifest' ? "border-primary text-primary" : "border-transparent text-muted-foreground hover:text-foreground"
                    )}
                >
                    Run Manifest
                </button>
            </div>

            {activeTopTab === 'results' ? (
                <>
                    {/* Metrics Summary Cards */}
                    {evaluation?.data.summary_metrics && (
                        <div className="space-y-6">
                            {baseline?.data && (
                                <BaselineComparison
                                    current={evaluation.data}
                                    baseline={baseline.data}
                                />
                            )}

                            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                                {/* Left Column: Metrics & Performance */}
                                <div className="lg:col-span-2 space-y-4">
                                    {!baseline?.data && (
                                        <div className="grid grid-cols-2 gap-4">
                                            {[
                                                { label: 'Faithfulness', value: evaluation.data.summary_metrics.faithfulness_avg, color: 'text-blue-500', bg: 'bg-blue-500/10', border: 'border-blue-500/20' },
                                                { label: 'Relevancy', value: evaluation.data.summary_metrics.relevancy_avg, color: 'text-green-500', bg: 'bg-green-500/10', border: 'border-green-500/20' },
                                                { label: 'Precision', value: evaluation.data.summary_metrics.precision_avg, color: 'text-purple-500', bg: 'bg-purple-500/10', border: 'border-purple-500/20' },
                                                { label: 'Recall', value: evaluation.data.summary_metrics.recall_avg, color: 'text-orange-500', bg: 'bg-orange-500/10', border: 'border-orange-500/20' }
                                            ].map((m) => (
                                                <div key={m.label} className={cn("rounded-xl border p-4", m.bg, m.border)}>
                                                    <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">{m.label}</p>
                                                    <p className={cn("text-2xl font-black mt-1", m.color)}>
                                                        {formatScore(m.value)}
                                                    </p>
                                                </div>
                                            ))}
                                        </div>
                                    )}

                                    <div className="rounded-xl border border-border bg-card p-4">
                                        <div className="flex items-center justify-between mb-2">
                                            <h4 className="text-sm font-bold uppercase tracking-wider text-muted-foreground">Performance & Cost</h4>
                                        </div>
                                        <div className="grid grid-cols-2 gap-4">
                                            <div>
                                                <p className="text-[10px] font-bold text-muted-foreground uppercase">Avg Latency</p>
                                                {formatScore(evaluation.data.performance_metrics?.avg_latency_seconds as number)}s
                                            </div>
                                            <div>
                                                <p className="text-[10px] font-bold text-muted-foreground uppercase">Total Cost</p>
                                                ${formatScore(evaluation.data.cost_metrics?.total_cost_usd as number, 4)}
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                {/* Right Column: Difficulty Breakdown */}
                                <div className="rounded-xl border border-border bg-card p-4">
                                    <h4 className="text-sm font-bold uppercase tracking-wider text-muted-foreground mb-4">Scores by Difficulty</h4>
                                    <DifficultyChart results={items} />
                                </div>
                            </div>
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
                                                {formatScore(result.latency_seconds)}s
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
                </>
            ) : (
                <ManifestViewer evaluationId={evaluationId} />
            )}
        </div>
    )
}

function ScoreBadge({ label, value }: { label: string, value: number | null | undefined }) {
    if (value === null || value === undefined) return null;

    // Color scale
    const colorClass = value >= 0.7 ? "bg-green-500/10 text-green-600 border-green-500/20" :
        value >= 0.4 ? "bg-yellow-500/10 text-yellow-600 border-yellow-500/20" :
            "bg-red-500/10 text-red-600 border-red-500/20";

    return (
        <div className={cn("flex items-center gap-1 rounded px-1.5 py-0.5 border text-[10px] font-bold", colorClass)}>
            <span>{label}</span>
            <span>{formatScore(value)}</span>
        </div>
    )
}
