import { useQuery } from '@tanstack/react-query'
import { Scale, CheckCircle2, XCircle, MinusCircle } from 'lucide-react'
import { api } from '../../api/client'
import { cn } from '@/lib/utils'

/** Per-result Legal RAG Bench payload stored inside the raw metrics artifact. */
interface LegalRagRetrieval {
    relevant_passage_id: string | null
    retrieved_passage_ids: string[]
    retrieval_metric: string
    hit_at_k: boolean | null
    gold_accessed: boolean
    gold_access_rank: number | null
    top_k: number
}

interface LegalRagJudge {
    correct: boolean | null
    grounded: boolean | null
    reasoning: string
    parse_error?: string | null
}

interface LegalRagSuccessSignals {
    g_eval_score?: number | null
    g_eval_pass?: boolean | null
    judge_correct?: boolean | null
    judge_grounded?: boolean | null
    taxonomy?: string | null
    taxonomy_success?: boolean | null
    retrieval_hit?: boolean | null
    gold_accessed?: boolean | null
    supported_by_retrieved_context?: boolean | null
    alternate_evidence_supported?: boolean | null
    correct_without_gold?: boolean | null
}

interface LegalRagResult {
    retrieval?: LegalRagRetrieval | null
    judge?: LegalRagJudge | null
    taxonomy?: string | null
    success_signals?: LegalRagSuccessSignals | null
}

/** Aggregate Legal RAG Bench payload stored in the evaluation summary metrics. */
export interface LegalRagBenchSummaryData {
    count?: number
    retrieval?: {
        count?: number
        hit_at_k_rate?: number | null
        gold_accessed_rate?: number | null
    }
    judge?: {
        count?: number
        scored_count?: number
        correct_rate?: number | null
        grounded_rate?: number | null
        parse_error_count?: number
    }
    taxonomy?: Record<string, number>
    taxonomy_success_rate?: number | null
    success_signals?: {
        count?: number
        g_eval_pass_rate?: number | null
        judge_correct_rate?: number | null
        judge_grounded_rate?: number | null
        taxonomy_success_rate?: number | null
        gold_accessed_rate?: number | null
        alternate_evidence_supported_rate?: number | null
        alternate_evidence_supported_count?: number
        correct_without_gold_rate?: number | null
        correct_without_gold_count?: number
    }
}

const TAXONOMY_STYLES: Record<string, { label: string; color: string; bg: string; border: string }> = {
    success: { label: 'Success', color: 'text-emerald-500', bg: 'bg-emerald-500/10', border: 'border-emerald-500/20' },
    reasoning_error: { label: 'Reasoning Error', color: 'text-amber-500', bg: 'bg-amber-500/10', border: 'border-amber-500/20' },
    retrieval_error: { label: 'Retrieval Error', color: 'text-orange-500', bg: 'bg-orange-500/10', border: 'border-orange-500/20' },
    hallucination_or_ungrounded: { label: 'Hallucination / Ungrounded', color: 'text-rose-500', bg: 'bg-rose-500/10', border: 'border-rose-500/20' },
    abstention: { label: 'Abstention', color: 'text-sky-500', bg: 'bg-sky-500/10', border: 'border-sky-500/20' },
    judge_error: { label: 'Judge Error', color: 'text-zinc-500', bg: 'bg-zinc-500/10', border: 'border-zinc-500/20' },
}

const formatRate = (val: number | null | undefined) =>
    val === null || val === undefined ? 'N/A' : `${(val * 100).toFixed(1)}%`

const formatScore = (val: number | null | undefined) =>
    val === null || val === undefined ? 'N/A' : val.toFixed(3)

const taxonomyStyle = (key: string) =>
    TAXONOMY_STYLES[key] ?? { label: key, color: 'text-muted-foreground', bg: 'bg-muted/40', border: 'border-border' }

function BoolBadge({
    value,
    label,
    neutralFalse = false,
}: {
    value: boolean | null | undefined
    label: string
    neutralFalse?: boolean
}) {
    const known = value === true || value === false
    const falseIsNeutral = value === false && neutralFalse
    const Icon = value === true ? CheckCircle2 : value === false && !falseIsNeutral ? XCircle : MinusCircle
    return (
        <div
            className={cn(
                'flex items-center gap-2 rounded-lg border px-3 py-2',
                value === true && 'bg-emerald-500/10 border-emerald-500/20 text-emerald-600',
                value === false && !falseIsNeutral && 'bg-rose-500/10 border-rose-500/20 text-rose-600',
                (!known || falseIsNeutral) && 'bg-muted/40 border-border text-muted-foreground',
            )}
        >
            <Icon className="h-4 w-4" />
            <span className="text-xs font-bold uppercase tracking-wider">{label}</span>
        </div>
    )
}

/** Neutral badge for refusals: an abstention is not correct nor grounded, but it
 * is not a hallucination either, so it gets its own state instead of red X badges. */
function AbstainedBadge() {
    return (
        <div className="flex items-center gap-2 rounded-lg border px-3 py-2 bg-sky-500/10 border-sky-500/20 text-sky-600">
            <MinusCircle className="h-4 w-4" />
            <span className="text-xs font-bold uppercase tracking-wider">Abstained</span>
        </div>
    )
}

/** Summary card shown above the results list when the evaluation ran Legal RAG Bench metrics. */
export function LegalRagBenchSummary({ data }: { data: LegalRagBenchSummaryData }) {
    const signals = data.success_signals
    const retrievalRate =
        data.retrieval?.hit_at_k_rate ?? signals?.gold_accessed_rate ?? data.retrieval?.gold_accessed_rate ?? null
    const retrievalLabel =
        data.retrieval?.hit_at_k_rate !== null && data.retrieval?.hit_at_k_rate !== undefined
            ? 'Hit@5'
            : 'Gold Accessed'
    const correctRate = signals?.judge_correct_rate ?? data.judge?.correct_rate ?? null
    const groundedRate = signals?.judge_grounded_rate ?? data.judge?.grounded_rate ?? null
    const taxonomySuccessRate =
        signals?.taxonomy_success_rate ?? data.taxonomy_success_rate ?? taxonomySuccessFromCounts(data.taxonomy)
    const taxonomyEntries = Object.entries(data.taxonomy ?? {})

    return (
        <div className="rounded-xl border-2 border-indigo-500/20 bg-indigo-500/5 p-4 sm:p-5 space-y-4">
            <div className="flex items-center gap-2">
                <Scale className="h-4 w-4 text-indigo-500" />
                <h4 className="text-sm font-bold uppercase tracking-wider text-indigo-500">Legal RAG Bench</h4>
                {data.count !== undefined && (
                    <span className="text-[10px] font-semibold text-muted-foreground">{data.count} questions</span>
                )}
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-4 gap-4">
                <div className="rounded-xl border border-indigo-500/20 bg-card p-4">
                    <p className="text-[10px] sm:text-xs font-semibold uppercase tracking-wider text-muted-foreground">{retrievalLabel}</p>
                    <p className="text-xl sm:text-2xl font-black mt-1 text-indigo-500">{formatRate(retrievalRate)}</p>
                </div>
                <div className="rounded-xl border border-indigo-500/20 bg-card p-4">
                    <p className="text-[10px] sm:text-xs font-semibold uppercase tracking-wider text-muted-foreground">G-Eval</p>
                    <p className="text-xl sm:text-2xl font-black mt-1 text-indigo-500">{formatRate(signals?.g_eval_pass_rate)}</p>
                </div>
                <div className="rounded-xl border border-indigo-500/20 bg-card p-4">
                    <p className="text-[10px] sm:text-xs font-semibold uppercase tracking-wider text-muted-foreground">Judge Correct</p>
                    <p className="text-xl sm:text-2xl font-black mt-1 text-emerald-500">{formatRate(correctRate)}</p>
                </div>
                <div className="rounded-xl border border-indigo-500/20 bg-card p-4">
                    <p className="text-[10px] sm:text-xs font-semibold uppercase tracking-wider text-muted-foreground">Grounded</p>
                    <p className="text-xl sm:text-2xl font-black mt-1 text-emerald-500">{formatRate(groundedRate)}</p>
                </div>
                <div className="rounded-xl border border-indigo-500/20 bg-card p-4">
                    <p className="text-[10px] sm:text-xs font-semibold uppercase tracking-wider text-muted-foreground">Taxonomy Success</p>
                    <p className="text-xl sm:text-2xl font-black mt-1 text-emerald-500">{formatRate(taxonomySuccessRate)}</p>
                </div>
                <div className="rounded-xl border border-indigo-500/20 bg-card p-4">
                    <p className="text-[10px] sm:text-xs font-semibold uppercase tracking-wider text-muted-foreground">Alt Evidence</p>
                    <p className="text-xl sm:text-2xl font-black mt-1 text-sky-500">{formatRate(signals?.alternate_evidence_supported_rate)}</p>
                </div>
                <div className="rounded-xl border border-indigo-500/20 bg-card p-4">
                    <p className="text-[10px] sm:text-xs font-semibold uppercase tracking-wider text-muted-foreground">Correct w/o Gold</p>
                    <p className="text-xl sm:text-2xl font-black mt-1 text-sky-500">{formatRate(signals?.correct_without_gold_rate)}</p>
                </div>
                {(data.judge?.parse_error_count ?? 0) > 0 && (
                    <div className="rounded-xl border border-zinc-500/20 bg-card p-4">
                        <p className="text-[10px] sm:text-xs font-semibold uppercase tracking-wider text-muted-foreground">Judge Errors</p>
                        <p className="text-xl sm:text-2xl font-black mt-1 text-zinc-500">{data.judge?.parse_error_count}</p>
                    </div>
                )}
            </div>

            {taxonomyEntries.length > 0 && (
                <div className="space-y-2">
                    <p className="text-[10px] sm:text-xs font-bold uppercase tracking-wider text-muted-foreground">Taxonomy</p>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                        {taxonomyEntries.map(([key, value]) => {
                            const style = taxonomyStyle(key)
                            return (
                                <div key={key} className={cn('rounded-lg border p-3', style.bg, style.border)}>
                                    <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">{style.label}</p>
                                    <p className={cn('text-lg font-black mt-0.5', style.color)}>{value}</p>
                                </div>
                            )
                        })}
                    </div>
                </div>
            )}
        </div>
    )
}

/** Per-result benchmark panel rendered in the result detail overview tab. */
export function LegalRagResultMetrics({
    evaluationId,
    resultId,
}: {
    evaluationId: string
    resultId: string
}) {
    const { data } = useQuery({
        queryKey: ['legal-rag-raw-metrics', evaluationId, resultId],
        queryFn: () => api.evaluations.getRawMetrics(evaluationId, resultId),
    })

    const legal = (data?.data?.legal_rag_bench as LegalRagResult | undefined) ?? undefined
    if (!legal) return null

    const retrieval = legal.retrieval ?? undefined
    const judge = legal.judge ?? undefined
    const signals = legal.success_signals ?? undefined
    const tax = legal.taxonomy ? taxonomyStyle(legal.taxonomy) : null

    return (
        <div className="rounded-xl border border-indigo-500/20 bg-indigo-500/5 p-4 space-y-3">
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <Scale className="h-4 w-4 text-indigo-500" />
                    <h4 className="text-sm font-bold uppercase tracking-wider text-indigo-500">Legal RAG Bench</h4>
                </div>
                {tax && (
                    <span className={cn('rounded-full border px-3 py-1 text-[10px] font-bold uppercase tracking-wider', tax.bg, tax.border, tax.color)}>
                        {tax.label}
                    </span>
                )}
            </div>

            <div className="flex flex-wrap gap-2">
                {signals?.g_eval_pass !== undefined && (
                    <BoolBadge value={signals.g_eval_pass} label="G-Eval" />
                )}
                {retrieval && retrieval.hit_at_k !== null && (
                    <BoolBadge value={retrieval.hit_at_k} label={retrieval.retrieval_metric} />
                )}
                {retrieval && retrieval.hit_at_k === null && (
                    <BoolBadge value={retrieval.gold_accessed} label="Gold Accessed" />
                )}
                {judge && legal.taxonomy === 'abstention' && <AbstainedBadge />}
                {judge && legal.taxonomy === 'judge_error' && (
                    <BoolBadge value={null} label="Judge Error" />
                )}
                {judge && legal.taxonomy !== 'abstention' && legal.taxonomy !== 'judge_error' && (
                    <>
                        <BoolBadge value={judge.correct} label="Correct" />
                        <BoolBadge value={judge.grounded} label="Grounded" />
                    </>
                )}
                {signals?.alternate_evidence_supported !== undefined && (
                    <BoolBadge
                        value={signals.alternate_evidence_supported}
                        label="Alt Evidence"
                        neutralFalse
                    />
                )}
                {signals?.correct_without_gold === true && (
                    <BoolBadge value={signals.correct_without_gold} label="Correct w/o Gold" />
                )}
            </div>

            {retrieval && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                    <div>
                        <p className="font-semibold uppercase tracking-wider text-muted-foreground">Gold Passage</p>
                        <p className="font-mono mt-0.5 break-all">{retrieval.relevant_passage_id ?? 'N/A'}</p>
                    </div>
                    <div>
                        <p className="font-semibold uppercase tracking-wider text-muted-foreground">
                            Gold Rank {retrieval.gold_access_rank ? `(of top ${retrieval.top_k})` : ''}
                        </p>
                        <p className="font-mono mt-0.5">{retrieval.gold_access_rank ?? 'not retrieved'}</p>
                    </div>
                    {retrieval.retrieved_passage_ids.length > 0 && (
                        <div className="sm:col-span-2">
                            <p className="font-semibold uppercase tracking-wider text-muted-foreground">Retrieved Passage IDs</p>
                            <p className="font-mono mt-0.5 break-all">{retrieval.retrieved_passage_ids.join(', ')}</p>
                        </div>
                    )}
                </div>
            )}

            {signals && (
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
                    <div>
                        <p className="font-semibold uppercase tracking-wider text-muted-foreground">G-Eval Score</p>
                        <p className="font-mono mt-0.5">{formatScore(signals.g_eval_score)}</p>
                    </div>
                    <div>
                        <p className="font-semibold uppercase tracking-wider text-muted-foreground">Taxonomy Success</p>
                        <p className="font-mono mt-0.5">{boolText(signals.taxonomy_success)}</p>
                    </div>
                    <div>
                        <p className="font-semibold uppercase tracking-wider text-muted-foreground">Context Support</p>
                        <p className="font-mono mt-0.5">{boolText(signals.supported_by_retrieved_context)}</p>
                    </div>
                </div>
            )}

            {judge?.reasoning && (
                <div className="text-xs">
                    <p className="font-semibold uppercase tracking-wider text-muted-foreground">Judge Reasoning</p>
                    <p className="mt-0.5 text-muted-foreground">{judge.reasoning}</p>
                </div>
            )}
            {judge?.parse_error && (
                <div className="text-xs">
                    <p className="font-semibold uppercase tracking-wider text-muted-foreground">Judge Error</p>
                    <p className="mt-0.5 text-muted-foreground">{judge.parse_error}</p>
                </div>
            )}
        </div>
    )
}

function taxonomySuccessFromCounts(taxonomy: Record<string, number> | undefined): number | null {
    const entries = Object.values(taxonomy ?? {})
    const total = entries.reduce((sum, value) => sum + value, 0)
    return total > 0 ? (taxonomy?.success ?? 0) / total : null
}

function boolText(value: boolean | null | undefined) {
    if (value === true) return 'true'
    if (value === false) return 'false'
    return 'N/A'
}
