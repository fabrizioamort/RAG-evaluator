import { AggregateMetrics, EvaluationComparisonResult, SummaryMetrics } from '@/api/client'
import type { LegalRagBenchSummaryData } from '../evaluations/LegalRagBenchMetrics'

/** A single evaluation normalized into a comparable shape (baseline + compared share this). */
export interface ComparisonMember {
    id: string
    label: string
    ragConfigName?: string | null
    isStoredBaseline: boolean
    summary?: Record<string, number | null | undefined> | null
    cost?: Record<string, number | string | null | undefined> | null
    performance?: Record<string, number | null | undefined> | null
    passRate?: number | null
    legalRagBench?: LegalRagBenchSummaryData | null
}

/** Pull the Legal RAG Bench summary out of an evaluation's summary metrics, if present. */
function legalRagBenchOf(summary?: SummaryMetrics | null): LegalRagBenchSummaryData | null {
    const data = summary?.legal_rag_bench
    return data && typeof data === 'object' ? (data as LegalRagBenchSummaryData) : null
}

/** Build the ordered member list from a comparison's aggregate metrics. */
export function buildMembers(agg: AggregateMetrics | null | undefined): ComparisonMember[] {
    if (!agg) return []
    const baseline: ComparisonMember = {
        id: agg.baseline_evaluation_id,
        label: memberLabel(agg.baseline_evaluation_name, agg.baseline_rag_config_name, agg.baseline_evaluation_id),
        ragConfigName: agg.baseline_rag_config_name,
        isStoredBaseline: true,
        summary: agg.baseline_summary as ComparisonMember['summary'],
        cost: agg.baseline_cost as ComparisonMember['cost'],
        performance: agg.baseline_performance as ComparisonMember['performance'],
        passRate: agg.baseline_pass_rate,
        legalRagBench: legalRagBenchOf(agg.baseline_summary),
    }
    const compared = (agg.comparison_results ?? []).map((r: EvaluationComparisonResult) => ({
        id: r.evaluation_id,
        label: memberLabel(r.evaluation_name, r.rag_config_name, r.evaluation_id),
        ragConfigName: r.rag_config_name,
        isStoredBaseline: false,
        summary: r.summary_metrics as ComparisonMember['summary'],
        cost: r.cost_metrics as ComparisonMember['cost'],
        performance: r.performance_metrics as ComparisonMember['performance'],
        passRate: r.pass_rate,
        legalRagBench: legalRagBenchOf(r.summary_metrics),
    }))
    return [baseline, ...compared]
}

function memberLabel(name?: string | null, ragConfig?: string | null, id?: string): string {
    if (name && name.trim()) return name
    if (ragConfig && ragConfig.trim()) return ragConfig
    return `#${(id ?? '').slice(0, 8)}`
}

export interface MetricRow {
    key: string
    label: string
    group: 'quality' | 'cost' | 'performance'
    isScore: boolean // 0-1 score metric (usable in radar/percent charts)
    higherIsBetter: boolean
    get: (m: ComparisonMember) => number | null
    format: (v: number | null) => string
}

const toNum = (v: unknown): number | null => {
    if (v === null || v === undefined || v === '') return null
    const n = Number(v)
    return Number.isFinite(n) ? n : null
}

const fmtScore = (v: number | null) => (v === null ? '—' : v.toFixed(3))
const fmtPct = (v: number | null) => (v === null ? '—' : `${(v * 100).toFixed(1)}%`)
const fmtSecs = (v: number | null) => (v === null ? '—' : `${v.toFixed(2)}s`)
const fmtCost = (v: number | null) => (v === null ? '—' : `$${v.toFixed(4)}`)
const fmtInt = (v: number | null) => (v === null ? '—' : Math.round(v).toLocaleString())

export const METRIC_ROWS: MetricRow[] = [
    { key: 'faithfulness', label: 'Faithfulness', group: 'quality', isScore: true, higherIsBetter: true, get: (m) => toNum(m.summary?.faithfulness_avg), format: fmtScore },
    { key: 'relevancy', label: 'Relevancy', group: 'quality', isScore: true, higherIsBetter: true, get: (m) => toNum(m.summary?.relevancy_avg), format: fmtScore },
    { key: 'precision', label: 'Precision', group: 'quality', isScore: true, higherIsBetter: true, get: (m) => toNum(m.summary?.precision_avg), format: fmtScore },
    { key: 'recall', label: 'Recall', group: 'quality', isScore: true, higherIsBetter: true, get: (m) => toNum(m.summary?.recall_avg), format: fmtScore },
    { key: 'g_eval', label: 'Correctness', group: 'quality', isScore: true, higherIsBetter: true, get: (m) => toNum(m.summary?.g_eval_avg), format: fmtScore },
    { key: 'overall', label: 'Overall', group: 'quality', isScore: true, higherIsBetter: true, get: (m) => toNum(m.summary?.overall_avg), format: fmtScore },
    { key: 'pass_rate', label: 'Pass rate', group: 'quality', isScore: false, higherIsBetter: true, get: (m) => toNum(m.passRate), format: fmtPct },
    { key: 'avg_latency', label: 'Avg latency', group: 'performance', isScore: false, higherIsBetter: false, get: (m) => toNum(m.performance?.avg_latency_seconds), format: fmtSecs },
    { key: 'p95_latency', label: 'P95 latency', group: 'performance', isScore: false, higherIsBetter: false, get: (m) => toNum(m.performance?.p95_latency_seconds), format: fmtSecs },
    { key: 'total_cost', label: 'Total cost', group: 'cost', isScore: false, higherIsBetter: false, get: (m) => toNum(m.cost?.total_cost_usd), format: fmtCost },
    { key: 'avg_cost', label: 'Cost / query', group: 'cost', isScore: false, higherIsBetter: false, get: (m) => toNum(m.cost?.avg_cost_per_query), format: fmtCost },
    {
        key: 'total_tokens', label: 'Total tokens', group: 'cost', isScore: false, higherIsBetter: false,
        get: (m) => {
            const p = toNum(m.cost?.total_prompt_tokens)
            const c = toNum(m.cost?.total_completion_tokens)
            if (p === null && c === null) return null
            return (p ?? 0) + (c ?? 0)
        },
        format: fmtInt,
    },
]

export interface Delta {
    absolute: number
    percentage: number | null
    improved: boolean | null
}

const EPS = 1e-9

/** Direction-aware delta of `value` relative to `baseline`. Null when either side is missing. */
export function computeDelta(
    baseline: number | null,
    value: number | null,
    higherIsBetter: boolean,
): Delta | null {
    if (baseline === null || value === null) return null
    const absolute = value - baseline
    const percentage = Math.abs(baseline) > EPS ? (absolute / baseline) * 100 : null
    let improved: boolean | null = null
    if (Math.abs(absolute) > EPS) {
        improved = higherIsBetter ? absolute > 0 : absolute < 0
    }
    return { absolute, percentage, improved }
}

/** Index of the best member for a metric row (for highlighting). Null if none comparable. */
export function bestMemberIndex(members: ComparisonMember[], row: MetricRow): number | null {
    let best: number | null = null
    let bestVal: number | null = null
    members.forEach((m, i) => {
        const v = row.get(m)
        if (v === null) return
        if (bestVal === null || (row.higherIsBetter ? v > bestVal : v < bestVal)) {
            bestVal = v
            best = i
        }
    })
    return best
}
