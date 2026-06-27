import { Scale, Crown } from 'lucide-react'
import { cn } from '@/lib/utils'
import { ComparisonMember } from './compare-utils'

interface LegalRagBenchComparisonProps {
    members: ComparisonMember[]
}

const formatRate = (val: number | null | undefined) =>
    val === null || val === undefined ? '—' : `${(val * 100).toFixed(1)}%`

/** Headline rate rows compared across members (higher is better). */
const RATE_ROWS: { key: string; label: string; get: (m: ComparisonMember) => number | null }[] = [
    { key: 'hit_at_k', label: 'Hit@5', get: (m) => m.legalRagBench?.retrieval?.hit_at_k_rate ?? null },
    { key: 'gold_accessed', label: 'Gold accessed', get: (m) => m.legalRagBench?.retrieval?.gold_accessed_rate ?? null },
    { key: 'correct', label: 'Correct', get: (m) => m.legalRagBench?.judge?.correct_rate ?? null },
    { key: 'grounded', label: 'Grounded', get: (m) => m.legalRagBench?.judge?.grounded_rate ?? null },
]

const TAXONOMY_ROWS: { key: string; label: string; color: string }[] = [
    { key: 'success', label: 'Success', color: 'text-emerald-500' },
    { key: 'reasoning_error', label: 'Reasoning Error', color: 'text-amber-500' },
    { key: 'retrieval_error', label: 'Retrieval Error', color: 'text-orange-500' },
    { key: 'hallucination_or_ungrounded', label: 'Hallucination / Ungrounded', color: 'text-rose-500' },
    { key: 'abstention', label: 'Abstention', color: 'text-sky-500' },
]

/** Index of the member with the highest value for a rate row (null if none comparable). */
function bestRateIndex(members: ComparisonMember[], get: (m: ComparisonMember) => number | null): number | null {
    let best: number | null = null
    let bestVal = -Infinity
    members.forEach((m, i) => {
        const v = get(m)
        if (v === null) return
        if (v > bestVal) {
            bestVal = v
            best = i
        }
    })
    return best
}

/** Side-by-side Legal RAG Bench retrieval, judge, and taxonomy metrics across evaluations. */
export function LegalRagBenchComparison({ members }: LegalRagBenchComparisonProps) {
    const withData = members.filter((m) => m.legalRagBench)
    if (withData.length === 0) return null

    // Only show taxonomy categories that appear in at least one member.
    const taxonomyRows = TAXONOMY_ROWS.filter((row) =>
        members.some((m) => (m.legalRagBench?.taxonomy?.[row.key] ?? 0) > 0),
    )

    return (
        <div className="space-y-4">
            <div className="flex items-center gap-2">
                <Scale className="h-4 w-4 text-indigo-500" />
                <h3 className="text-sm font-bold uppercase tracking-wider text-indigo-500">Legal RAG Bench</h3>
            </div>

            <div className="overflow-x-auto rounded-xl border border-indigo-500/20">
                <table className="w-full border-collapse text-sm">
                    <thead>
                        <tr className="border-b border-border bg-indigo-500/5">
                            <th className="sticky left-0 z-10 bg-indigo-500/5 px-4 py-3 text-left text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
                                Metric
                            </th>
                            {members.map((m) => (
                                <th key={m.id} className="px-4 py-3 text-left min-w-[140px]">
                                    <span className="font-bold truncate max-w-[180px] block">{m.label}</span>
                                    {m.ragConfigName && (
                                        <p className="text-[10px] font-medium text-muted-foreground truncate max-w-[180px]">{m.ragConfigName}</p>
                                    )}
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {RATE_ROWS.map((row) => {
                            const best = bestRateIndex(members, row.get)
                            return (
                                <tr key={row.key} className="border-b border-border/60 last:border-0 hover:bg-muted/20">
                                    <td className="sticky left-0 z-10 bg-card px-4 py-3 text-[11px] font-bold uppercase tracking-wider text-muted-foreground whitespace-nowrap">
                                        {row.label}
                                    </td>
                                    {members.map((m, i) => {
                                        const v = row.get(m)
                                        return (
                                            <td key={m.id} className="px-4 py-3">
                                                <span className={cn('font-semibold tabular-nums', i === best && v !== null && 'text-indigo-500')}>
                                                    {formatRate(v)}
                                                </span>
                                                {i === best && v !== null && members.length > 1 && (
                                                    <Crown className="ml-1 inline h-3 w-3 text-indigo-500" />
                                                )}
                                            </td>
                                        )
                                    })}
                                </tr>
                            )
                        })}

                        {taxonomyRows.length > 0 && (
                            <tr className="border-b border-border/60 bg-muted/30">
                                <td
                                    colSpan={members.length + 1}
                                    className="sticky left-0 px-4 py-2 text-[10px] font-bold uppercase tracking-widest text-muted-foreground"
                                >
                                    Taxonomy (questions)
                                </td>
                            </tr>
                        )}
                        {taxonomyRows.map((row) => (
                            <tr key={row.key} className="border-b border-border/60 last:border-0 hover:bg-muted/20">
                                <td className={cn('sticky left-0 z-10 bg-card px-4 py-3 text-[11px] font-bold uppercase tracking-wider whitespace-nowrap', row.color)}>
                                    {row.label}
                                </td>
                                {members.map((m) => {
                                    const count = m.legalRagBench?.taxonomy?.[row.key]
                                    return (
                                        <td key={m.id} className="px-4 py-3 font-semibold tabular-nums">
                                            {count ?? '—'}
                                        </td>
                                    )
                                })}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    )
}
