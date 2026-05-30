import { useMemo, useState } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'
import { cn } from '@/lib/utils'
import { PerQuestionDelta } from '@/api/client'
import { ComparisonMember } from './compare-utils'

interface PerQuestionTableProps {
    members: ComparisonMember[]
    storedBaselineId: string
    baselineId: string
    deltas: PerQuestionDelta[]
}

const SCORE_KEYS = [
    { key: 'faithfulness', label: 'Faith' },
    { key: 'relevancy', label: 'Rel' },
    { key: 'precision', label: 'Prec' },
    { key: 'recall', label: 'Recall' },
    { key: 'g_eval', label: 'Correct' },
]

const num = (v: unknown): number | null => {
    if (v === null || v === undefined || v === '') return null
    const n = Number(v)
    return Number.isFinite(n) ? n : null
}

function scoresFor(item: PerQuestionDelta, member: ComparisonMember, storedBaselineId: string): Record<string, unknown> | null {
    if (member.id === storedBaselineId) return item.baseline_result ?? null
    return item.compared_results?.[member.id] ?? null
}

function overall(raw: Record<string, unknown> | null): number | null {
    if (!raw) return null
    const vals = SCORE_KEYS.map((s) => num(raw[s.key])).filter((v): v is number => v !== null)
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null
}

function scoreColor(v: number | null): string {
    if (v === null) return 'bg-muted text-muted-foreground'
    if (v >= 0.7) return 'bg-green-500/15 text-green-600'
    if (v >= 0.4) return 'bg-amber-500/15 text-amber-600'
    return 'bg-red-500/15 text-red-600'
}

export function PerQuestionTable({ members, storedBaselineId, baselineId, deltas }: PerQuestionTableProps) {
    const [expanded, setExpanded] = useState<Set<string>>(new Set())
    const [sortByDelta, setSortByDelta] = useState(true)
    const [regressionsOnly, setRegressionsOnly] = useState(false)

    const rows = useMemo(() => {
        const enriched = deltas.map((item) => {
            const baselineMember = members.find((m) => m.id === baselineId) ?? members[0]
            const baseOverall = overall(scoresFor(item, baselineMember, storedBaselineId))
            let maxDrop = 0
            let maxSpread = 0
            members.forEach((m) => {
                const o = overall(scoresFor(item, m, storedBaselineId))
                if (o !== null && baseOverall !== null) {
                    maxSpread = Math.max(maxSpread, Math.abs(o - baseOverall))
                    maxDrop = Math.min(maxDrop, o - baseOverall)
                }
            })
            return { item, maxSpread, hasRegression: maxDrop < -1e-6 }
        })
        let filtered = regressionsOnly ? enriched.filter((e) => e.hasRegression) : enriched
        if (sortByDelta) filtered = [...filtered].sort((a, b) => b.maxSpread - a.maxSpread)
        return filtered
    }, [deltas, members, baselineId, storedBaselineId, sortByDelta, regressionsOnly])

    const toggle = (id: string) =>
        setExpanded((prev) => {
            const next = new Set(prev)
            next.has(id) ? next.delete(id) : next.add(id)
            return next
        })

    if (!deltas.length) {
        return <p className="text-sm text-muted-foreground">No per-question data available for this comparison.</p>
    }

    return (
        <div className="space-y-3">
            <div className="flex flex-wrap items-center gap-2">
                <FilterToggle active={sortByDelta} onClick={() => setSortByDelta((v) => !v)}>Sort by largest difference</FilterToggle>
                <FilterToggle active={regressionsOnly} onClick={() => setRegressionsOnly((v) => !v)}>Regressions only</FilterToggle>
                <span className="ml-auto text-xs text-muted-foreground">{rows.length} questions</span>
            </div>

            <div className="space-y-2">
                {rows.map(({ item }) => {
                    const isOpen = expanded.has(item.test_case_id)
                    return (
                        <div key={item.test_case_id} className="rounded-lg border border-border bg-card">
                            <button onClick={() => toggle(item.test_case_id)} className="flex w-full items-center gap-3 p-3 text-left hover:bg-muted/30">
                                {isOpen ? <ChevronDown className="h-4 w-4 shrink-0 text-muted-foreground" /> : <ChevronRight className="h-4 w-4 shrink-0 text-muted-foreground" />}
                                <span className="flex-1 truncate text-sm font-medium">{item.question || `Test case ${item.test_case_id.slice(0, 8)}`}</span>
                                <div className="flex shrink-0 items-center gap-1.5">
                                    {members.map((m) => {
                                        const o = overall(scoresFor(item, m, storedBaselineId))
                                        return (
                                            <span key={m.id} className={cn('rounded px-1.5 py-0.5 text-[10px] font-bold tabular-nums', scoreColor(o))} title={m.label}>
                                                {o === null ? '—' : o.toFixed(2)}
                                            </span>
                                        )
                                    })}
                                </div>
                            </button>

                            {isOpen && (
                                <div className="grid gap-3 border-t border-border p-3 md:grid-cols-2 lg:grid-cols-3">
                                    {members.map((m) => {
                                        const raw = scoresFor(item, m, storedBaselineId)
                                        return (
                                            <div key={m.id} className={cn('rounded-lg border p-3', m.id === baselineId ? 'border-primary/40 bg-primary/5' : 'border-border')}>
                                                <div className="mb-2 flex items-center justify-between">
                                                    <span className="truncate text-xs font-bold">{m.label}</span>
                                                    {m.id === baselineId && <span className="text-[9px] font-bold uppercase text-primary">Baseline</span>}
                                                </div>
                                                <div className="mb-2 flex flex-wrap gap-1">
                                                    {SCORE_KEYS.map((s) => {
                                                        const v = num(raw?.[s.key])
                                                        return (
                                                            <span key={s.key} className={cn('rounded px-1.5 py-0.5 text-[10px] font-semibold tabular-nums', scoreColor(v))}>
                                                                {s.label} {v === null ? '—' : v.toFixed(2)}
                                                            </span>
                                                        )
                                                    })}
                                                </div>
                                                <p className="whitespace-pre-wrap text-xs text-muted-foreground line-clamp-6">
                                                    {(raw?.generated_answer as string) || 'No answer recorded.'}
                                                </p>
                                            </div>
                                        )
                                    })}
                                </div>
                            )}
                        </div>
                    )
                })}
            </div>
        </div>
    )
}

function FilterToggle({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
    return (
        <button
            onClick={onClick}
            className={cn(
                'rounded-full border px-3 py-1 text-xs font-semibold transition-colors',
                active ? 'border-primary bg-primary/10 text-primary' : 'border-border text-muted-foreground hover:bg-muted',
            )}
        >
            {children}
        </button>
    )
}
