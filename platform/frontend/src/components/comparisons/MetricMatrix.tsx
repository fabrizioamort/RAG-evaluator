import { TrendingDown, TrendingUp, Minus, Crown } from 'lucide-react'
import { cn } from '@/lib/utils'
import { ComparisonMember, METRIC_ROWS, computeDelta, bestMemberIndex } from './compare-utils'

interface MetricMatrixProps {
    members: ComparisonMember[]
    baselineId: string
}

export function MetricMatrix({ members, baselineId }: MetricMatrixProps) {
    const baseline = members.find((m) => m.id === baselineId) ?? members[0]

    return (
        <div className="overflow-x-auto rounded-xl border border-border">
            <table className="w-full border-collapse text-sm">
                <thead>
                    <tr className="border-b border-border bg-muted/40">
                        <th className="sticky left-0 z-10 bg-muted/40 px-4 py-3 text-left text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
                            Metric
                        </th>
                        {members.map((m) => (
                            <th key={m.id} className="px-4 py-3 text-left min-w-[160px]">
                                <div className="flex items-center gap-2">
                                    <span className="font-bold truncate max-w-[180px]">{m.label}</span>
                                    {m.id === baseline.id && (
                                        <span className="rounded-full bg-primary/15 px-2 py-0.5 text-[9px] font-bold uppercase tracking-tight text-primary">
                                            Baseline
                                        </span>
                                    )}
                                </div>
                                {m.ragConfigName && (
                                    <p className="text-[10px] font-medium text-muted-foreground truncate max-w-[180px]">{m.ragConfigName}</p>
                                )}
                            </th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {METRIC_ROWS.map((row) => {
                        const baseVal = row.get(baseline)
                        const best = bestMemberIndex(members, row)
                        return (
                            <tr key={row.key} className="border-b border-border/60 last:border-0 hover:bg-muted/20">
                                <td className="sticky left-0 z-10 bg-card px-4 py-3 text-[11px] font-bold uppercase tracking-wider text-muted-foreground whitespace-nowrap">
                                    {row.label}
                                </td>
                                {members.map((m, i) => {
                                    const val = row.get(m)
                                    const isBaselineCol = m.id === baseline.id
                                    const delta = isBaselineCol ? null : computeDelta(baseVal, val, row.higherIsBetter)
                                    const isBest = best === i && members.length > 1
                                    return (
                                        <td key={m.id} className="px-4 py-3">
                                            <div className="flex items-center gap-2">
                                                <span className={cn('text-base font-black tabular-nums tracking-tight', isBest && 'text-primary')}>
                                                    {row.format(val)}
                                                </span>
                                                {isBest && members.length > 1 && (
                                                    <Crown className="h-3 w-3 text-primary" />
                                                )}
                                            </div>
                                            {delta && (
                                                <DeltaBadge
                                                    absolute={delta.absolute}
                                                    percentage={delta.percentage}
                                                    improved={delta.improved}
                                                    format={row.format}
                                                />
                                            )}
                                        </td>
                                    )
                                })}
                            </tr>
                        )
                    })}
                </tbody>
            </table>
        </div>
    )
}

function DeltaBadge({
    absolute,
    percentage,
    improved,
    format,
}: {
    absolute: number
    percentage: number | null
    improved: boolean | null
    format: (v: number | null) => string
}) {
    const sign = absolute > 0 ? '+' : ''
    return (
        <div
            className={cn(
                'mt-0.5 flex items-center gap-1 text-[10px] font-bold',
                improved === true ? 'text-green-500' : improved === false ? 'text-red-500' : 'text-muted-foreground',
            )}
        >
            {improved === true && <TrendingUp className="h-3 w-3" />}
            {improved === false && <TrendingDown className="h-3 w-3" />}
            {improved === null && <Minus className="h-3 w-3" />}
            <span className="tabular-nums">
                {sign}
                {format(absolute)}
                {percentage !== null && ` (${sign}${percentage.toFixed(1)}%)`}
            </span>
        </div>
    )
}
