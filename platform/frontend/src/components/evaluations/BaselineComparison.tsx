import { TrendingDown, TrendingUp, Minus } from 'lucide-react'
import { Evaluation } from '../../api/client'
import { cn } from '@/lib/utils'

interface BaselineComparisonProps {
    current: Evaluation
    baseline: Evaluation
}

export function BaselineComparison({ current, baseline }: BaselineComparisonProps) {
    if (!current?.summary_metrics || !baseline?.summary_metrics) return null

    const metrics = [
        { key: 'faithfulness_avg', label: 'Faithfulness' },
        { key: 'relevancy_avg', label: 'Relevancy' },
        { key: 'precision_avg', label: 'Precision' },
        { key: 'recall_avg', label: 'Recall' },
        { key: 'overall_avg', label: 'Overall' },
    ] as const

    return (
        <div className="rounded-xl border border-primary/20 bg-primary/5 p-6 animate-in fade-in zoom-in-95 duration-500">
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h3 className="text-sm font-black uppercase tracking-widest text-primary">Baseline Comparison</h3>
                    <p className="text-xs text-muted-foreground mt-1">
                        Comparing against baseline from {baseline.created_at ? new Date(baseline.created_at).toLocaleDateString() : 'Unknown Date'}
                    </p>
                </div>
                {baseline.id === current.id && (
                    <span className="rounded-full bg-primary/20 px-3 py-1 text-[10px] font-bold text-primary border border-primary/30 uppercase tracking-tight">
                        Current is Baseline
                    </span>
                )}
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
                {metrics.map((m) => {
                    const currVal = current.summary_metrics ? current.summary_metrics[m.key] : undefined
                    const baseVal = baseline.summary_metrics ? baseline.summary_metrics[m.key] : undefined

                    if (currVal === undefined || currVal === null || baseVal === undefined || baseVal === null) return null

                    const delta = Number(currVal) - Number(baseVal)
                    const isPositive = delta > 0.005
                    const isNegative = delta < -0.005
                    const isNeutral = !isPositive && !isNegative

                    return (
                        <div key={m.key} className="space-y-2">
                            <p className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">{m.label}</p>
                            <div className="flex items-end gap-2">
                                <span className="text-xl font-black tabular-nums tracking-tighter">
                                    {Number(currVal).toFixed(2)}
                                </span>
                                <div className={cn(
                                    "flex items-center gap-0.5 text-[10px] font-bold pb-1",
                                    isPositive ? "text-green-500" : isNegative ? "text-red-500" : "text-muted-foreground"
                                )}>
                                    {isPositive && <TrendingUp className="h-3 w-3" />}
                                    {isNegative && <TrendingDown className="h-3 w-3" />}
                                    {isNeutral && <Minus className="h-3 w-3" />}
                                    <span>{delta > 0 ? '+' : ''}{delta.toFixed(2)}</span>
                                </div>
                            </div>
                            {/* Small progress bar comparison */}
                            <div className="h-1 w-full bg-muted rounded-full overflow-hidden flex">
                                <div
                                    className="h-full bg-primary/30"
                                    style={{ width: `${baseVal * 100}%` }}
                                />
                                <div
                                    className={cn(
                                        "h-full",
                                        isPositive ? "bg-green-500" : isNegative ? "bg-red-500" : "bg-primary"
                                    )}
                                    style={{
                                        width: `${Math.abs(delta) * 100}%`,
                                        marginLeft: isNegative ? `-${Math.abs(delta) * 100}%` : '0'
                                    }}
                                />
                            </div>
                        </div>
                    )
                })}
            </div>
        </div>
    )
}
