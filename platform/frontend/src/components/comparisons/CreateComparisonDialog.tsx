import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { X, Loader2, AlertTriangle, Check } from 'lucide-react'
import { cn } from '@/lib/utils'
import { api, Evaluation } from '@/api/client'

interface CreateComparisonDialogProps {
    projectId: string
    isOpen: boolean
    onClose: () => void
    onCreated: (comparisonId: string) => void
}

export function CreateComparisonDialog({ projectId, isOpen, onClose, onCreated }: CreateComparisonDialogProps) {
    const [name, setName] = useState('')
    const [selected, setSelected] = useState<string[]>([])
    const [isSubmitting, setIsSubmitting] = useState(false)
    const [error, setError] = useState<string | null>(null)

    const { data, isLoading } = useQuery({
        queryKey: ['evaluations', projectId],
        queryFn: () => api.evaluations.list(projectId),
        enabled: isOpen && !!projectId,
    })

    if (!isOpen) return null

    const completed = (data?.data?.items ?? []).filter((e) => e.status === 'completed')
    const selectedEvals = selected.map((id) => completed.find((e) => e.id === id)).filter(Boolean) as Evaluation[]
    const testSetIds = new Set(selectedEvals.map((e) => e.test_set_id).filter(Boolean))
    const mixedTestSets = testSetIds.size > 1

    const toggle = (id: string) =>
        setSelected((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]))

    const handleSubmit = async () => {
        if (selected.length < 2) return
        setIsSubmitting(true)
        setError(null)
        try {
            const res = await api.comparisons.create({
                name: name.trim() || undefined,
                baseline_evaluation_id: selected[0],
                compared_evaluation_ids: selected.slice(1),
            })
            setName('')
            setSelected([])
            onCreated(res.data.id)
        } catch (err) {
            console.error('Failed to create comparison:', err)
            setError('Failed to create comparison. Please try again.')
        } finally {
            setIsSubmitting(false)
        }
    }

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <div className="absolute inset-0 bg-background/80 backdrop-blur-sm animate-in fade-in" onClick={onClose} />
            <div className="relative flex max-h-[85vh] w-full max-w-2xl flex-col rounded-xl border border-border bg-card shadow-2xl animate-in zoom-in-95 duration-200">
                <div className="flex items-center justify-between border-b border-border p-6">
                    <div>
                        <h2 className="text-2xl font-bold tracking-tight">New Comparison</h2>
                        <p className="mt-1 text-sm text-muted-foreground">Select 2 or more completed evaluations. The first selected is the baseline (you can change it later).</p>
                    </div>
                    <button onClick={onClose} className="rounded-full p-2 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors">
                        <X className="h-5 w-5" />
                    </button>
                </div>

                <div className="flex-1 space-y-4 overflow-y-auto p-6">
                    <div className="space-y-2">
                        <label htmlFor="cmp-name" className="text-sm font-semibold">Name (optional)</label>
                        <input
                            id="cmp-name"
                            value={name}
                            onChange={(e) => setName(e.target.value)}
                            placeholder="e.g., Hybrid vs Semantic on Legal KB"
                            className="w-full rounded-lg border border-input bg-background px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
                        />
                    </div>

                    {mixedTestSets && (
                        <div className="flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-amber-700">
                            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                            <span>Selected evaluations use different test sets. Aggregate metrics still compare, but the per-question breakdown will only show overlapping questions.</span>
                        </div>
                    )}

                    {isLoading ? (
                        <div className="flex justify-center py-10"><Loader2 className="h-6 w-6 animate-spin text-primary/50" /></div>
                    ) : completed.length < 2 ? (
                        <p className="py-6 text-center text-sm text-muted-foreground">You need at least 2 completed evaluations in this project to create a comparison.</p>
                    ) : (
                        <div className="space-y-2">
                            {completed.map((e) => {
                                const idx = selected.indexOf(e.id)
                                const isSelected = idx >= 0
                                return (
                                    <button
                                        key={e.id}
                                        onClick={() => toggle(e.id)}
                                        className={cn(
                                            'flex w-full items-center gap-3 rounded-lg border p-3 text-left transition-all',
                                            isSelected ? 'border-primary bg-primary/5' : 'border-border hover:border-primary/40',
                                        )}
                                    >
                                        <div className={cn('flex h-5 w-5 shrink-0 items-center justify-center rounded border', isSelected ? 'border-primary bg-primary text-primary-foreground' : 'border-muted-foreground/40')}>
                                            {isSelected && <Check className="h-3.5 w-3.5" />}
                                        </div>
                                        <div className="min-w-0 flex-1">
                                            <div className="flex items-center gap-2">
                                                <span className="truncate font-semibold">{e.name || `Evaluation #${e.id.slice(0, 8)}`}</span>
                                                {idx === 0 && <span className="rounded-full bg-primary/15 px-2 py-0.5 text-[9px] font-bold uppercase text-primary">Baseline</span>}
                                            </div>
                                            <span className="text-xs text-muted-foreground">{new Date(e.created_at).toLocaleString()}</span>
                                        </div>
                                        <div className="shrink-0 text-right">
                                            {e.pass_rate !== null && <p className="text-sm font-black tabular-nums">{(e.pass_rate * 100).toFixed(0)}%</p>}
                                            {e.summary_metrics?.overall_avg !== undefined && <p className="text-[10px] text-muted-foreground">avg {e.summary_metrics.overall_avg.toFixed(2)}</p>}
                                        </div>
                                    </button>
                                )
                            })}
                        </div>
                    )}

                    {error && <p className="text-sm text-destructive">{error}</p>}
                </div>

                <div className="flex items-center justify-between border-t border-border p-6">
                    <span className="text-xs text-muted-foreground">{selected.length} selected{selected.length > 10 ? ' (max 10 compared)' : ''}</span>
                    <div className="flex gap-3">
                        <button onClick={onClose} className="rounded-lg px-6 py-2.5 text-sm font-semibold hover:bg-muted transition-colors">Cancel</button>
                        <button
                            onClick={handleSubmit}
                            disabled={isSubmitting || selected.length < 2 || selected.length > 11}
                            className="flex items-center gap-2 rounded-lg bg-primary px-8 py-2.5 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-all shadow-md active:scale-95 disabled:opacity-50 disabled:pointer-events-none"
                        >
                            {isSubmitting ? <><Loader2 className="h-4 w-4 animate-spin" />Creating...</> : 'Compare'}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    )
}
