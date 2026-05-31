import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { GitCompare, Plus, Loader2, Calendar, Trash2, ChevronRight } from 'lucide-react'
import { api } from '@/api/client'
import { useToast } from '@/components/ui/toast-context'
import { CreateComparisonDialog } from './CreateComparisonDialog'

export function ComparisonsTab({ projectId }: { projectId: string }) {
    const [isDialogOpen, setIsDialogOpen] = useState(false)
    const navigate = useNavigate()
    const queryClient = useQueryClient()
    const { success, error } = useToast()

    const { data, isLoading } = useQuery({
        queryKey: ['comparisons', projectId],
        queryFn: () => api.comparisons.list(projectId),
        enabled: !!projectId,
    })

    const deleteMutation = useMutation({
        mutationFn: (id: string) => api.comparisons.delete(id),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['comparisons', projectId] })
            success('Comparison deleted', 'The comparison has been removed.')
        },
        onError: () => error('Failed to delete', 'Please try again.'),
    })

    if (isLoading) {
        return (
            <div className="flex justify-center py-20">
                <Loader2 className="h-8 w-8 animate-spin text-primary/50" />
            </div>
        )
    }

    const comparisons = data?.data?.items ?? []

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-xl font-semibold">Comparisons</h2>
                    <p className="text-sm text-muted-foreground">Compare metrics and answers across two or more evaluations.</p>
                </div>
                {comparisons.length > 0 && (
                    <button
                        onClick={() => setIsDialogOpen(true)}
                        className="flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-all shadow-md"
                    >
                        <Plus className="h-4 w-4" />
                        New Comparison
                    </button>
                )}
            </div>

            {comparisons.length === 0 ? (
                <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-border py-20 bg-card/50">
                    <div className="rounded-full bg-primary/10 p-5 text-primary">
                        <GitCompare className="h-10 w-10" />
                    </div>
                    <h3 className="mt-5 text-xl font-semibold">No comparisons yet</h3>
                    <p className="mt-2 max-w-sm text-center text-muted-foreground">
                        Pick two or more completed evaluations to see how their metrics, costs, and answers stack up.
                    </p>
                    <button
                        onClick={() => setIsDialogOpen(true)}
                        className="mt-6 flex items-center gap-2 rounded-lg bg-primary px-6 py-2.5 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-all shadow-md"
                    >
                        <Plus className="h-4 w-4" />
                        Create First Comparison
                    </button>
                </div>
            ) : (
                <div className="grid gap-4">
                    {comparisons.map((c) => {
                        const memberCount = 1 + (c.compared_evaluation_ids?.length ?? 0)
                        return (
                            <div
                                key={c.id}
                                className="group relative flex items-center justify-between rounded-xl border border-border bg-card p-4 hover:border-primary/50 hover:shadow-md transition-all cursor-pointer"
                                onClick={() => navigate(`/projects/${projectId}/comparisons/${c.id}`)}
                            >
                                <div className="flex items-center gap-4">
                                    <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
                                        <GitCompare className="h-5 w-5" />
                                    </div>
                                    <div>
                                        <p className="font-bold">{c.name || `Comparison #${c.id.slice(0, 8)}`}</p>
                                        <div className="mt-1 flex items-center gap-3 text-xs font-medium text-muted-foreground">
                                            <div className="flex items-center gap-1">
                                                <Calendar className="h-3 w-3" />
                                                {new Date(c.created_at).toLocaleString()}
                                            </div>
                                            <div>{memberCount} evaluations</div>
                                        </div>
                                    </div>
                                </div>
                                <div className="flex items-center gap-2">
                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation()
                                            deleteMutation.mutate(c.id)
                                        }}
                                        className="rounded-lg p-2 text-muted-foreground hover:bg-destructive/10 hover:text-destructive transition-colors"
                                        title="Delete comparison"
                                    >
                                        <Trash2 className="h-4 w-4" />
                                    </button>
                                    <div className="rounded-full bg-muted/50 p-2 group-hover:bg-primary group-hover:text-primary-foreground transition-all">
                                        <ChevronRight className="h-4 w-4" />
                                    </div>
                                </div>
                            </div>
                        )
                    })}
                </div>
            )}

            <CreateComparisonDialog
                projectId={projectId}
                isOpen={isDialogOpen}
                onClose={() => setIsDialogOpen(false)}
                onCreated={(id) => {
                    setIsDialogOpen(false)
                    queryClient.invalidateQueries({ queryKey: ['comparisons', projectId] })
                    navigate(`/projects/${projectId}/comparisons/${id}`)
                }}
            />
        </div>
    )
}
