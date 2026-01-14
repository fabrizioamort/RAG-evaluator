import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
    Loader2,
    CheckCircle2,
    XCircle,
    AlertTriangle,
    Sparkles,
    Pause,
} from 'lucide-react'
import { api } from '@/api/client'
import { cn } from '@/lib/utils'

interface GenerationProgressProps {
    testSetId: string
    onComplete: () => void
    onClose: () => void
}

export function GenerationProgress({ testSetId, onComplete, onClose }: GenerationProgressProps) {
    const queryClient = useQueryClient()
    const [pollInterval, setPollInterval] = useState<number | false>(2000)

    // Fetch generation status
    const { data: statusData, isLoading, isError } = useQuery({
        queryKey: ['generation-status', testSetId],
        queryFn: () => api.testSets.getGenerationStatus(testSetId),
        refetchInterval: pollInterval,
        retry: false,
    })

    // Cancel mutation
    const cancelMutation = useMutation({
        mutationFn: () => api.testSets.cancelGeneration(testSetId),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['generation-status', testSetId] })
            queryClient.invalidateQueries({ queryKey: ['test-set', testSetId] })
        },
    })

    const status = statusData?.data

    // Stop polling when generation is complete or failed
    useEffect(() => {
        if (status?.status === 'completed' || status?.status === 'failed' || status?.status === 'cancelled') {
            setPollInterval(false)
            if (status?.status === 'completed') {
                // Delay slightly to show completion state
                setTimeout(() => {
                    onComplete()
                }, 1500)
            }
        }
    }, [status?.status, onComplete])

    const handleCancel = async () => {
        if (confirm('Are you sure you want to cancel the generation?')) {
            await cancelMutation.mutateAsync()
        }
    }

    const getStatusIcon = () => {
        switch (status?.status) {
            case 'running':
            case 'pending':
                return <Loader2 className="h-12 w-12 animate-spin text-primary" />
            case 'completed':
                return <CheckCircle2 className="h-12 w-12 text-green-500" />
            case 'failed':
                return <XCircle className="h-12 w-12 text-destructive" />
            case 'cancelled':
                return <Pause className="h-12 w-12 text-muted-foreground" />
            default:
                return <Sparkles className="h-12 w-12 text-primary" />
        }
    }

    const getStatusMessage = () => {
        switch (status?.status) {
            case 'pending':
                return 'Preparing generation...'
            case 'running':
                return 'Generating test cases...'
            case 'completed':
                return 'Generation complete!'
            case 'failed':
                return 'Generation failed'
            case 'cancelled':
                return 'Generation cancelled'
            default:
                return 'Loading...'
        }
    }

    const progressPercent = status ? Math.round(status.progress * 100) : 0

    if (isLoading) {
        return (
            <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
                <div className="absolute inset-0 bg-background/80 backdrop-blur-sm" />
                <div className="relative w-full max-w-md rounded-xl border border-border bg-card shadow-2xl p-8">
                    <div className="flex flex-col items-center justify-center gap-4">
                        <Loader2 className="h-10 w-10 animate-spin text-primary" />
                        <p className="text-muted-foreground font-medium">Loading status...</p>
                    </div>
                </div>
            </div>
        )
    }

    if (isError) {
        return (
            <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
                <div className="absolute inset-0 bg-background/80 backdrop-blur-sm" onClick={onClose} />
                <div className="relative w-full max-w-md rounded-xl border border-border bg-card shadow-2xl p-8">
                    <div className="flex flex-col items-center justify-center gap-4 text-center">
                        <AlertTriangle className="h-12 w-12 text-amber-500" />
                        <div>
                            <h3 className="text-lg font-bold">No Active Generation</h3>
                            <p className="text-sm text-muted-foreground mt-1">
                                There is no generation job running for this test set.
                            </p>
                        </div>
                        <button
                            onClick={onClose}
                            className="mt-4 rounded-lg bg-primary px-6 py-2 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-all"
                        >
                            Close
                        </button>
                    </div>
                </div>
            </div>
        )
    }

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <div
                className="absolute inset-0 bg-background/80 backdrop-blur-sm"
                onClick={status?.status === 'completed' || status?.status === 'failed' || status?.status === 'cancelled' ? onClose : undefined}
            />
            <div className="relative w-full max-w-md rounded-xl border border-border bg-card shadow-2xl animate-in zoom-in-95 duration-200">
                {/* Content */}
                <div className="p-8">
                    <div className="flex flex-col items-center justify-center gap-6 text-center">
                        {/* Status Icon */}
                        {getStatusIcon()}

                        {/* Status Message */}
                        <div>
                            <h3 className="text-xl font-bold">{getStatusMessage()}</h3>
                            {status?.error_message && (
                                <p className="text-sm text-destructive mt-2">
                                    {status.error_message}
                                </p>
                            )}
                        </div>

                        {/* Progress Bar */}
                        {(status?.status === 'running' || status?.status === 'pending') && (
                            <div className="w-full space-y-2">
                                <div className="h-3 w-full rounded-full bg-muted overflow-hidden">
                                    <div
                                        className="h-full bg-primary transition-all duration-500 ease-out rounded-full"
                                        style={{ width: `${progressPercent}%` }}
                                    />
                                </div>
                                <p className="text-sm text-muted-foreground">
                                    {progressPercent}% complete
                                </p>
                            </div>
                        )}

                        {/* Stats */}
                        <div className="grid grid-cols-3 gap-4 w-full rounded-xl bg-muted/50 p-4">
                            <div className="text-center">
                                <p className="text-2xl font-bold text-primary">
                                    {status?.questions_generated || 0}
                                </p>
                                <p className="text-[10px] uppercase font-bold tracking-wider text-muted-foreground">
                                    Generated
                                </p>
                            </div>
                            <div className="text-center">
                                <p className="text-2xl font-bold">
                                    {status?.questions_total || 0}
                                </p>
                                <p className="text-[10px] uppercase font-bold tracking-wider text-muted-foreground">
                                    Target
                                </p>
                            </div>
                            <div className="text-center">
                                <p className={cn(
                                    "text-2xl font-bold",
                                    (status?.questions_rejected || 0) > 0 ? "text-amber-500" : "text-muted-foreground"
                                )}>
                                    {status?.questions_rejected || 0}
                                </p>
                                <p className="text-[10px] uppercase font-bold tracking-wider text-muted-foreground">
                                    Rejected
                                </p>
                            </div>
                        </div>

                        {/* Quality Gate Info */}
                        {(status?.questions_rejected || 0) > 0 && (status?.status === 'running' || status?.status === 'completed') && (
                            <p className="text-xs text-muted-foreground">
                                Questions rejected by quality gates (duplicates, low quality, etc.)
                            </p>
                        )}
                    </div>
                </div>

                {/* Footer */}
                <div className="flex items-center justify-center gap-3 border-t border-border p-4 bg-muted/20 rounded-b-xl">
                    {(status?.status === 'running' || status?.status === 'pending') && (
                        <button
                            onClick={handleCancel}
                            disabled={cancelMutation.isPending}
                            className="flex items-center gap-2 rounded-lg border border-border bg-card px-6 py-2 text-sm font-semibold hover:bg-muted transition-colors disabled:opacity-50"
                        >
                            {cancelMutation.isPending ? (
                                <Loader2 className="h-4 w-4 animate-spin" />
                            ) : (
                                <XCircle className="h-4 w-4" />
                            )}
                            Cancel
                        </button>
                    )}

                    {(status?.status === 'completed' || status?.status === 'failed' || status?.status === 'cancelled') && (
                        <button
                            onClick={onClose}
                            className="flex items-center gap-2 rounded-lg bg-primary px-8 py-2 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-all"
                        >
                            {status?.status === 'completed' ? (
                                <>
                                    <CheckCircle2 className="h-4 w-4" />
                                    View Results
                                </>
                            ) : (
                                'Close'
                            )}
                        </button>
                    )}
                </div>
            </div>
        </div>
    )
}
