import { useState } from 'react'
import { Play, Pause, XCircle, CheckCircle2, AlertCircle, Clock, RefreshCcw } from 'lucide-react'
import { useEvaluationStream } from '../../hooks/useEvaluationStream'
import { api, type Evaluation } from '../../api/client'

interface EvaluationProgressProps {
    evaluationId: string
    evaluation?: Pick<Evaluation, 'status' | 'result_count' | 'error_message'>
    onClose?: () => void
    onRetryStarted?: () => void
}

export function EvaluationProgress({ evaluationId, evaluation, onClose, onRetryStarted }: EvaluationProgressProps) {
    const {
        completed: streamCompleted,
        total,
        status: streamStatus,
        currentQuestion,
        error: streamError,
        caseError,
        summaryMetrics,
        reconnect,
    } = useEvaluationStream(evaluationId)
    const [isRetrying, setIsRetrying] = useState(false)
    const [retryAccepted, setRetryAccepted] = useState(false)
    const [retryError, setRetryError] = useState<string | null>(null)

    const hasStreamState =
        streamStatus !== 'pending' ||
        total > 0 ||
        streamCompleted > 0 ||
        Boolean(streamError || caseError || summaryMetrics)
    const fallbackStatus = retryAccepted ? 'pending' : evaluation?.status
    const status = hasStreamState ? streamStatus : fallbackStatus ?? streamStatus
    const completed = Math.max(streamCompleted, evaluation?.result_count ?? 0)
    const displayError = retryError ?? streamError ?? (retryAccepted ? undefined : evaluation?.error_message) ?? undefined
    const showCaseError =
        Boolean(caseError) &&
        (status === 'running' || (status === 'failed' && !displayError?.includes(caseError ?? '')))
    const canRetry = status === 'failed' || status === 'cancelled'

    const progress = total > 0 ? (completed / total) * 100 : 0
    const progressLabel = total > 0
        ? `${completed} / ${total} test cases (${Math.round(progress)}%)`
        : `${completed} test cases saved`

    const handlePause = async () => {
        try {
            await api.evaluations.pause(evaluationId)
        } catch (err) {
            console.error('Failed to pause evaluation:', err)
        }
    }

    const handleResume = async () => {
        try {
            await api.evaluations.resume(evaluationId)
        } catch (err) {
            console.error('Failed to resume evaluation:', err)
        }
    }

    const handleCancel = async () => {
        if (window.confirm('Are you sure you want to cancel this evaluation?')) {
            try {
                await api.evaluations.cancel(evaluationId)
            } catch (err) {
                console.error('Failed to cancel evaluation:', err)
            }
        }
    }

    const handleRetry = async () => {
        if (!canRetry || isRetrying) return

        setIsRetrying(true)
        setRetryAccepted(false)
        setRetryError(null)
        try {
            await api.evaluations.retry(evaluationId)
            setRetryAccepted(true)
            onRetryStarted?.()
            reconnect()
        } catch (err) {
            console.error('Failed to retry evaluation:', err)
            setRetryError('Failed to retry evaluation. Please try again.')
        } finally {
            setIsRetrying(false)
        }
    }

    return (
        <div className="rounded-xl border border-border bg-card p-6 shadow-lg">
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h3 className="text-xl font-bold flex items-center gap-2">
                        {status === 'running' && <Clock className="h-5 w-5 text-blue-500 animate-spin" />}
                        {status === 'completed' && <CheckCircle2 className="h-5 w-5 text-green-500" />}
                        {status === 'failed' && <AlertCircle className="h-5 w-5 text-red-500" />}
                        {status === 'paused' && <Pause className="h-5 w-5 text-yellow-500" />}
                        {status === 'cancelled' && <XCircle className="h-5 w-5 text-gray-500" />}
                        Evaluation Status: <span className="capitalize">{status}</span>
                    </h3>
                    <p className="text-sm text-muted-foreground mt-1">
                        ID: {evaluationId}
                    </p>
                </div>

                <div className="flex gap-2">
                    {status === 'running' && (
                        <button
                            onClick={handlePause}
                            className="flex items-center gap-2 rounded-lg bg-yellow-500 px-4 py-2 text-sm font-medium text-white hover:bg-yellow-600 transition-colors"
                        >
                            <Pause className="h-4 w-4" /> Pause
                        </button>
                    )}
                    {status === 'paused' && (
                        <button
                            onClick={handleResume}
                            className="flex items-center gap-2 rounded-lg bg-green-500 px-4 py-2 text-sm font-medium text-white hover:bg-green-600 transition-colors"
                        >
                            <Play className="h-4 w-4" /> Resume
                        </button>
                    )}
                    {(status === 'running' || status === 'paused') && (
                        <button
                            onClick={handleCancel}
                            className="flex items-center gap-2 rounded-lg bg-red-500 px-4 py-2 text-sm font-medium text-white hover:bg-red-600 transition-colors"
                        >
                            <XCircle className="h-4 w-4" /> Cancel
                        </button>
                    )}
                    {canRetry && (
                        <button
                            onClick={handleRetry}
                            disabled={isRetrying}
                            className="flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors disabled:cursor-not-allowed disabled:opacity-60"
                        >
                            <RefreshCcw className={`h-4 w-4 ${isRetrying ? 'animate-spin' : ''}`} />
                            {isRetrying ? 'Retrying...' : 'Retry'}
                        </button>
                    )}
                    {onClose && (
                        <button
                            onClick={onClose}
                            className="rounded-lg border border-border px-4 py-2 text-sm font-medium hover:bg-accent transition-colors"
                        >
                            Close
                        </button>
                    )}
                </div>
            </div>

            <div className="space-y-4">
                <div className="flex items-center justify-between text-sm font-medium">
                    <span>Overall Progress</span>
                    <span>{progressLabel}</span>
                </div>

                <div className="h-4 w-full overflow-hidden rounded-full bg-secondary/30">
                    <div
                        className="h-full bg-primary transition-all duration-500 ease-in-out"
                        style={{ width: `${progress}%` }}
                    />
                </div>

                {status === 'running' && currentQuestion && (
                    <div className="mt-6 animate-in fade-in slide-in-from-bottom-2 duration-500">
                        <h4 className="text-sm font-semibold text-muted-foreground mb-2">Current Question:</h4>
                        <div className="rounded-lg bg-accent/50 p-4 border border-border">
                            <p className="text-sm font-medium italic">"{currentQuestion}"</p>
                        </div>
                    </div>
                )}

                {caseError && showCaseError && (
                    <div className="mt-4 rounded-lg bg-yellow-500/10 p-4 border border-yellow-500/20 text-yellow-700">
                        <div className="flex items-center gap-2 font-semibold">
                            <AlertCircle className="h-4 w-4" />
                            <span>{status === 'failed' ? 'Failed Test Case' : 'Latest Test Case Warning'}</span>
                        </div>
                        <p className="mt-1 text-sm">{caseError}</p>
                    </div>
                )}

                {displayError && (
                    <div className="mt-4 rounded-lg bg-red-500/10 p-4 border border-red-500/20 text-red-600">
                        <div className="flex items-center gap-2 font-semibold">
                            <AlertCircle className="h-4 w-4" />
                            <span>Evaluation Error</span>
                        </div>
                        <p className="mt-1 text-sm">{displayError}</p>
                    </div>
                )}

                {status === 'completed' && summaryMetrics && (
                    <div className="mt-8">
                        <h4 className="text-lg font-bold mb-4">Summary results</h4>
                        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                            {[
                                { label: 'Faithfulness', value: summaryMetrics.faithfulness_avg, color: 'text-blue-500' },
                                { label: 'Relevancy', value: summaryMetrics.relevancy_avg, color: 'text-green-500' },
                                { label: 'Precision', value: summaryMetrics.precision_avg, color: 'text-purple-500' },
                                { label: 'Recall', value: summaryMetrics.recall_avg, color: 'text-orange-500' },
                                { label: 'Correctness', value: summaryMetrics.g_eval_avg, color: 'text-rose-500' }
                            ].map((m) => (
                                <div key={m.label} className="rounded-xl border border-border p-4 bg-muted/30">
                                    <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">{m.label}</p>
                                    <p className={`text-2xl font-black mt-1 ${m.color}`}>
                                        {m.value?.toFixed(2) || 'N/A'}
                                    </p>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    )
}
