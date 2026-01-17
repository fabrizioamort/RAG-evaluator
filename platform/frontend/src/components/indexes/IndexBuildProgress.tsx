import { useEffect, useState } from 'react'
import { api } from '../../api/client'
import { CheckCircle2, AlertCircle, Loader2 } from 'lucide-react'
import { clsx } from 'clsx'

interface IndexBuildProgressProps {
    indexId: string
    onComplete?: () => void
    onError?: (error: string) => void
}

interface BuildEvent {
    event_type: 'building' | 'progress' | 'complete' | 'failed'
    current?: number
    total?: number
    chunk_count?: number
    message?: string
    error?: string
    document?: string
}

export function IndexBuildProgress({ indexId, onComplete, onError }: IndexBuildProgressProps) {
    const [status, setStatus] = useState<BuildEvent['event_type']>('building')
    const [progress, setProgress] = useState(0)
    const [message, setMessage] = useState('Initializing build...')
    const [error, setError] = useState<string | null>(null)

    useEffect(() => {
        const streamUrl = api.indexes.getStreamUrl(indexId)
        const eventSource = new EventSource(streamUrl)

        eventSource.onmessage = () => {
            try {
                // We handle specific events below
                // Just keeping onmessage for fallback or debugging if needed
            } catch (e) {
                console.error('Failed to parse SSE event', e)
            }
        }

        // We need to listen to specific events if they are named
        const handleEvent = (type: BuildEvent['event_type'], data: BuildEvent) => {
            if (type === 'progress') {
                setStatus('progress')
                if (data.current && data.total) {
                    setProgress(Math.round((data.current / data.total) * 100))
                    setMessage(data.document ? `Processing: ${data.document}` : `Processing ${data.current}/${data.total}`)
                }
            } else if (type === 'complete') {
                setStatus('complete')
                setProgress(100)
                setMessage(`Build complete! Created ${data.chunk_count} chunks.`)
                eventSource.close()
                onComplete?.()
            } else if (type === 'failed') {
                setStatus('failed')
                setError(data.error || 'Build failed')
                eventSource.close()
                onError?.(data.error || 'Build failed')
            } else if (type === 'building') {
                setStatus('building')
                setMessage(data.message || 'Build started...')
                if (data.total) {
                    setMessage(`Starting build for ${data.total} documents...`)
                }
            }
        }

        eventSource.addEventListener('building', (e) => handleEvent('building', JSON.parse(e.data)))
        eventSource.addEventListener('progress', (e) => handleEvent('progress', JSON.parse(e.data)))
        eventSource.addEventListener('complete', (e) => handleEvent('complete', JSON.parse(e.data)))
        eventSource.addEventListener('failed', (e) => handleEvent('failed', JSON.parse(e.data)))

        // Also catch generic messages just in case
        eventSource.onmessage = (e) => {
            try {
                const payload = JSON.parse(e.data)
                // If the payload has event_type or status, use it
                if (payload.event_type) {
                    handleEvent(payload.event_type, payload)
                }
            } catch (err) {
                // ignore
            }
        }

        eventSource.onerror = (e) => {
            console.error('SSE error', e)
            if (eventSource.readyState === EventSource.CLOSED) {
                // Connection closed
            }
        }

        return () => {
            eventSource.close()
        }
    }, [indexId, onComplete, onError])

    return (
        <div className="w-full space-y-2 p-4 border rounded-lg bg-gray-50">
            <div className="flex items-center justify-between">
                <h4 className="font-medium text-sm text-gray-900">Build Progress</h4>
                {status === 'complete' ? (
                    <CheckCircle2 className="h-5 w-5 text-green-500" />
                ) : status === 'failed' ? (
                    <AlertCircle className="h-5 w-5 text-red-500" />
                ) : (
                    <Loader2 className="h-5 w-5 text-blue-500 animate-spin" />
                )}
            </div>

            <div className="w-full bg-gray-200 rounded-full h-2.5">
                <div
                    className={clsx(
                        "h-2.5 rounded-full transition-all duration-300",
                        status === 'failed' ? "bg-red-500" :
                            status === 'complete' ? "bg-green-500" : "bg-blue-600"
                    )}
                    style={{ width: `${progress}%` }}
                ></div>
            </div>

            <div className="flex justify-between text-xs text-gray-500">
                <span className="truncate max-w-[80%]">{message}</span>
                <span>{progress}%</span>
            </div>

            {error && (
                <div className="text-xs text-red-600 mt-2">
                    Error: {error}
                </div>
            )}
        </div>
    )
}
