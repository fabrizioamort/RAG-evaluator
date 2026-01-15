import { useState, useEffect, useCallback, useRef } from 'react'
import { ProgressEvent, SummaryMetrics, api } from '../api/client'

export interface EvaluationStreamState {
    completed: number
    total: number
    status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | 'paused'
    currentQuestion?: string
    lastResult?: unknown
    error?: string
    summaryMetrics?: SummaryMetrics
}

export function useEvaluationStream(evaluationId: string | null) {
    const [state, setState] = useState<EvaluationStreamState>({
        completed: 0,
        total: 0,
        status: 'pending',
    })
    const eventSourceRef = useRef<EventSource | null>(null)

    const connect = useCallback(() => {
        if (!evaluationId) return

        // Close existing connection if any
        if (eventSourceRef.current) {
            eventSourceRef.current.close()
        }

        const url = api.evaluations.getStreamUrl(evaluationId)
        const es = new EventSource(url)
        eventSourceRef.current = es

        es.addEventListener('started', (event: MessageEvent) => {
            const data: ProgressEvent = JSON.parse(event.data)
            setState((s) => ({
                ...s,
                status: 'running',
                total: data.total_test_cases || s.total,
            }))
        })

        es.addEventListener('progress', (event: MessageEvent) => {
            const data: ProgressEvent = JSON.parse(event.data)
            setState((s) => ({
                ...s,
                status: 'running',
                completed: data.completed || s.completed,
                total: data.total || s.total,
                currentQuestion: data.current_question,
                lastResult: data.last_result,
            }))
        })

        es.addEventListener('completed', (event: MessageEvent) => {
            const data: ProgressEvent = JSON.parse(event.data)
            setState((s) => ({
                ...s,
                status: 'completed',
                summaryMetrics: data.summary_metrics,
            }))
            es.close()
        })

        es.addEventListener('error', (event: MessageEvent) => {
            // Check if it's a "real" evaluation error vs a connection error
            try {
                const data: ProgressEvent = JSON.parse(event.data)
                setState((s) => ({
                    ...s,
                    status: 'failed',
                    error: data.error_message,
                }))
            } catch {
                // SSE connection error
                console.error('SSE Connection Error')
            }
            es.close()
        })

        es.addEventListener('paused', (event: MessageEvent) => {
            const data: ProgressEvent = JSON.parse(event.data)
            setState((s) => ({
                ...s,
                status: 'paused',
                completed: data.completed || s.completed,
            }))
        })

        es.addEventListener('resumed', (event: MessageEvent) => {
            const data: ProgressEvent = JSON.parse(event.data)
            setState((s) => ({
                ...s,
                status: 'running',
                completed: data.resuming_from || s.completed,
            }))
        })

        // General connection error
        es.onerror = () => {
            console.error('EventSource failed')
            es.close()
        }
    }, [evaluationId])

    useEffect(() => {
        connect()
        return () => {
            if (eventSourceRef.current) {
                eventSourceRef.current.close()
            }
        }
    }, [connect])

    return { ...state, reconnect: connect }
}
