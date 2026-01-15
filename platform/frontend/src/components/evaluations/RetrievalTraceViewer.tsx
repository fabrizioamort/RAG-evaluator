import { useQuery } from '@tanstack/react-query'
import {
    Loader2,
    Search,
    Clock,
    FileText,
    Layers,
    AlertCircle
} from 'lucide-react'
import { api, RetrievalTrace, RetrievalTraceStep, RetrievalTraceChunk } from '../../api/client'
import { cn } from '@/lib/utils'

interface RetrievalTraceViewerProps {
    evaluationId: string
    resultId: string
}

export function RetrievalTraceViewer({ evaluationId, resultId }: RetrievalTraceViewerProps) {
    const { data: trace, isLoading, error } = useQuery<RetrievalTrace>({
        queryKey: ['evaluation-trace', evaluationId, resultId],
        queryFn: async () => {
            const response = await api.evaluations.getTrace(evaluationId, resultId)
            return response.data
        },
    })

    if (isLoading) {
        return (
            <div className="flex flex-col items-center justify-center py-12 space-y-4">
                <Loader2 className="h-8 w-8 animate-spin text-primary/50" />
                <p className="text-sm text-muted-foreground font-medium text-center">Loading retrieval trace...</p>
            </div>
        )
    }

    if (error || !trace) {
        return (
            <div className="flex flex-col items-center justify-center py-12 space-y-4 text-center">
                <AlertCircle className="h-10 w-10 text-destructive/50" />
                <div className="space-y-1">
                    <p className="font-semibold text-destructive">Failed to load trace</p>
                    <p className="text-sm text-muted-foreground px-10">
                        The retrieval trace for this result may have been deleted or is unavailable.
                    </p>
                </div>
            </div>
        )
    }

    return (
        <div className="space-y-8 animate-in fade-in duration-500">
            {/* Strategy Header */}
            <div className="flex items-center justify-between border-b border-border pb-4">
                <div className="flex items-center gap-3">
                    <div className="rounded-lg bg-primary/10 p-2 text-primary">
                        <Layers className="h-5 w-5" />
                    </div>
                    <div>
                        <h3 className="font-bold text-lg">Retrieval Strategy: <span className="capitalize text-primary">{trace.strategy}</span></h3>
                        <p className="text-sm text-muted-foreground">Detailed execution trace of the retrieval process.</p>
                    </div>
                </div>
                <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-muted border border-border">
                    <Clock className="h-4 w-4 text-muted-foreground" />
                    <span className="text-sm font-bold truncate">Total time: {trace.total_duration_ms?.toFixed(1) || 0}ms</span>
                </div>
            </div>

            {/* Execution Steps */}
            <div className="space-y-6">
                <h4 className="text-sm font-bold uppercase tracking-wider text-muted-foreground flex items-center gap-2">
                    <Search className="h-4 w-4" />
                    Execution Steps
                </h4>
                <div className="relative space-y-8 before:absolute before:inset-0 before:ml-5 before:-translate-x-px before:h-full before:w-0.5 before:bg-gradient-to-b before:from-transparent before:via-border/50 before:to-transparent">
                    {trace.steps?.map((step: RetrievalTraceStep, index: number) => (
                        <div key={index} className="relative flex items-start group">
                            <div className="absolute left-0 mt-1.5 flex h-10 w-10 items-center justify-center rounded-full border border-border bg-card shadow-sm group-hover:border-primary/50 transition-colors z-10">
                                <div className="h-2.5 w-2.5 rounded-full bg-primary/50 group-hover:bg-primary transition-colors" />
                            </div>
                            <div className="ml-14 w-full space-y-2">
                                <div className="flex items-center justify-between">
                                    <h5 className="font-bold text-sm tracking-tight capitalize">{step.type?.replace(/_/g, ' ')}</h5>
                                    {step.duration_ms !== undefined && (
                                        <span className="text-[10px] font-black text-muted-foreground tabular-nums bg-muted px-2 py-0.5 rounded-full border border-border">
                                            {step.duration_ms?.toFixed(1)}ms
                                        </span>
                                    )}
                                </div>
                                <div className="rounded-xl border border-border/50 bg-muted/20 p-4 text-xs font-medium space-y-3">
                                    <div className="space-y-1">
                                        <p className="text-[10px] uppercase font-bold text-muted-foreground">Input</p>
                                        <div className="text-foreground leading-relaxed break-words">
                                            {typeof step.input === 'string' ? step.input : <pre className="whitespace-pre-wrap">{JSON.stringify(step.input, null, 2)}</pre>}
                                        </div>
                                    </div>
                                    {step.metadata && Object.keys(step.metadata).length > 0 && (
                                        <div className="pt-2 border-t border-border/30 grid grid-cols-1 gap-y-3">
                                            {Object.entries(step.metadata).map(([key, value]) => {
                                                const isObject = value !== null && typeof value === 'object';

                                                return (
                                                    <div key={key} className="space-y-1 min-w-0">
                                                        <p className="text-[10px] uppercase font-bold text-muted-foreground">{key}</p>
                                                        <div className={cn(
                                                            "text-foreground",
                                                            isObject ? "bg-background/50 rounded-lg p-2 border border-border/50 font-mono text-[10px] whitespace-pre-wrap" : "font-mono truncate"
                                                        )}>
                                                            {isObject ? (
                                                                JSON.stringify(value, null, 2)
                                                            ) : (
                                                                String(value)
                                                            )}
                                                        </div>
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Retrieved Chunks */}
            <div className="space-y-6 pt-4 border-t border-border">
                <h4 className="text-sm font-bold uppercase tracking-wider text-muted-foreground flex items-center gap-2">
                    <FileText className="h-4 w-4" />
                    Retrieved Chunks ({trace.retrieved_chunks?.length || 0})
                </h4>
                <div className="grid gap-4">
                    {trace.retrieved_chunks?.map((chunk: RetrievalTraceChunk, index: number) => (
                        <div key={index} className="group rounded-xl border border-border bg-card hover:border-primary/30 hover:shadow-sm transition-all overflow-hidden">
                            <div className="flex items-center justify-between bg-muted/30 px-4 py-2 border-b border-border/50">
                                <div className="flex items-center gap-2">
                                    <span className="flex h-6 w-6 items-center justify-center rounded-full bg-primary/10 text-primary text-[10px] font-black italic">
                                        #{chunk.rank || index + 1}
                                    </span>
                                    <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-tight truncate max-w-[200px]">
                                        Source: {chunk.source || 'Unknown'}
                                    </span>
                                </div>
                                <div className="flex items-center gap-3">
                                    <div className="text-right">
                                        <span className="text-[10px] font-bold text-muted-foreground uppercase mr-1.5">Score</span>
                                        <span className="text-xs font-black text-primary tabular-nums">
                                            {chunk.score?.toFixed(4) || '0.000'}
                                        </span>
                                    </div>
                                </div>
                            </div>
                            <div className="p-4 space-y-3">
                                <p className="text-xs leading-relaxed text-foreground whitespace-pre-wrap font-medium h-fit max-h-[150px] overflow-y-auto scrollbar-thin">
                                    {chunk.content}
                                </p>
                                {chunk.metadata && Object.keys(chunk.metadata).length > 0 && (
                                    <div className="flex flex-wrap gap-2 pt-2 border-t border-border/50">
                                        {Object.entries(chunk.metadata).map(([key, value]) => (
                                            <span key={key} className="inline-flex items-center px-2 py-0.5 rounded-full bg-muted border border-border text-[9px] font-bold text-muted-foreground">
                                                {key}: {String(value)}
                                            </span>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    )
}
