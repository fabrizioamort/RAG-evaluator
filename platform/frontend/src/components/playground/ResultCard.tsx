import { useState } from 'react'
import {
  CheckCircle2,
  XCircle,
  Clock,
  Coins,
  ChevronDown,
  ChevronUp,
  FileText,
  Layers,
  AlertCircle,
  Zap
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { PlaygroundQueryResult } from '@/api/client'

interface ResultCardProps {
  result: PlaygroundQueryResult
}

// RAG type badge colors
const ragTypeColors: Record<string, string> = {
  vector_semantic: 'bg-blue-500/10 text-blue-600 border-blue-500/20',
  vector_hybrid: 'bg-purple-500/10 text-purple-600 border-purple-500/20',
  graph_rag: 'bg-amber-500/10 text-amber-600 border-amber-500/20',
  filesystem_rag: 'bg-green-500/10 text-green-600 border-green-500/20',
  rlm_rag: 'bg-cyan-500/10 text-cyan-600 border-cyan-500/20',
  google_vertex_search: 'bg-yellow-500/10 text-yellow-600 border-yellow-500/20',
}

const ragTypeLabels: Record<string, string> = {
  vector_semantic: 'Vector Semantic',
  vector_hybrid: 'Hybrid Search',
  graph_rag: 'Graph RAG',
  filesystem_rag: 'Filesystem',
  rlm_rag: 'RLM-RAG',
  google_vertex_search: 'Vertex AI Search',
}

export function ResultCard({ result }: ResultCardProps) {
  const [showChunks, setShowChunks] = useState(false)
  const [showTrace, setShowTrace] = useState(false)

  const hasChunks = result.retrieved_context?.chunk_details && result.retrieved_context.chunk_details.length > 0
  const hasTrace = result.trace?.steps && result.trace.steps.length > 0

  return (
    <div className={cn(
      "rounded-xl border bg-card overflow-hidden",
      result.success ? "border-border" : "border-destructive/50"
    )}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 bg-muted/30 border-b border-border">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            {result.success ? (
              <CheckCircle2 className="h-4 w-4 text-green-500" />
            ) : (
              <XCircle className="h-4 w-4 text-destructive" />
            )}
            <h3 className="font-semibold text-sm">{result.index_name}</h3>
          </div>
          <span
            className={cn(
              "px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider border",
              ragTypeColors[result.rag_type] || 'bg-gray-500/10 text-gray-600 border-gray-500/20'
            )}
          >
            {ragTypeLabels[result.rag_type] || result.rag_type}
          </span>
        </div>

        {/* Quick metrics */}
        {result.metrics && (
          <div className="flex items-center gap-4 text-xs text-muted-foreground">
            <span className="flex items-center gap-1" title="Total time">
              <Clock className="h-3 w-3" />
              {result.metrics.total_time_ms.toFixed(0)}ms
            </span>
            <span className="flex items-center gap-1" title="Tokens used">
              <Zap className="h-3 w-3" />
              {result.metrics.total_tokens}
            </span>
            {result.metrics.cost_usd && (
              <span className="flex items-center gap-1" title="Estimated cost">
                <Coins className="h-3 w-3" />
                ${parseFloat(result.metrics.cost_usd).toFixed(4)}
              </span>
            )}
          </div>
        )}
      </div>

      {/* Content */}
      <div className="p-4 space-y-4">
        {/* Error message */}
        {!result.success && result.error && (
          <div className="flex items-start gap-3 p-3 rounded-lg bg-destructive/10 border border-destructive/20">
            <AlertCircle className="h-5 w-5 text-destructive flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-medium text-destructive">Query Failed</p>
              <p className="text-xs text-destructive/80 mt-1">{result.error}</p>
            </div>
          </div>
        )}

        {/* Answer */}
        {result.success && result.answer && (
          <div className="space-y-2">
            <h4 className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Answer</h4>
            <div className="text-sm leading-relaxed whitespace-pre-wrap bg-muted/20 rounded-lg p-4 border border-border/50">
              {result.answer}
            </div>
          </div>
        )}

        {/* Detailed metrics */}
        {result.metrics && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="rounded-lg bg-muted/30 p-3 text-center">
              <p className="text-[10px] uppercase font-bold text-muted-foreground">Retrieval</p>
              <p className="text-lg font-bold">{result.metrics.retrieval_time_ms.toFixed(0)}<span className="text-xs text-muted-foreground">ms</span></p>
            </div>
            <div className="rounded-lg bg-muted/30 p-3 text-center">
              <p className="text-[10px] uppercase font-bold text-muted-foreground">Generation</p>
              <p className="text-lg font-bold">{result.metrics.generation_time_ms.toFixed(0)}<span className="text-xs text-muted-foreground">ms</span></p>
            </div>
            <div className="rounded-lg bg-muted/30 p-3 text-center">
              <p className="text-[10px] uppercase font-bold text-muted-foreground">Prompt</p>
              <p className="text-lg font-bold">{result.metrics.prompt_tokens}</p>
            </div>
            <div className="rounded-lg bg-muted/30 p-3 text-center">
              <p className="text-[10px] uppercase font-bold text-muted-foreground">Completion</p>
              <p className="text-lg font-bold">{result.metrics.completion_tokens}</p>
            </div>
          </div>
        )}

        {/* Expandable sections */}
        <div className="space-y-2 pt-2 border-t border-border">
          {/* Retrieved Chunks */}
          {hasChunks && (
            <div>
              <button
                onClick={() => setShowChunks(!showChunks)}
                className="w-full flex items-center justify-between p-2 rounded-lg hover:bg-muted/30 transition-colors"
              >
                <span className="flex items-center gap-2 text-sm font-medium">
                  <FileText className="h-4 w-4" />
                  Retrieved Chunks ({result.retrieved_context!.chunk_details.length})
                </span>
                {showChunks ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
              </button>

              {showChunks && (
                <div className="mt-2 space-y-3 animate-in slide-in-from-top-2 duration-200">
                  {result.retrieved_context!.chunk_details.map((chunk, index) => (
                    <div key={index} className="rounded-lg border border-border overflow-hidden">
                      <div className="flex items-center justify-between px-3 py-2 bg-muted/30 border-b border-border/50">
                        <div className="flex items-center gap-2">
                          <span className="flex h-5 w-5 items-center justify-center rounded-full bg-primary/10 text-primary text-[10px] font-black">
                            #{chunk.rank}
                          </span>
                          <span className="text-[10px] font-bold text-muted-foreground uppercase truncate max-w-[200px]">
                            {chunk.source}
                          </span>
                        </div>
                        <span className="text-xs font-bold text-primary tabular-nums">
                          Score: {chunk.score.toFixed(4)}
                        </span>
                      </div>
                      <div className="p-3">
                        <p className="text-xs leading-relaxed whitespace-pre-wrap max-h-[150px] overflow-y-auto">
                          {chunk.content}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Retrieval Trace */}
          {hasTrace && (
            <div>
              <button
                onClick={() => setShowTrace(!showTrace)}
                className="w-full flex items-center justify-between p-2 rounded-lg hover:bg-muted/30 transition-colors"
              >
                <span className="flex items-center gap-2 text-sm font-medium">
                  <Layers className="h-4 w-4" />
                  Retrieval Trace ({result.trace!.steps.length} steps)
                </span>
                {showTrace ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
              </button>

              {showTrace && (
                <div className="mt-2 space-y-3 animate-in slide-in-from-top-2 duration-200">
                  <div className="flex items-center gap-2 px-2 text-xs text-muted-foreground">
                    <span className="font-medium">Strategy:</span>
                    <span className="capitalize font-bold text-foreground">{result.trace!.strategy}</span>
                    <span className="mx-2">|</span>
                    <span className="font-medium">Total:</span>
                    <span className="font-bold text-foreground">{result.trace!.total_duration_ms.toFixed(1)}ms</span>
                  </div>

                  {result.trace!.steps.map((step, index) => (
                    <div key={index} className="rounded-lg border border-border/50 p-3">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-xs font-bold capitalize">{step.step_type.replace(/_/g, ' ')}</span>
                        <span className="text-[10px] font-bold text-muted-foreground tabular-nums bg-muted px-2 py-0.5 rounded-full">
                          {step.duration_ms.toFixed(1)}ms
                        </span>
                      </div>
                      {step.input_data !== null && step.input_data !== undefined && (
                        <div className="text-[10px] text-muted-foreground">
                          <span className="font-bold uppercase">Input: </span>
                          {typeof step.input_data === 'string'
                            ? step.input_data
                            : JSON.stringify(step.input_data).slice(0, 100) + '...'
                          }
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
