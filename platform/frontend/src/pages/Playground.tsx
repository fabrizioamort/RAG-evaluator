import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  FlaskConical,
  Loader2,
  History,
  Send,
  AlertCircle,
  Layers,
  X
} from 'lucide-react'
import { api, PlaygroundQueryResponse } from '@/api/client'
import { cn } from '@/lib/utils'
import { IndexSelector } from '@/components/playground/IndexSelector'
import { ResultCard } from '@/components/playground/ResultCard'

export function Playground() {
  const queryClient = useQueryClient()
  const [selectedIndexIds, setSelectedIndexIds] = useState<string[]>([])
  const [question, setQuestion] = useState('')
  const [topK, setTopK] = useState(5)
  const [queryResults, setQueryResults] = useState<PlaygroundQueryResponse | null>(null)
  const [showHistory, setShowHistory] = useState(false)

  // Fetch available indexes
  const { data: indexesData, isLoading: indexesLoading } = useQuery({
    queryKey: ['playground-indexes'],
    queryFn: async () => {
      const response = await api.playground.getIndexes()
      return response.data
    },
  })

  // Query mutation
  const queryMutation = useMutation({
    mutationFn: async () => {
      const response = await api.playground.executeQuery({
        question,
        index_ids: selectedIndexIds,
        top_k: topK,
      })
      return response.data
    },
    onSuccess: (data) => {
      setQueryResults(data)
      queryClient.invalidateQueries({ queryKey: ['playground-history'] })
    },
  })

  // Fetch history
  const { data: historyData } = useQuery({
    queryKey: ['playground-history'],
    queryFn: async () => {
      const response = await api.playground.getHistory({ limit: 10 })
      return response.data
    },
    enabled: showHistory,
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!question.trim() || selectedIndexIds.length === 0) return
    queryMutation.mutate()
  }

  const handleIndexToggle = (indexId: string) => {
    setSelectedIndexIds((prev) => {
      if (prev.includes(indexId)) {
        return prev.filter((id) => id !== indexId)
      }
      if (prev.length >= 4) {
        return prev // Max 4 selected
      }
      return [...prev, indexId]
    })
  }

  const handleHistoryItemClick = async (queryId: string) => {
    try {
      const response = await api.playground.getQueryDetail(queryId)
      const detail = response.data
      setQuestion(detail.question)
      setQueryResults({
        query_id: detail.id,
        question: detail.question,
        results: detail.results,
        created_at: detail.created_at,
      })
      setShowHistory(false)
    } catch (error) {
      console.error('Failed to load history item:', error)
    }
  }

  const indexes = indexesData?.indexes || []
  const selectedIndexes = indexes.filter((idx) => selectedIndexIds.includes(idx.id))

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <FlaskConical className="h-8 w-8 text-primary" />
            RAG Playground
          </h1>
          <p className="mt-2 text-muted-foreground">
            Test and compare RAG systems interactively
          </p>
        </div>
        <button
          onClick={() => setShowHistory(!showHistory)}
          className={cn(
            "flex items-center gap-2 px-4 py-2 rounded-lg border transition-colors",
            showHistory
              ? "bg-primary text-primary-foreground border-primary"
              : "bg-card border-border hover:border-primary/50"
          )}
        >
          <History className="h-4 w-4" />
          History
        </button>
      </div>

      {/* History Panel */}
      {showHistory && (
        <div className="rounded-xl border border-border bg-card p-4 animate-in slide-in-from-top-2 duration-300">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold">Recent Queries</h3>
            <button onClick={() => setShowHistory(false)} className="text-muted-foreground hover:text-foreground">
              <X className="h-4 w-4" />
            </button>
          </div>
          {historyData?.items && historyData.items.length > 0 ? (
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {historyData.items.map((item) => (
                <button
                  key={item.id}
                  onClick={() => handleHistoryItemClick(item.id)}
                  className="w-full text-left p-3 rounded-lg border border-border hover:border-primary/50 hover:bg-muted/30 transition-colors"
                >
                  <p className="text-sm font-medium truncate">{item.question}</p>
                  <div className="flex items-center gap-3 mt-1 text-xs text-muted-foreground">
                    <span>{item.index_count} indexes</span>
                    <span>{item.success_count}/{item.index_count} success</span>
                    {item.total_time_ms && <span>{item.total_time_ms.toFixed(0)}ms</span>}
                    <span>{new Date(item.created_at).toLocaleString()}</span>
                  </div>
                </button>
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground text-center py-4">No query history yet</p>
          )}
        </div>
      )}

      {/* Index Selection */}
      <div className="rounded-xl border border-border bg-card p-6">
        <h2 className="text-lg font-semibold mb-2">Select RAG Systems to Compare</h2>
        <p className="text-sm text-muted-foreground mb-4">
          Choose up to 4 indexes to query and compare side-by-side
        </p>

        {indexesLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ) : indexes.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 text-center">
            <AlertCircle className="h-10 w-10 text-muted-foreground" />
            <p className="mt-3 text-sm text-muted-foreground">No indexes available</p>
            <p className="text-xs text-muted-foreground">Build an index first to use the playground</p>
          </div>
        ) : (
          <IndexSelector
            indexes={indexes}
            selectedIds={selectedIndexIds}
            onToggle={handleIndexToggle}
            maxSelections={4}
          />
        )}
      </div>

      {/* Query Input */}
      <form onSubmit={handleSubmit} className="rounded-xl border border-border bg-card p-6">
        <div className="flex flex-col gap-4">
          <div className="flex-1">
            <label htmlFor="question" className="block text-sm font-medium mb-2">
              Your Question
            </label>
            <textarea
              id="question"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask a question about your documents..."
              className="w-full min-h-[100px] rounded-lg border border-border bg-background px-4 py-3 text-sm placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary resize-none"
            />
          </div>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <label htmlFor="topK" className="text-sm font-medium">
                  Top K:
                </label>
                <select
                  id="topK"
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  className="rounded-lg border border-border bg-background px-3 py-1.5 text-sm focus:border-primary focus:outline-none"
                >
                  {[3, 5, 7, 10, 15, 20].map((k) => (
                    <option key={k} value={k}>{k}</option>
                  ))}
                </select>
              </div>
              {selectedIndexes.length > 0 && (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Layers className="h-4 w-4" />
                  <span>{selectedIndexes.length} index{selectedIndexes.length > 1 ? 'es' : ''} selected</span>
                </div>
              )}
            </div>
            <button
              type="submit"
              disabled={!question.trim() || selectedIndexIds.length === 0 || queryMutation.isPending}
              className="flex items-center gap-2 px-6 py-2.5 rounded-lg bg-primary text-primary-foreground font-semibold hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {queryMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Querying...
                </>
              ) : (
                <>
                  <Send className="h-4 w-4" />
                  Ask
                </>
              )}
            </button>
          </div>
        </div>
      </form>

      {/* Error Display */}
      {queryMutation.isError && (
        <div className="rounded-xl border border-destructive/50 bg-destructive/10 p-4 flex items-center gap-3">
          <AlertCircle className="h-5 w-5 text-destructive" />
          <p className="text-sm text-destructive">
            {queryMutation.error instanceof Error ? queryMutation.error.message : 'Query failed'}
          </p>
        </div>
      )}

      {/* Results */}
      {queryResults && (
        <div className="space-y-4 animate-in fade-in duration-500">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">Results</h2>
            <span className="text-sm text-muted-foreground">
              {new Date(queryResults.created_at).toLocaleString()}
            </span>
          </div>

          {/* Results Grid */}
          <div className={cn(
            "grid gap-4",
            queryResults.results.length === 1 && "grid-cols-1",
            queryResults.results.length === 2 && "grid-cols-1 lg:grid-cols-2",
            queryResults.results.length >= 3 && "grid-cols-1 lg:grid-cols-2"
          )}>
            {queryResults.results.map((result) => (
              <ResultCard key={result.index_id} result={result} />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
