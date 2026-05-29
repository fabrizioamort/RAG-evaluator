import { CheckCircle2, Database, FileText, Layers } from 'lucide-react'
import { cn } from '@/lib/utils'
import { PlaygroundIndexInfo } from '@/api/client'

interface IndexSelectorProps {
  indexes: PlaygroundIndexInfo[]
  selectedIds: string[]
  onToggle: (indexId: string) => void
  maxSelections: number
}

// RAG type badge colors
const ragTypeColors: Record<string, string> = {
  vector_semantic: 'bg-blue-500/10 text-blue-600 border-blue-500/20',
  vector_hybrid: 'bg-purple-500/10 text-purple-600 border-purple-500/20',
  graph_rag: 'bg-amber-500/10 text-amber-600 border-amber-500/20',
  filesystem_rag: 'bg-green-500/10 text-green-600 border-green-500/20',
  rlm_rag: 'bg-cyan-500/10 text-cyan-600 border-cyan-500/20',
}

const ragTypeLabels: Record<string, string> = {
  vector_semantic: 'Vector Semantic',
  vector_hybrid: 'Hybrid Search',
  graph_rag: 'Graph RAG',
  filesystem_rag: 'Filesystem',
  rlm_rag: 'RLM-RAG',
}

export function IndexSelector({ indexes, selectedIds, onToggle, maxSelections }: IndexSelectorProps) {
  // Group indexes by knowledge base
  const groupedIndexes = indexes.reduce((acc, idx) => {
    const kbName = idx.knowledge_base_name
    if (!acc[kbName]) {
      acc[kbName] = []
    }
    acc[kbName].push(idx)
    return acc
  }, {} as Record<string, PlaygroundIndexInfo[]>)

  const canSelectMore = selectedIds.length < maxSelections

  return (
    <div className="space-y-4">
      {Object.entries(groupedIndexes).map(([kbName, kbIndexes]) => (
        <div key={kbName} className="space-y-2">
          <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
            <Database className="h-4 w-4" />
            {kbName}
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
            {kbIndexes.map((index) => {
              const isSelected = selectedIds.includes(index.id)
              const isDisabled = !isSelected && !canSelectMore

              return (
                <button
                  key={index.id}
                  onClick={() => !isDisabled && onToggle(index.id)}
                  disabled={isDisabled}
                  className={cn(
                    "relative flex flex-col p-4 rounded-xl border-2 text-left transition-all",
                    isSelected
                      ? "border-primary bg-primary/5 shadow-sm"
                      : isDisabled
                      ? "border-border bg-muted/30 opacity-50 cursor-not-allowed"
                      : "border-border hover:border-primary/50 hover:shadow-sm"
                  )}
                >
                  {/* Selection indicator */}
                  {isSelected && (
                    <div className="absolute top-2 right-2">
                      <CheckCircle2 className="h-5 w-5 text-primary" />
                    </div>
                  )}

                  {/* Index name */}
                  <h4 className="font-semibold text-sm truncate pr-6">{index.name}</h4>

                  {/* RAG type badge */}
                  <span
                    className={cn(
                      "mt-2 inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider border w-fit",
                      ragTypeColors[index.rag_type] || 'bg-gray-500/10 text-gray-600 border-gray-500/20'
                    )}
                  >
                    {ragTypeLabels[index.rag_type] || index.rag_type}
                  </span>

                  {/* Stats */}
                  <div className="flex items-center gap-3 mt-3 text-xs text-muted-foreground">
                    <span className="flex items-center gap-1">
                      <FileText className="h-3 w-3" />
                      {index.document_count} docs
                    </span>
                    <span className="flex items-center gap-1">
                      <Layers className="h-3 w-3" />
                      {index.chunk_count} chunks
                    </span>
                  </div>
                </button>
              )
            })}
          </div>
        </div>
      ))}

      {/* Selection count */}
      <div className="text-xs text-muted-foreground pt-2 border-t border-border">
        {selectedIds.length} of {maxSelections} indexes selected
        {!canSelectMore && selectedIds.length > 0 && (
          <span className="ml-2 text-amber-600">(maximum reached)</span>
        )}
      </div>
    </div>
  )
}
