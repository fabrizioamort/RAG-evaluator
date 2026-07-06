import { Link } from 'react-router-dom'
import { KnowledgeBaseIndex, api } from '../../api/client'
import { Database, Trash2, Play, HardDrive, Cpu, FileText, FolderKanban } from 'lucide-react'
import { useState } from 'react'

interface IndexCardProps {
  index: KnowledgeBaseIndex
  onDelete?: () => void
  onRunEvaluation?: () => void
}

function timeAgo(dateString: string) {
  const date = new Date(dateString)
  const now = new Date()
  const seconds = Math.floor((now.getTime() - date.getTime()) / 1000)

  if (seconds < 60) return 'just now'
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  return `${days}d ago`
}

export function IndexCard({ index, onDelete, onRunEvaluation }: IndexCardProps) {
  const [isDeleting, setIsDeleting] = useState(false)

  const handleDelete = async (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (!confirm('Are you sure you want to delete this index? This cannot be undone.')) return

    setIsDeleting(true)
    try {
      await api.indexes.delete(index.id)
      onDelete?.()
    } catch (err) {
      const apiError = err as { response?: { data?: { detail?: unknown } }; message?: string }
      const detail =
        typeof apiError.response?.data?.detail === 'string'
          ? apiError.response.data.detail
          : apiError.message || 'Unknown error'
      alert('Failed to delete index: ' + detail)
    } finally {
      setIsDeleting(false)
    }
  }

  return (
    <div className="rounded-lg border border-border bg-card p-4 transition-shadow hover:shadow-md">
      <div className="flex justify-between items-start mb-2">
        <div className="flex-1 min-w-0 mr-4">
          <Link to={`/indexes/${index.id}`} className="block truncate text-lg font-medium text-primary hover:underline">
            {index.name}
          </Link>
          <p className="line-clamp-2 text-sm text-muted-foreground">{index.description || 'No description'}</p>
        </div>
        <div className="flex space-x-2 flex-shrink-0">
          {index.status === 'ready' && onRunEvaluation && (
            <button
              onClick={(e) => {
                e.preventDefault()
                e.stopPropagation()
                onRunEvaluation()
              }}
              className="p-1 text-muted-foreground hover:text-green-600"
              title="Run Evaluation"
            >
              <Play className="h-5 w-5" />
            </button>
          )}
          <button
            onClick={handleDelete}
            disabled={isDeleting}
            className="p-1 text-muted-foreground hover:text-destructive disabled:opacity-50"
            title="Delete Index"
          >
            <Trash2 className="h-5 w-5" />
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 text-sm mt-4">
        {index.project_name && index.project_id && (
          <Link
            to={`/projects/${index.project_id}`}
            className="col-span-2 flex min-w-0 items-center text-muted-foreground hover:text-primary"
          >
            <FolderKanban className="h-4 w-4 mr-2 flex-shrink-0" />
            <span className="truncate" title={index.project_name}>
              Project: {index.project_name}
            </span>
          </Link>
        )}
        <div className="flex items-center text-muted-foreground min-w-0">
          <Database className="h-4 w-4 mr-2 flex-shrink-0" />
          <span className="truncate" title={index.knowledge_base_name}>
            KB: {index.knowledge_base_name}
          </span>
        </div>
        <div className="flex items-center text-muted-foreground min-w-0">
          <Cpu className="h-4 w-4 mr-2 flex-shrink-0" />
          <span className="truncate" title={index.rag_config_name}>
            RAG: {index.rag_config_name}
          </span>
        </div>
        <div className="flex items-center text-muted-foreground">
          <HardDrive className="h-4 w-4 mr-2 flex-shrink-0" />
          <span>{index.storage_type}</span>
        </div>
        <div className="flex items-center text-muted-foreground">
          <FileText className="h-4 w-4 mr-2 flex-shrink-0" />
          <span>{index.chunk_count} chunks</span>
        </div>
      </div>

      <div className="mt-4 pt-3 border-t border-border flex justify-between items-center text-xs text-muted-foreground">
        <span className={`px-2 py-0.5 rounded-full ${index.status === 'ready' ? 'bg-green-500/10 text-green-700' :
          index.status === 'building' ? 'bg-blue-500/10 text-blue-700' :
            index.status === 'failed' ? 'bg-red-500/10 text-red-700' :
              'bg-muted text-muted-foreground'
          }`}>
          {index.status.charAt(0).toUpperCase() + index.status.slice(1)}
        </span>
        <span>
          Created {timeAgo(index.created_at)}
        </span>
      </div>
    </div>
  )
}
