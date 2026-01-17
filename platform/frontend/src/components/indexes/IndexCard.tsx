import { Link } from 'react-router-dom'
import { KnowledgeBaseIndex, api } from '../../api/client'
import { Database, Trash2, Play, HardDrive, Cpu, FileText } from 'lucide-react'
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
      alert('Failed to delete index: ' + (err as Error).message)
    } finally {
      setIsDeleting(false)
    }
  }

  return (
    <div className="bg-white border rounded-lg p-4 hover:shadow-md transition-shadow">
      <div className="flex justify-between items-start mb-2">
        <div className="flex-1 min-w-0 mr-4">
          <Link to={`/indexes/${index.id}`} className="font-medium text-lg text-blue-600 hover:underline truncate block">
            {index.name}
          </Link>
          <p className="text-sm text-gray-500 line-clamp-2">{index.description || 'No description'}</p>
        </div>
        <div className="flex space-x-2 flex-shrink-0">
          {index.status === 'ready' && onRunEvaluation && (
            <button
              onClick={(e) => {
                e.preventDefault()
                e.stopPropagation()
                onRunEvaluation()
              }}
              className="p-1 text-gray-400 hover:text-green-600"
              title="Run Evaluation"
            >
              <Play className="h-5 w-5" />
            </button>
          )}
          <button
            onClick={handleDelete}
            disabled={isDeleting}
            className="p-1 text-gray-400 hover:text-red-600 disabled:opacity-50"
            title="Delete Index"
          >
            <Trash2 className="h-5 w-5" />
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 text-sm mt-4">
        <div className="flex items-center text-gray-600 min-w-0">
          <Database className="h-4 w-4 mr-2 flex-shrink-0" />
          <span className="truncate" title={index.knowledge_base_name}>
            KB: {index.knowledge_base_name}
          </span>
        </div>
        <div className="flex items-center text-gray-600 min-w-0">
          <Cpu className="h-4 w-4 mr-2 flex-shrink-0" />
          <span className="truncate" title={index.rag_config_name}>
            RAG: {index.rag_config_name}
          </span>
        </div>
        <div className="flex items-center text-gray-600">
          <HardDrive className="h-4 w-4 mr-2 flex-shrink-0" />
          <span>{index.storage_type}</span>
        </div>
        <div className="flex items-center text-gray-600">
          <FileText className="h-4 w-4 mr-2 flex-shrink-0" />
          <span>{index.chunk_count} chunks</span>
        </div>
      </div>

      <div className="mt-4 pt-3 border-t flex justify-between items-center text-xs text-gray-500">
        <span className={`px-2 py-0.5 rounded-full ${index.status === 'ready' ? 'bg-green-100 text-green-800' :
          index.status === 'building' ? 'bg-blue-100 text-blue-800' :
            index.status === 'failed' ? 'bg-red-100 text-red-800' :
              'bg-gray-100 text-gray-800'
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
