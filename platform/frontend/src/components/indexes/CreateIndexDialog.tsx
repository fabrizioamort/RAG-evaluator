import { useState, useEffect } from 'react'
import { api, RAGConfig } from '../../api/client'
import { X, Loader2 } from 'lucide-react'

interface CreateIndexDialogProps {
  knowledgeBaseId: string
  projectId: string
  onClose: () => void
  onCreated: () => void
}

export function CreateIndexDialog({ knowledgeBaseId, projectId, onClose, onCreated }: CreateIndexDialogProps) {
  const [ragConfigs, setRagConfigs] = useState<RAGConfig[]>([])
  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [selectedConfigId, setSelectedConfigId] = useState('')
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const loadConfigs = async () => {
      try {
        const response = await api.ragConfigs.list(projectId)
        setRagConfigs(response.data.items)
      } catch (err) {
        setError('Failed to load RAG configs')
      } finally {
        setLoading(false)
      }
    }
    loadConfigs()
  }, [projectId])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!selectedConfigId) return

    setSubmitting(true)
    setError(null)
    try {
      await api.indexes.create(knowledgeBaseId, {
        rag_config_id: selectedConfigId,
        name: name || undefined,
        description: description || undefined,
      })
      onCreated()
      onClose()
    } catch (err) {
      setError('Failed to create index: ' + (err as Error).message)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-lg max-w-md w-full p-6 relative">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-gray-400 hover:text-gray-600"
        >
          <X className="h-5 w-5" />
        </button>

        <h2 className="text-xl font-semibold mb-4">Create New Index</h2>

        {loading ? (
          <div className="flex justify-center py-8">
            <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-4">
            {error && (
              <div className="bg-red-50 text-red-700 p-3 rounded-md text-sm">
                {error}
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                RAG Configuration
              </label>
              <select
                required
                value={selectedConfigId}
                onChange={(e) => setSelectedConfigId(e.target.value)}
                className="w-full border rounded-md p-2 focus:ring-2 focus:ring-blue-500 outline-none"
              >
                <option value="">Select a configuration...</option>
                {ragConfigs.map(config => (
                  <option key={config.id} value={config.id}>
                    {config.name} ({config.rag_type})
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Name (Optional)
              </label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Auto-generated if empty"
                className="w-full border rounded-md p-2 focus:ring-2 focus:ring-blue-500 outline-none"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Description (Optional)
              </label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                rows={3}
                className="w-full border rounded-md p-2 focus:ring-2 focus:ring-blue-500 outline-none"
              />
            </div>

            <div className="flex justify-end space-x-3 mt-6">
              <button
                type="button"
                onClick={onClose}
                className="px-4 py-2 text-gray-700 hover:bg-gray-100 rounded-md"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={submitting || !selectedConfigId}
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 flex items-center"
              >
                {submitting && <Loader2 className="h-4 w-4 animate-spin mr-2" />}
                Create Index
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  )
}
