import { useState, useEffect } from 'react'
import { api, RAGConfig } from '../../api/client'
import { Loader2 } from 'lucide-react'
import { DialogShell } from '@/components/ui/DialogShell'

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
    <DialogShell
      isOpen
      title="Create New Index"
      onClose={onClose}
      closeDisabled={submitting}
      footer={!loading && (
        <div className="flex justify-end gap-3">
          <button
            type="button"
            onClick={onClose}
            disabled={submitting}
            className="rounded-lg px-4 py-2 text-sm font-semibold hover:bg-muted transition-colors disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            type="submit"
            form="create-index-form"
            disabled={submitting || !selectedConfigId}
            className="flex items-center rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          >
            {submitting && <Loader2 className="h-4 w-4 animate-spin mr-2" />}
            Create Index
          </button>
        </div>
      )}
    >
        {loading ? (
          <div className="flex justify-center py-8">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
          </div>
        ) : (
          <form id="create-index-form" onSubmit={handleSubmit} className="space-y-4">
            {error && (
              <div className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">
                {error}
              </div>
            )}

            <div>
              <label className="mb-1 block text-sm font-medium text-foreground">
                RAG Configuration
              </label>
              <select
                required
                value={selectedConfigId}
                onChange={(e) => setSelectedConfigId(e.target.value)}
                className="w-full rounded-md border border-input bg-background p-2 outline-none focus:ring-2 focus:ring-ring"
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
              <label className="mb-1 block text-sm font-medium text-foreground">
                Name (Optional)
              </label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Auto-generated if empty"
                className="w-full rounded-md border border-input bg-background p-2 outline-none focus:ring-2 focus:ring-ring"
              />
            </div>

            <div>
              <label className="mb-1 block text-sm font-medium text-foreground">
                Description (Optional)
              </label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                rows={3}
                className="w-full rounded-md border border-input bg-background p-2 outline-none focus:ring-2 focus:ring-ring"
              />
            </div>
          </form>
        )}
    </DialogShell>
  )
}
