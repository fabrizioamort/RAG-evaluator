import React, { useState } from 'react'
import { Database, Loader2 } from 'lucide-react'
import { KnowledgeBaseCreate } from '@/api/client'
import { DialogShell } from '@/components/ui/DialogShell'

interface CreateKBDialogProps {
    isOpen: boolean
    onClose: () => void
    onSubmit: (data: KnowledgeBaseCreate) => Promise<void>
}

export function CreateKBDialog({ isOpen, onClose, onSubmit }: CreateKBDialogProps) {
    const [name, setName] = useState('')
    const [description, setDescription] = useState('')
    const [isSubmitting, setIsSubmitting] = useState(false)

    if (!isOpen) return null

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        setIsSubmitting(true)
        try {
            await onSubmit({
                name,
                description: description || undefined,
            })
            setName('')
            setDescription('')
            onClose()
        } catch (error) {
            console.error('Failed to create KB:', error)
        } finally {
            setIsSubmitting(false)
        }
    }

    return (
        <DialogShell
            isOpen={isOpen}
            title="New Knowledge Base"
            icon={<Database className="h-5 w-5 text-primary" />}
            onClose={onClose}
            size="sm"
            closeDisabled={isSubmitting}
            footer={(
                <div className="flex justify-end gap-3">
                    <button
                        type="button"
                        onClick={onClose}
                        disabled={isSubmitting}
                        className="rounded-md px-4 py-2 text-sm font-medium transition-colors hover:bg-accent disabled:opacity-50"
                    >
                        Cancel
                    </button>
                    <button
                        type="submit"
                        form="create-kb-form"
                        disabled={isSubmitting || !name}
                        className="inline-flex items-center justify-center rounded-md bg-primary px-6 py-2 text-sm font-semibold text-primary-foreground transition-all hover:bg-primary/90 disabled:opacity-50"
                    >
                        {isSubmitting ? (
                            <>
                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                Creating...
                            </>
                        ) : (
                            'Create'
                        )}
                    </button>
                </div>
            )}
        >
                <form id="create-kb-form" onSubmit={handleSubmit} className="space-y-4">
                    <div className="space-y-2">
                        <label htmlFor="kb-name" className="text-sm font-medium">
                            Name
                        </label>
                        <input
                            id="kb-name"
                            required
                            value={name}
                            onChange={(e) => setName(e.target.value)}
                            placeholder="e.g. Technical Docs v1"
                            className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                        />
                    </div>

                    <div className="space-y-2">
                        <label htmlFor="kb-description" className="text-sm font-medium">
                            Description (Optional)
                        </label>
                        <textarea
                            id="kb-description"
                            value={description}
                            onChange={(e) => setDescription(e.target.value)}
                            placeholder="What kind of documents are in this KB?"
                            className="flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                        />
                    </div>
                </form>
        </DialogShell>
    )
}
