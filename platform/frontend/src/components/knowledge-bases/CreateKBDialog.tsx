import React, { useState } from 'react'
import { X, Database, Loader2 } from 'lucide-react'
import { KnowledgeBaseCreate } from '@/api/client'

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
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <div
                className="absolute inset-0 bg-background/80 backdrop-blur-sm animate-in fade-in duration-200"
                onClick={onClose}
            />
            <div className="relative w-full max-w-md rounded-xl border border-border bg-card shadow-2xl animate-in zoom-in-95 duration-200">
                <div className="flex items-center justify-between border-b border-border p-6">
                    <div className="flex items-center gap-2">
                        <Database className="h-5 w-5 text-primary" />
                        <h2 className="text-xl font-semibold">New Knowledge Base</h2>
                    </div>
                    <button
                        onClick={onClose}
                        className="rounded-md p-1 hover:bg-muted text-muted-foreground transition-colors"
                    >
                        <X className="h-5 w-5" />
                    </button>
                </div>

                <form onSubmit={handleSubmit} className="p-6 space-y-4">
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

                    <div className="flex justify-end gap-3 pt-4 border-t border-border mt-4">
                        <button
                            type="button"
                            onClick={onClose}
                            className="px-4 py-2 text-sm font-medium hover:bg-accent rounded-md transition-colors"
                        >
                            Cancel
                        </button>
                        <button
                            type="submit"
                            disabled={isSubmitting || !name}
                            className="inline-flex items-center justify-center rounded-md bg-primary px-6 py-2 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-all disabled:opacity-50"
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
                </form>
            </div>
        </div>
    )
}
