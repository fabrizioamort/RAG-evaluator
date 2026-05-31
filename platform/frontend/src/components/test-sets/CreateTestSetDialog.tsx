import { useState } from 'react'
import { Loader2 } from 'lucide-react'
import { TestSetCreate } from '@/api/client'
import { DialogShell } from '@/components/ui/DialogShell'

interface CreateTestSetDialogProps {
    isOpen: boolean
    onClose: () => void
    onSubmit: (data: TestSetCreate) => Promise<void>
}

export function CreateTestSetDialog({ isOpen, onClose, onSubmit }: CreateTestSetDialogProps) {
    const [name, setName] = useState('')
    const [description, setDescription] = useState('')
    const [isSubmitting, setIsSubmitting] = useState(false)

    if (!isOpen) return null

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        if (!name.trim()) return

        setIsSubmitting(true)
        try {
            await onSubmit({
                name: name.trim(),
                description: description.trim() || undefined,
            })
            setName('')
            setDescription('')
            onClose()
        } catch (error) {
            console.error('Failed to create test set:', error)
        } finally {
            setIsSubmitting(false)
        }
    }

    return (
        <DialogShell
            isOpen={isOpen}
            title="Create Test Set"
            description="Define a new collection of test cases for evaluation."
            onClose={onClose}
            closeDisabled={isSubmitting}
            footer={(
                <div className="flex justify-end gap-3">
                    <button
                        type="button"
                        onClick={onClose}
                        disabled={isSubmitting}
                        className="rounded-lg px-6 py-2.5 text-sm font-semibold hover:bg-muted transition-colors disabled:opacity-50"
                    >
                        Cancel
                    </button>
                    <button
                        type="submit"
                        form="create-test-set-form"
                        disabled={isSubmitting || !name.trim()}
                        className="flex items-center gap-2 rounded-lg bg-primary px-8 py-2.5 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-all shadow-md active:scale-95 disabled:pointer-events-none disabled:opacity-50"
                    >
                        {isSubmitting ? (
                            <>
                                <Loader2 className="h-4 w-4 animate-spin" />
                                Creating...
                            </>
                        ) : (
                            'Create'
                        )}
                    </button>
                </div>
            )}
        >
                <form id="create-test-set-form" onSubmit={handleSubmit} className="space-y-6">
                    <div className="space-y-2">
                        <label htmlFor="name" className="text-sm font-semibold text-foreground">
                            Name
                        </label>
                        <input
                            id="name"
                            type="text"
                            value={name}
                            onChange={(e) => setName(e.target.value)}
                            placeholder="e.g., Factual Retrieval Set"
                            className="w-full rounded-lg border border-input bg-background px-4 py-3 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all"
                            required
                            autoFocus
                        />
                    </div>

                    <div className="space-y-2">
                        <label htmlFor="description" className="text-sm font-semibold text-foreground">
                            Description
                        </label>
                        <textarea
                            id="description"
                            value={description}
                            onChange={(e) => setDescription(e.target.value)}
                            placeholder="Briefly describe the purpose of this test set..."
                            className="w-full min-h-[120px] rounded-lg border border-input bg-background px-4 py-3 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all resize-none"
                        />
                    </div>
                </form>
        </DialogShell>
    )
}
