import React, { useEffect, useRef, useState } from 'react'
import { X, Tag as TagIcon, Loader2 } from 'lucide-react'
import { Project, ProjectUpdate } from '@/api/client'
import { DialogShell } from '@/components/ui/DialogShell'

interface EditProjectDialogProps {
    isOpen: boolean
    project: Project
    onClose: () => void
    onSubmit: (data: ProjectUpdate) => Promise<void>
}

export function EditProjectDialog({ isOpen, project, onClose, onSubmit }: EditProjectDialogProps) {
    const [name, setName] = useState(project.name)
    const [description, setDescription] = useState(project.description || '')
    const [tagInput, setTagInput] = useState('')
    const [tags, setTags] = useState<string[]>(project.tags || [])
    const [isSubmitting, setIsSubmitting] = useState(false)
    const wasOpen = useRef(false)

    useEffect(() => {
        if (isOpen && !wasOpen.current) {
            setName(project.name)
            setDescription(project.description || '')
            setTags(project.tags || [])
            setTagInput('')
        }
        wasOpen.current = isOpen
    }, [isOpen, project.name, project.description, project.tags])

    if (!isOpen) return null

    const handleAddTag = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && tagInput.trim()) {
            e.preventDefault()
            if (!tags.includes(tagInput.trim())) {
                setTags([...tags, tagInput.trim()])
            }
            setTagInput('')
        }
    }

    const removeTag = (tagToRemove: string) => {
        setTags(tags.filter((tag) => tag !== tagToRemove))
    }

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        const trimmedName = name.trim()
        if (!trimmedName) return
        setIsSubmitting(true)
        try {
            await onSubmit({
                name: trimmedName,
                description: description.trim(),
                tags,
            })
            onClose()
        } catch (error) {
            console.error('Failed to update project:', error)
        } finally {
            setIsSubmitting(false)
        }
    }

    return (
        <DialogShell
            isOpen={isOpen}
            title="Edit Project"
            onClose={onClose}
            closeDisabled={isSubmitting}
            footer={(
                <div className="flex justify-end gap-3">
                    <button
                        type="button"
                        onClick={onClose}
                        disabled={isSubmitting}
                        className="inline-flex items-center justify-center rounded-md px-4 py-2 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground disabled:opacity-50"
                    >
                        Cancel
                    </button>
                    <button
                        type="submit"
                        form="edit-project-form"
                        disabled={isSubmitting || !name.trim()}
                        className="inline-flex items-center justify-center rounded-md bg-primary px-6 py-2 text-sm font-semibold text-primary-foreground shadow-md transition-all hover:bg-primary/90 active:scale-95 disabled:pointer-events-none disabled:opacity-50"
                    >
                        {isSubmitting ? (
                            <>
                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                Saving...
                            </>
                        ) : (
                            'Save Changes'
                        )}
                    </button>
                </div>
            )}
        >
                <form id="edit-project-form" onSubmit={handleSubmit} className="space-y-6">
                    <div className="space-y-2">
                        <label htmlFor="edit-project-name" className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                            Project Name
                        </label>
                        <input
                            id="edit-project-name"
                            required
                            value={name}
                            onChange={(e) => setName(e.target.value)}
                            placeholder="e.g. Documentation Assistant"
                            className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                        />
                    </div>

                    <div className="space-y-2">
                        <label htmlFor="edit-project-description" className="text-sm font-medium">
                            Description (Optional)
                        </label>
                        <textarea
                            id="edit-project-description"
                            value={description}
                            onChange={(e) => setDescription(e.target.value)}
                            placeholder="Briefly describe the goal of this RAG evaluation project..."
                            className="flex min-h-[100px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                        />
                    </div>

                    <div className="space-y-3">
                        <label className="text-sm font-medium">Tags</label>
                        <div className="flex flex-wrap gap-2 mb-2">
                            {tags.map((tag) => (
                                <span
                                    key={tag}
                                    className="flex items-center gap-1 rounded-md bg-primary/10 px-2 py-1 text-xs font-medium text-primary"
                                >
                                    <TagIcon className="h-3 w-3" />
                                    {tag}
                                    <button type="button" onClick={() => removeTag(tag)}>
                                        <X className="h-3 w-3 hover:text-red-500" />
                                    </button>
                                </span>
                            ))}
                        </div>
                        <div className="relative">
                            <input
                                value={tagInput}
                                onChange={(e) => setTagInput(e.target.value)}
                                onKeyDown={handleAddTag}
                                placeholder="Type a tag and press Enter..."
                                className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                            />
                        </div>
                    </div>
                </form>
        </DialogShell>
    )
}
