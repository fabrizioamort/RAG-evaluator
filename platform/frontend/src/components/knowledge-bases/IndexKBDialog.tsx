import { useState, useEffect } from 'react'
import { Play, Loader2, Cpu, Database } from 'lucide-react'
import { api, RAGConfig } from '@/api/client'
import { cn } from '@/lib/utils'
import { DialogShell } from '@/components/ui/DialogShell'

interface IndexKBDialogProps {
    projectId: string
    kbName: string
    isOpen: boolean
    onClose: () => void
    onConfirm: (ragConfigId: string) => void
}

export function IndexKBDialog({ projectId, kbName, isOpen, onClose, onConfirm }: IndexKBDialogProps) {
    const [configs, setConfigs] = useState<RAGConfig[]>([])
    const [selectedId, setSelectedId] = useState<string>('')
    const [isLoading, setIsLoading] = useState(false)
    const [isStarting, setIsStarting] = useState(false)

    useEffect(() => {
        if (isOpen) {
            setIsStarting(false)
            if (projectId) {
                setIsLoading(true)
                api.ragConfigs.list(projectId)
                    .then(res => {
                        setConfigs(res.data.items)
                    })
                    .catch(err => {
                        console.error('Failed to load RAG configs:', err)
                    })
                    .finally(() => {
                        setIsLoading(false)
                    })
            }
        }
    }, [isOpen, projectId])

    const handleConfirm = () => {
        if (!selectedId) return
        setIsStarting(true)
        onConfirm(selectedId)
    }

    if (!isOpen) return null

    return (
        <DialogShell
            isOpen={isOpen}
            title="Index Knowledge Base"
            description={kbName}
            icon={(
                <span className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10 text-primary">
                    <Database className="h-5 w-5" />
                </span>
            )}
            onClose={onClose}
            closeDisabled={isStarting}
            footer={(
                <div className="flex items-center justify-end gap-3">
                    <button
                        onClick={onClose}
                        disabled={isStarting}
                        className="px-6 py-2 text-sm font-medium hover:bg-accent rounded-lg transition-colors disabled:opacity-50"
                    >
                        Cancel
                    </button>
                    <button
                        onClick={handleConfirm}
                        disabled={!selectedId || isStarting}
                        className="flex items-center gap-2 rounded-lg bg-primary px-8 py-2 text-sm font-bold text-primary-foreground hover:bg-primary/90 transition-all shadow-lg shadow-primary/20 active:scale-95 disabled:opacity-50"
                    >
                        {isStarting ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                            <Play className="h-4 w-4 fill-current" />
                        )}
                        {isStarting ? 'Starting...' : 'Start Indexing'}
                    </button>
                </div>
            )}
        >
                    <div className="mb-4 text-sm font-medium">Select RAG Configuration for Indexing</div>

                    {isLoading ? (
                        <div className="flex h-[200px] flex-col items-center justify-center gap-4">
                            <Loader2 className="h-8 w-8 animate-spin text-primary" />
                            <p className="text-sm text-muted-foreground">Loading configurations...</p>
                        </div>
                    ) : configs.length === 0 ? (
                        <div className="flex h-[200px] flex-col items-center justify-center gap-2 border-2 border-dashed border-border rounded-xl">
                            <p className="text-sm text-muted-foreground">No RAG configurations found.</p>
                            <p className="text-xs text-muted-foreground">Please create a RAG configuration first.</p>
                        </div>
                    ) : (
                        <div className="grid gap-2 max-h-[300px] overflow-y-auto pr-2 custom-scrollbar">
                            {configs.map(config => (
                                <button
                                    key={config.id}
                                    onClick={() => setSelectedId(config.id)}
                                    disabled={isStarting}
                                    className={cn(
                                        "flex items-center justify-between rounded-xl border p-4 text-left transition-all",
                                        selectedId === config.id
                                            ? "border-primary bg-primary/5 ring-1 ring-primary"
                                            : "border-border hover:border-primary/50 hover:bg-accent",
                                        isStarting && "opacity-50"
                                    )}
                                >
                                    <div className="flex items-center gap-3">
                                        <Cpu className={cn("h-4 w-4", selectedId === config.id ? "text-primary" : "text-muted-foreground")} />
                                        <div>
                                            <p className="text-sm font-bold">{config.name}</p>
                                            <p className="text-[10px] text-muted-foreground mt-0.5 capitalize">{config.rag_type.replace('_', ' ')} • {config.llm_model}</p>
                                        </div>
                                    </div>
                                    {selectedId === config.id && <div className="h-2 w-2 rounded-full bg-primary" />}
                                </button>
                            ))}
                        </div>
                    )}
        </DialogShell>
    )
}
