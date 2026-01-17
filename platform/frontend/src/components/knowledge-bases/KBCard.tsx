import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useQueryClient } from '@tanstack/react-query'
import {
    Database,
    FileText,
    Clock,
    MoreVertical,
    CheckCircle2,
    AlertCircle,
    Loader2,
    Layers
} from 'lucide-react'
import { KnowledgeBase } from '@/api/client'
import { cn } from '@/lib/utils'
import { CreateIndexDialog } from '@/components/indexes/CreateIndexDialog'

interface KBCardProps {
    kb: KnowledgeBase
}

export function KBCard({ kb }: KBCardProps) {
    const navigate = useNavigate()
    const queryClient = useQueryClient()
    const [isIndexDialogOpen, setIsIndexDialogOpen] = useState(false)

    const handleIndex = (e: React.MouseEvent) => {
        e.stopPropagation()
        setIsIndexDialogOpen(true)
    }

    const statusConfig = {
        pending: { icon: Clock, color: 'text-yellow-500', bg: 'bg-yellow-500/10' },
        ready: { icon: CheckCircle2, color: 'text-green-500', bg: 'bg-green-500/10' },
        indexing: { icon: Loader2, color: 'text-primary', bg: 'bg-primary/10', spin: true },
        error: { icon: AlertCircle, color: 'text-destructive', bg: 'bg-destructive/10' },
    }

    const activeConfig = statusConfig[kb.status] || statusConfig.pending
    const { icon: StatusIcon, color: statusColor, bg: statusBg } = activeConfig
    const spin = 'spin' in activeConfig ? activeConfig.spin : false

    return (
        <div
            className="group relative flex flex-col rounded-xl border border-border bg-card p-5 transition-all hover:border-primary/50 hover:shadow-lg"
        >
            <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                    <div className={cn("rounded-lg p-2 transition-colors", statusBg, statusColor)}>
                        <Database className="h-5 w-5" />
                    </div>
                    <div>
                        <h3 className="font-semibold text-base leading-none">{kb.name}</h3>
                        <p className="mt-1.5 text-xs text-muted-foreground line-clamp-1">
                            {kb.description || 'No description'}
                        </p>
                    </div>
                </div>
                <button className="rounded-md p-1 hover:bg-muted text-muted-foreground transition-colors">
                    <MoreVertical className="h-4 w-4" />
                </button>
            </div>

            <div className="mt-6 flex items-center justify-between text-xs">
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-1.5 text-muted-foreground">
                        <FileText className="h-3.5 w-3.5" />
                        <span className="font-medium text-foreground">{kb.document_count}</span>
                        <span>files</span>
                    </div>
                    <div className="flex items-center gap-1.5 text-muted-foreground">
                        <Clock className="h-3.5 w-3.5" />
                        <span>v{kb.current_version}</span>
                    </div>
                </div>
                <div className={cn("flex items-center gap-1.5 px-2 py-1 rounded-full font-medium whitespace-nowrap", statusBg, statusColor)}>
                    <StatusIcon className={cn("h-3 w-3", spin && "animate-spin")} />
                    <span className="capitalize">{kb.status}</span>
                </div>
            </div>

            <div className="mt-4 pt-4 border-t border-border flex gap-2">
                <button
                    onClick={() => navigate(`/knowledge-bases/${kb.id}`)}
                    className="flex-1 rounded-md bg-secondary px-3 py-1.5 text-xs font-medium hover:bg-secondary/80 transition-colors"
                >
                    View Files
                </button>
                <button
                    onClick={handleIndex}
                    className="flex-1 rounded-md bg-primary/10 text-primary px-3 py-1.5 text-xs font-medium hover:bg-primary/20 transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
                    disabled={kb.document_count === 0}
                >
                    <Layers className="h-3 w-3" />
                    Create Index
                </button>
            </div>

            {isIndexDialogOpen && (
                <CreateIndexDialog
                    projectId={kb.project_id}
                    knowledgeBaseId={kb.id}
                    onClose={() => setIsIndexDialogOpen(false)}
                    onCreated={() => {
                         // Optional: show toast
                         queryClient.invalidateQueries({ queryKey: ['indexes'] })
                    }}
                />
            )}
        </div>
    )
}