import { Settings2, Plus, Edit2, Trash2, Cpu, Database, Network, FolderTree, Cloud } from 'lucide-react'
import { RAGConfig } from '@/api/client'

interface RAGConfigListProps {
    configs: RAGConfig[]
    onCreateClick: () => void
    onEdit: (config: RAGConfig) => void
    onDelete: (id: string) => void
}

export function RAGConfigList({ configs, onCreateClick, onEdit, onDelete }: RAGConfigListProps) {
    if (configs.length === 0) {
        return (
            <div className="flex min-h-[400px] flex-col items-center justify-center rounded-xl border border-dashed border-border bg-card/50 p-8 text-center transition-all hover:bg-card/80">
                <div className="flex h-16 w-16 items-center justify-center rounded-full bg-primary/10 shadow-inner">
                    <Settings2 className="h-8 w-8 text-primary" />
                </div>
                <h3 className="mt-4 text-xl font-semibold tracking-tight">No RAG configurations yet</h3>
                <p className="mt-2 max-w-sm text-sm text-muted-foreground leading-relaxed">
                    Configure your RAG systems with different models, providers, and parameters.
                </p>
                <button
                    onClick={onCreateClick}
                    className="mt-6 flex items-center gap-2 rounded-lg bg-primary px-6 py-2.5 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-all shadow-md active:scale-95"
                >
                    <Plus className="h-4 w-4" />
                    Create RAG Config
                </button>
            </div>
        )
    }

    const getTypeIcon = (type: string) => {
        switch (type) {
            case 'vector_semantic': return <Database className="h-4 w-4" />
            case 'hybrid': return <Network className="h-4 w-4" />
            case 'graph': return <Network className="h-4 w-4" />
            case 'filesystem': return <FolderTree className="h-4 w-4" />
            case 'google_vertex_search': return <Cloud className="h-4 w-4" />
            default: return <Settings2 className="h-4 w-4" />
        }
    }

    return (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {configs.map((config) => (
                <div
                    key={config.id}
                    className="group relative overflow-hidden rounded-xl border border-border bg-card p-6 shadow-sm transition-all hover:-translate-y-1 hover:shadow-md"
                >
                    <div className="space-y-4">
                        <div className="flex items-start justify-between">
                            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 transition-colors group-hover:bg-primary/20">
                                <Cpu className="h-5 w-5 text-primary" />
                            </div>
                            <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                                <button
                                    onClick={() => onEdit(config)}
                                    className="rounded-md p-1.5 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
                                >
                                    <Edit2 className="h-4 w-4" />
                                </button>
                                <button
                                    onClick={() => {
                                        if (confirm('Are you sure you want to delete this configuration?')) {
                                            onDelete(config.id)
                                        }
                                    }}
                                    className="rounded-md p-1.5 text-muted-foreground hover:bg-destructive/10 hover:text-destructive transition-colors"
                                >
                                    <Trash2 className="h-4 w-4" />
                                </button>
                            </div>
                        </div>

                        <div>
                            <h3 className="font-semibold text-lg leading-tight group-hover:text-primary transition-colors">
                                {config.name}
                            </h3>
                            <div className="mt-2 flex items-center gap-2">
                                <span className="inline-flex items-center gap-1 rounded-full bg-muted px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
                                    {getTypeIcon(config.rag_type)}
                                    {config.rag_type.replace('_', ' ')}
                                </span>
                            </div>
                        </div>

                        <div className="space-y-2 border-t border-border pt-4">
                            <div className="flex items-center justify-between text-xs">
                                <span className="text-muted-foreground uppercase tracking-tight font-medium">Provider</span>
                                <span className="font-semibold text-foreground">{config.llm_provider}</span>
                            </div>
                            <div className="flex items-center justify-between text-xs">
                                <span className="text-muted-foreground uppercase tracking-tight font-medium">Model</span>
                                <span className="font-semibold text-foreground truncate max-w-[150px]">{config.llm_model}</span>
                            </div>
                        </div>
                    </div>
                </div>
            ))}
        </div>
    )
}
