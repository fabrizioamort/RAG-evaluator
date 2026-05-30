import { useQuery } from '@tanstack/react-query'
import {
    Loader2,
    Settings,
    Database,
    Binary,
    Code,
    Clock,
    AlertCircle,
    ChevronDown,
    ChevronRight,
    Search
} from 'lucide-react'
import { useState } from 'react'
import { api, RunManifest } from '../../api/client'
import { cn } from '@/lib/utils'

interface ManifestViewerProps {
    evaluationId: string
}

export function ManifestViewer({ evaluationId }: ManifestViewerProps) {
    const { data: manifest, isLoading, error } = useQuery<RunManifest>({
        queryKey: ['evaluation-manifest', evaluationId],
        queryFn: async () => {
            const response = await api.evaluations.getManifest(evaluationId)
            return response.data
        },
    })

    if (isLoading) {
        return (
            <div className="flex flex-col items-center justify-center py-20 space-y-4">
                <Loader2 className="h-8 w-8 animate-spin text-primary/50" />
                <p className="text-sm text-muted-foreground font-medium">Loading run manifest...</p>
            </div>
        )
    }

    if (error || !manifest) {
        return (
            <div className="flex flex-col items-center justify-center py-20 space-y-4 text-center">
                <AlertCircle className="h-10 w-10 text-destructive/50" />
                <div className="space-y-1">
                    <p className="font-semibold text-destructive">Failed to load manifest</p>
                    <p className="text-sm text-muted-foreground px-10">
                        The configuration snapshot for this evaluation is unavailable.
                    </p>
                </div>
            </div>
        )
    }

    return (
        <div className="space-y-6 animate-in fade-in duration-500">
            <div className="flex items-center justify-between border-b border-border pb-4">
                <div className="flex items-center gap-3">
                    <div className="rounded-lg bg-primary/10 p-2 text-primary">
                        <Settings className="h-5 w-5" />
                    </div>
                    <div>
                        <h3 className="font-bold text-lg">Run Manifest</h3>
                        <p className="text-sm text-muted-foreground">Reproducible snapshot of the environment and configuration.</p>
                    </div>
                </div>
                <div className="text-right">
                    <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider">Executed At</p>
                    <div className="flex items-center gap-1.5 text-sm font-bold">
                        <Clock className="h-3.5 w-3.5 text-primary" />
                        {new Date(manifest.created_at).toLocaleString()}
                    </div>
                </div>
            </div>

            <div className="grid gap-6">
                <ManifestSection
                    title="Build Snapshot"
                    icon={<Code className="h-4 w-4" />}
                    subtitle="Frozen index build configuration"
                    data={manifest.build_config_snapshot || manifest.rag_config_snapshot}
                    defaultOpen={true}
                />

                <ManifestSection
                    title="Query Overrides"
                    icon={<Search className="h-4 w-4" />}
                    subtitle="Runtime changes applied to the ready index"
                    data={manifest.query_overrides}
                />

                <ManifestSection
                    title="Effective Config"
                    icon={<Code className="h-4 w-4" />}
                    subtitle="Configuration used for RAG query execution"
                    data={manifest.effective_config_snapshot || manifest.rag_config_snapshot}
                />

                <ManifestSection
                    title="Knowledge Base Version"
                    icon={<Database className="h-4 w-4" />}
                    subtitle="Snapshot of documents and versioning"
                    data={manifest.kb_version_snapshot}
                />

                <ManifestSection
                    title="Model Ecosystem"
                    icon={<Binary className="h-4 w-4" />}
                    subtitle="LLM and Judge models utilized"
                    data={{
                        generation_model: manifest.generation_model,
                        eval_judge_model: manifest.eval_judge_model,
                        rag_evaluator_version: manifest.rag_evaluator_version,
                        platform_version: manifest.platform_version,
                    }}
                />

                {manifest.prompt_templates && Object.keys(manifest.prompt_templates).length > 0 && (
                    <ManifestSection
                        title="Prompt Templates"
                        icon={<Search className="h-4 w-4" />}
                        subtitle="Custom prompts used during the run"
                        data={manifest.prompt_templates}
                    />
                )}
            </div>
        </div>
    )
}

interface ManifestSectionProps {
    title: string
    icon: React.ReactNode
    subtitle: string
    data: unknown
    defaultOpen?: boolean
}

function ManifestSection({ title, icon, subtitle, data, defaultOpen = false }: ManifestSectionProps) {
    const [isOpen, setIsOpen] = useState(defaultOpen)

    return (
        <div className="rounded-xl border border-border bg-card overflow-hidden transition-all">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="w-full flex items-center justify-between p-4 hover:bg-muted/30 transition-colors text-left"
            >
                <div className="flex items-center gap-3">
                    <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-muted border border-border/50 text-muted-foreground">
                        {icon}
                    </div>
                    <div>
                        <h4 className="font-bold text-sm tracking-tight">{title}</h4>
                        <p className="text-[10px] text-muted-foreground font-medium uppercase tracking-wider">{subtitle}</p>
                    </div>
                </div>
                {isOpen ? <ChevronDown className="h-4 w-4 text-muted-foreground" /> : <ChevronRight className="h-4 w-4 text-muted-foreground" />}
            </button>

            {isOpen && (
                <div className="p-4 border-t border-border/50 bg-muted/10">
                    <JsonTree data={data} />
                </div>
            )}
        </div>
    )
}

function JsonTree({ data, level = 0 }: { data: unknown, level?: number }) {
    if (data === null) return <span className="text-muted-foreground">null</span>
    if (typeof data !== 'object') {
        if (typeof data === 'string') return <span className="text-green-600">"{data}"</span>
        if (typeof data === 'number') return <span className="text-blue-600 font-mono">{data}</span>
        if (typeof data === 'boolean') return <span className="text-purple-600 font-bold">{data ? 'true' : 'false'}</span>
        return <span>{String(data)}</span>
    }

    const isArray = Array.isArray(data)

    return (
        <div className={cn("space-y-1.5", level > 0 ? "ml-4 border-l border-border/50 pl-4 mt-1.5" : "")}>
            {Object.entries(data).map(([key, value]) => (
                <div key={key} className="text-xs">
                    {!isArray && <span className="font-bold text-muted-foreground mr-2">{key}:</span>}
                    {typeof value === 'object' && value !== null ? (
                        <div className="mt-1">
                            <JsonTree data={value} level={level + 1} />
                        </div>
                    ) : (
                        <span>
                            {typeof value === 'string' ? (
                                <span className="text-foreground/90 font-medium leading-relaxed break-all">"{value}"</span>
                            ) : typeof value === 'number' ? (
                                <span className="text-blue-500 font-mono font-black">{value}</span>
                            ) : typeof value === 'boolean' ? (
                                <span className="text-purple-500 font-bold uppercase text-[10px] tracking-widest">{String(value)}</span>
                            ) : (
                                <span className="text-muted-foreground">{String(value)}</span>
                            )}
                        </span>
                    )}
                </div>
            ))}
            {Object.keys(data).length === 0 && <span className="text-muted-foreground italic text-[10px]">Empty {isArray ? 'array' : 'object'}</span>}
        </div>
    )
}
