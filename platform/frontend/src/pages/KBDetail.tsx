import { useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
    ArrowLeft,
    Database,
    FileText,
    Upload,
    Trash2,
    Loader2,
    AlertCircle,
    Clock,
    CheckCircle2,
    FileBox
} from 'lucide-react'
import { api } from '@/api/client'
import { cn } from '@/lib/utils'
import { IndexKBDialog } from '@/components/knowledge-bases/IndexKBDialog'

export function KBDetail() {
    const { id } = useParams<{ id: string }>()
    const navigate = useNavigate()
    const queryClient = useQueryClient()
    const [isUploading, setIsUploading] = useState(false)

    const { data: kb, isLoading, isError } = useQuery({
        queryKey: ['knowledge-base', id],
        queryFn: () => api.knowledgeBases.get(id!),
        enabled: !!id,
    })

    const deleteDocMutation = useMutation({
        mutationFn: (docId: string) => api.knowledgeBases.deleteDocument(id!, docId),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['knowledge-base', id] })
        },
    })

    const [isIndexDialogOpen, setIsIndexDialogOpen] = useState(false)

    const indexMutation = useMutation({
        mutationFn: (ragConfigId: string) => api.knowledgeBases.index(id!, { rag_config_id: ragConfigId }),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['knowledge-base', id] })
            setIsIndexDialogOpen(false)
        },
    })

    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = Array.from(e.target.files || [])
        if (files.length === 0) return

        setIsUploading(true)
        try {
            await api.knowledgeBases.uploadDocuments(id!, files)
            queryClient.invalidateQueries({ queryKey: ['knowledge-base', id] })
        } catch (error) {
            console.error('Upload failed:', error)
        } finally {
            setIsUploading(false)
            e.target.value = ''
        }
    }

    if (isLoading) {
        return (
            <div className="flex h-[60vh] items-center justify-center">
                <Loader2 className="h-10 w-10 animate-spin text-primary" />
            </div>
        )
    }

    if (isError || !kb) {
        return (
            <div className="flex h-[60vh] flex-col items-center justify-center space-y-4">
                <AlertCircle className="h-12 w-12 text-destructive" />
                <p className="text-lg font-medium">Knowledge Base not found</p>
                <button
                    onClick={() => navigate(-1)}
                    className="text-primary hover:underline"
                >
                    Go Back
                </button>
            </div>
        )
    }

    const k = kb.data
    const documents = k.documents || []

    return (
        <div className="space-y-6 pb-12">
            <button
                onClick={() => navigate(`/projects/${k.project_id}`)}
                className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
                <ArrowLeft className="h-4 w-4" />
                Back to Project
            </button>

            <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
                <div className="flex items-center gap-4">
                    <div className="rounded-xl bg-primary/10 p-3 text-primary">
                        <Database className="h-8 w-8" />
                    </div>
                    <div>
                        <h1 className="text-3xl font-bold tracking-tight">{k.name}</h1>
                        <p className="text-muted-foreground">{k.description || 'No description provided.'}</p>
                    </div>
                </div>

                <div className="relative">
                    <input
                        type="file"
                        id="file-upload"
                        className="hidden"
                        multiple
                        onChange={handleFileUpload}
                        disabled={isUploading}
                    />
                    <label
                        htmlFor="file-upload"
                        className={cn(
                            "flex items-center gap-2 rounded-lg bg-primary px-5 py-2.5 text-sm font-bold text-primary-foreground hover:bg-primary/90 transition-all shadow-md cursor-pointer",
                            isUploading && "opacity-50 pointer-events-none"
                        )}
                    >
                        {isUploading ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                            <Upload className="h-4 w-4" />
                        )}
                        Upload Documents
                    </label>
                </div>
            </div>

            <div className="grid gap-6 md:grid-cols-3">
                <div className="md:col-span-2 space-y-6">
                    <div className="rounded-xl border border-border bg-card overflow-hidden">
                        <div className="border-b border-border bg-muted/30 px-6 py-4">
                            <h3 className="font-semibold flex items-center gap-2">
                                <FileBox className="h-4 w-4 text-muted-foreground" />
                                Documents ({documents.length})
                            </h3>
                        </div>

                        <div className="divide-y divide-border">
                            {documents.length === 0 ? (
                                <div className="py-12 text-center text-muted-foreground">
                                    <FileText className="h-10 w-10 mx-auto opacity-20 mb-3" />
                                    <p>No documents uploaded yet.</p>
                                </div>
                            ) : (
                                documents.map((doc) => (
                                    <div key={doc.id} className="flex items-center justify-between px-6 py-4 hover:bg-muted/30 transition-colors group">
                                        <div className="flex items-center gap-3 min-w-0">
                                            <div className="rounded bg-accent p-2 text-accent-foreground">
                                                <FileText className="h-4 w-4" />
                                            </div>
                                            <div className="min-w-0">
                                                <p className="text-sm font-medium truncate">{doc.filename}</p>
                                                <p className="text-[10px] text-muted-foreground">
                                                    {(doc.size_bytes / 1024).toFixed(1)} KB • {new Date(doc.created_at).toLocaleDateString()}
                                                </p>
                                            </div>
                                        </div>

                                        <div className="flex items-center gap-4">
                                            <div className={cn(
                                                "flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-medium",
                                                doc.status === 'processed' ? "bg-green-500/10 text-green-600" : "bg-yellow-500/10 text-yellow-600"
                                            )}>
                                                {doc.status === 'processed' ? <CheckCircle2 className="h-2.5 w-2.5" /> : <Clock className="h-2.5 w-2.5" />}
                                                {doc.status}
                                            </div>
                                            <button
                                                onClick={() => deleteDocMutation.mutate(doc.id)}
                                                className="rounded-md p-1.5 text-muted-foreground hover:bg-destructive/10 hover:text-destructive transition-colors opacity-0 group-hover:opacity-100"
                                            >
                                                <Trash2 className="h-4 w-4" />
                                            </button>
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>
                </div>

                <div className="space-y-6">
                    <div className="rounded-xl border border-border bg-card p-6">
                        <h3 className="font-semibold mb-4 text-sm uppercase tracking-wider text-muted-foreground">Status & Meta</h3>
                        <div className="space-y-4">
                            <div className="flex justify-between items-center text-sm">
                                <span className="text-muted-foreground">Status</span>
                                <span className={cn(
                                    "px-2 py-0.5 rounded-full text-[10px] font-bold uppercase",
                                    k.status === 'ready' ? "bg-green-500/10 text-green-600" : "bg-yellow-500/10 text-yellow-600"
                                )}>{k.status}</span>
                            </div>
                            <div className="flex justify-between items-center text-sm">
                                <span className="text-muted-foreground">Version</span>
                                <span className="font-medium font-mono text-xs">v{k.current_version}</span>
                            </div>
                            <div className="flex justify-between items-center text-sm">
                                <span className="text-muted-foreground">Total Size</span>
                                <span className="font-medium">
                                    {(documents.reduce((acc, doc) => acc + doc.size_bytes, 0) / (1024 * 1024)).toFixed(2)} MB
                                </span>
                            </div>
                        </div>

                        <button
                            onClick={() => setIsIndexDialogOpen(true)}
                            className="w-full mt-6 flex items-center justify-center gap-2 rounded-lg bg-primary px-4 py-2.5 text-sm font-bold text-primary-foreground hover:bg-primary/90 transition-all shadow-md disabled:opacity-50"
                            disabled={k.status === 'indexing' || documents.length === 0 || indexMutation.isPending}
                        >
                            {(k.status === 'indexing' || indexMutation.isPending) && <Loader2 className="h-4 w-4 animate-spin" />}
                            Re-index Knowledge Base
                        </button>
                    </div>
                </div>
            </div>

            <IndexKBDialog
                projectId={k.project_id}
                kbName={k.name}
                isOpen={isIndexDialogOpen}
                onClose={() => setIsIndexDialogOpen(false)}
                onConfirm={(ragConfigId) => indexMutation.mutate(ragConfigId)}
            />
        </div>
    )
}
