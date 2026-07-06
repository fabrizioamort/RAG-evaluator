import { useState, useEffect, useCallback } from 'react'
import { useParams, useNavigate, useSearchParams } from 'react-router-dom'
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
    FileBox,
    Layers,
    Search
} from 'lucide-react'
import { api, KnowledgeBaseIndex } from '@/api/client'
import { cn } from '@/lib/utils'
import { CreateIndexDialog } from '@/components/indexes/CreateIndexDialog'
import { IndexCard } from '@/components/indexes/IndexCard'
import { useToast } from '@/components/ui/toast-context'
import { PaginationFooter } from '@/components/ui/PaginationFooter'

export function KBDetail() {
    const { id } = useParams<{ id: string }>()
    const navigate = useNavigate()
    const [searchParams, setSearchParams] = useSearchParams()
    const queryClient = useQueryClient()
    const [isUploading, setIsUploading] = useState(false)
    const { success, error } = useToast()
    const [indexes, setIndexes] = useState<KnowledgeBaseIndex[]>([])
    const [isIndexDialogOpen, setIsIndexDialogOpen] = useState(false)
    const documentPageSize = 20
    const documentSearch = searchParams.get('docSearch') || ''
    const documentOffsetParam = Number(searchParams.get('docOffset') || '0')
    const documentOffset = Number.isFinite(documentOffsetParam) && documentOffsetParam > 0 ? documentOffsetParam : 0

    const { data: kb, isLoading, isError } = useQuery({
        queryKey: ['knowledge-base', id],
        queryFn: () => api.knowledgeBases.get(id!),
        enabled: !!id,
    })

    const { data: documentsData, isLoading: isLoadingDocuments } = useQuery({
        queryKey: ['knowledge-base-documents', id, documentSearch, documentOffset],
        queryFn: () => api.knowledgeBases.listDocuments(id!, {
            limit: documentPageSize,
            offset: documentOffset,
            search: documentSearch || undefined,
        }),
        enabled: !!id,
    })

    const fetchIndexes = useCallback(async () => {
        if (!id) return
        try {
            const response = await api.indexes.list({ kb_id: id })
            setIndexes(response.data.items)
        } catch (e) {
            console.error('Failed to fetch indexes', e)
        }
    }, [id])

    useEffect(() => {
        fetchIndexes()
    }, [fetchIndexes])

    const deleteDocMutation = useMutation({
        mutationFn: (docId: string) => api.knowledgeBases.deleteDocument(id!, docId),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['knowledge-base', id] })
            queryClient.invalidateQueries({ queryKey: ['knowledge-base-documents', id] })
            success('Document deleted', 'The document has been removed.')
        },
        onError: () => {
            error('Delete failed', 'Could not delete the document.')
        },
    })

    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = Array.from(e.target.files || [])
        if (files.length === 0) return

        setIsUploading(true)
        try {
            const result = await api.knowledgeBases.uploadDocuments(id!, files)
            queryClient.invalidateQueries({ queryKey: ['knowledge-base', id] })
            queryClient.invalidateQueries({ queryKey: ['knowledge-base-documents', id] })
            const uploadedCount = result.data.uploaded.length
            success('Upload complete', `${uploadedCount} document${uploadedCount !== 1 ? 's' : ''} uploaded.`)
        } catch (err) {
            console.error('Upload failed:', err)
            error('Upload failed', 'Could not upload documents.')
        } finally {
            setIsUploading(false)
            e.target.value = ''
        }
    }

    const updateDocumentSearch = (value: string) => {
        const next = new URLSearchParams(searchParams)
        if (value.trim()) {
            next.set('docSearch', value)
        } else {
            next.delete('docSearch')
        }
        next.delete('docOffset')
        setSearchParams(next)
    }

    const updateDocumentOffset = (nextOffset: number) => {
        const next = new URLSearchParams(searchParams)
        if (nextOffset > 0) {
            next.set('docOffset', String(nextOffset))
        } else {
            next.delete('docOffset')
        }
        setSearchParams(next)
    }

    const handleDeleteDocument = (docId: string, filename: string) => {
        if (!confirm(`Delete "${filename}"? This removes the document from the knowledge base.`)) return
        deleteDocMutation.mutate(docId)
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
    const documents = documentsData?.data.items || []
    const documentTotal = documentsData?.data.total ?? k.document_count

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
                        <div className="flex flex-col gap-3 border-b border-border bg-muted/30 px-6 py-4 sm:flex-row sm:items-center sm:justify-between">
                            <h3 className="font-semibold flex items-center gap-2">
                                <FileBox className="h-4 w-4 text-muted-foreground" />
                                Documents ({documentTotal})
                            </h3>
                            <div className="relative w-full sm:w-72">
                                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                                <input
                                    type="text"
                                    value={documentSearch}
                                    onChange={(event) => updateDocumentSearch(event.target.value)}
                                    placeholder="Search filenames..."
                                    className="h-9 w-full rounded-lg border border-input bg-background pl-9 pr-3 text-sm"
                                />
                            </div>
                        </div>

                        <div className="divide-y divide-border">
                            {isLoadingDocuments ? (
                                <div className="flex justify-center py-12">
                                    <Loader2 className="h-6 w-6 animate-spin text-primary/50" />
                                </div>
                            ) : documents.length === 0 ? (
                                <div className="py-12 text-center text-muted-foreground">
                                    <FileText className="h-10 w-10 mx-auto opacity-20 mb-3" />
                                    <p>{documentSearch ? 'No documents match your search.' : 'No documents uploaded yet.'}</p>
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
                                                onClick={() => handleDeleteDocument(doc.id, doc.filename)}
                                                className="rounded-md p-1.5 text-muted-foreground hover:bg-destructive/10 hover:text-destructive transition-colors opacity-0 group-hover:opacity-100"
                                            >
                                                <Trash2 className="h-4 w-4" />
                                            </button>
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>
                        <PaginationFooter
                            total={documentTotal}
                            offset={documentsData?.data.offset ?? documentOffset}
                            limit={documentsData?.data.limit ?? documentPageSize}
                            onPageChange={updateDocumentOffset}
                            isLoading={isLoadingDocuments}
                        />
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
                                <span className="text-muted-foreground">Documents</span>
                                <span className="font-medium">
                                    {k.document_count}
                                </span>
                            </div>
                        </div>

                        <button
                            onClick={() => setIsIndexDialogOpen(true)}
                            className="w-full mt-6 flex items-center justify-center gap-2 rounded-lg bg-primary px-4 py-2.5 text-sm font-bold text-primary-foreground hover:bg-primary/90 transition-all shadow-md disabled:opacity-50"
                            disabled={k.document_count === 0}
                        >
                            <Layers className="h-4 w-4" />
                            Create Index
                        </button>
                    </div>
                </div>
            </div>

            {/* Indexes Section */}
            <div className="space-y-4">
                <div className="flex items-center justify-between border-b pb-2">
                    <h2 className="text-lg font-semibold flex items-center gap-2">
                        <Layers className="h-5 w-5 text-muted-foreground" />
                        Indexes
                    </h2>
                    <span className="text-sm text-muted-foreground">
                        {indexes.length} index{indexes.length !== 1 ? 'es' : ''} available
                    </span>
                </div>

                {indexes.length === 0 ? (
                    <div className="rounded-xl border border-dashed border-border p-8 text-center text-muted-foreground bg-muted/20">
                        <Layers className="h-12 w-12 mx-auto opacity-20 mb-3" />
                        <p className="font-medium">No indexes yet</p>
                        <p className="text-sm mt-1">Create an index to start running evaluations.</p>
                        <button
                            onClick={() => setIsIndexDialogOpen(true)}
                            className="mt-4 text-primary hover:underline text-sm"
                            disabled={k.document_count === 0}
                        >
                            Create your first index
                        </button>
                    </div>
                ) : (
                    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                        {indexes.map(index => (
                            <IndexCard
                                key={index.id}
                                index={index}
                                onDelete={fetchIndexes}
                            />
                        ))}
                    </div>
                )}
            </div>

            {isIndexDialogOpen && (
                <CreateIndexDialog
                    projectId={k.project_id}
                    knowledgeBaseId={k.id}
                    onClose={() => setIsIndexDialogOpen(false)}
                    onCreated={() => {
                        fetchIndexes()
                    }}
                />
            )}
        </div>
    )
}
