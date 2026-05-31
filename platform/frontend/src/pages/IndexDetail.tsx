import { useState, useEffect, useCallback } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'
import { api, KnowledgeBaseIndex } from '../api/client'
import { IndexBuildProgress } from '../components/indexes/IndexBuildProgress'
import { ArrowLeft, Loader2, Database, Cpu, Calendar, FileText, HardDrive, Trash2, Play } from 'lucide-react'

function timeAgo(dateString: string) {
    const date = new Date(dateString)
    const now = new Date()
    const seconds = Math.floor((now.getTime() - date.getTime()) / 1000)

    if (seconds < 60) return 'just now'
    const minutes = Math.floor(seconds / 60)
    if (minutes < 60) return `${minutes}m ago`
    const hours = Math.floor(minutes / 60)
    if (hours < 24) return `${hours}h ago`
    const days = Math.floor(hours / 24)
    return `${days}d ago`
}

export function IndexDetail() {
    const { id } = useParams<{ id: string }>()
    const navigate = useNavigate()
    const [index, setIndex] = useState<KnowledgeBaseIndex | null>(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)

    const fetchIndex = useCallback(async () => {
        if (!id) return
        try {
            const response = await api.indexes.get(id)
            setIndex(response.data)
        } catch (err: unknown) {
            setError('Failed to fetch index details')
        } finally {
            setLoading(false)
        }
    }, [id])

    useEffect(() => {
        fetchIndex()
    }, [fetchIndex])

    const handleDelete = async () => {
        if (!id || !confirm('Are you sure you want to delete this index?')) return
        try {
            await api.indexes.delete(id)
            navigate('/indexes')
        } catch (err) {
            alert('Failed to delete: ' + (err as Error).message)
        }
    }

    const handleRunEvaluation = () => {
        if (!index?.project_id) {
            alert('Unable to start evaluation: missing project information.')
            return
        }
        const params = new URLSearchParams({
            tab: 'evals',
            startEval: '1',
            kbId: index.knowledge_base_id,
            indexId: index.id,
        })
        navigate(`/projects/${index.project_id}?${params.toString()}`)
    }

    if (loading) {
        return (
            <div className="flex justify-center items-center h-full min-h-[400px]">
                <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
            </div>
        )
    }

    if (error || !index) {
        return (
            <div className="text-center py-12">
                <h2 className="text-xl font-semibold text-gray-900">Error</h2>
                <p className="text-gray-500 mt-2">{error || 'Index not found'}</p>
                <Link to="/indexes" className="text-blue-600 hover:underline mt-4 inline-block">
                    Back to Indexes
                </Link>
            </div>
        )
    }

    return (
        <div className="space-y-6">
            <div className="flex items-center space-x-4">
                <Link to="/indexes" className="text-gray-500 hover:text-gray-700">
                    <ArrowLeft className="h-5 w-5" />
                </Link>
                <div>
                    <h1 className="text-2xl font-bold text-gray-900">{index.name}</h1>
                    <p className="text-sm text-gray-500">{index.physical_id}</p>
                </div>
                <div className="ml-auto flex space-x-3">
                    {index.status === 'ready' && (
                        <button
                            onClick={handleRunEvaluation}
                            className="flex items-center px-3 py-2 bg-green-600 text-white rounded-md hover:bg-green-700"
                        >
                            <Play className="h-4 w-4 mr-2" />
                            Run Evaluation
                        </button>
                    )}
                    <button
                        onClick={handleDelete}
                        className="flex items-center px-3 py-2 border border-red-200 text-red-600 rounded-md hover:bg-red-50"
                    >
                        <Trash2 className="h-4 w-4 mr-2" />
                        Delete
                    </button>
                </div>
            </div>

            {/* Build Progress if active */}
            {(index.status === 'building' || index.status === 'pending') && (
                <IndexBuildProgress
                    indexId={index.id}
                    onComplete={fetchIndex}
                    onError={() => fetchIndex()} // Refresh to show error state
                />
            )}

            {index.status === 'failed' && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                    <h3 className="text-red-800 font-medium">Build Failed</h3>
                    <p className="text-red-600 text-sm mt-1">{index.error_message}</p>
                </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Overview Card */}
                <div className="bg-white p-6 rounded-lg border shadow-sm space-y-4">
                    <h3 className="font-semibold text-lg border-b pb-2">Overview</h3>

                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <span className="text-sm text-gray-500 block">Status</span>
                            <span className={`inline-block px-2 py-0.5 rounded-full text-sm font-medium ${index.status === 'ready' ? 'bg-green-100 text-green-800' :
                                index.status === 'building' ? 'bg-blue-100 text-blue-800' :
                                    index.status === 'failed' ? 'bg-red-100 text-red-800' :
                                        'bg-gray-100 text-gray-800'
                                }`}>
                                {index.status.charAt(0).toUpperCase() + index.status.slice(1)}
                            </span>
                        </div>
                        <div>
                            <span className="text-sm text-gray-500 block">Created</span>
                            <span className="text-sm font-medium flex items-center">
                                <Calendar className="h-3 w-3 mr-1 text-gray-400" />
                                {timeAgo(index.created_at)}
                            </span>
                        </div>
                        <div>
                            <span className="text-sm text-gray-500 block">Chunks</span>
                            <span className="text-sm font-medium flex items-center">
                                <FileText className="h-3 w-3 mr-1 text-gray-400" />
                                {index.chunk_count}
                            </span>
                        </div>
                        <div>
                            <span className="text-sm text-gray-500 block">Documents</span>
                            <span className="text-sm font-medium flex items-center">
                                <Database className="h-3 w-3 mr-1 text-gray-400" />
                                {index.document_count}
                            </span>
                        </div>
                        <div>
                            <span className="text-sm text-gray-500 block">Storage Type</span>
                            <span className="text-sm font-medium flex items-center">
                                <HardDrive className="h-3 w-3 mr-1 text-gray-400" />
                                {index.storage_type}
                            </span>
                        </div>
                    </div>

                    <div className="pt-4">
                        <span className="text-sm text-gray-500 block mb-1">Source Knowledge Base</span>
                        <Link to={`/knowledge-bases/${index.knowledge_base_id}`} className="text-blue-600 hover:underline flex items-center">
                            <Database className="h-4 w-4 mr-2" />
                            {index.knowledge_base_name}
                        </Link>
                    </div>
                </div>

                {/* Config Card */}
                <div className="bg-white p-6 rounded-lg border shadow-sm space-y-4">
                    <h3 className="font-semibold text-lg border-b pb-2">Configuration Snapshot</h3>

                    <div className="space-y-3">
                        <div>
                            <span className="text-sm text-gray-500 block">Original RAG Config</span>
                            {index.project_id ? (
                                <Link to={`/projects/${index.project_id}?tab=rags`} className="text-blue-600 hover:underline flex items-center">
                                    <Cpu className="h-4 w-4 mr-2" />
                                    {index.rag_config_name}
                                </Link>
                            ) : (
                                <span className="flex items-center text-sm font-medium text-gray-900">
                                    <Cpu className="h-4 w-4 mr-2 text-gray-400" />
                                    {index.rag_config_name}
                                </span>
                            )}
                        </div>

                        <div className="bg-gray-50 p-3 rounded-md text-sm font-mono overflow-auto max-h-[300px]">
                            <pre>{JSON.stringify(index.config_snapshot, null, 2)}</pre>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
