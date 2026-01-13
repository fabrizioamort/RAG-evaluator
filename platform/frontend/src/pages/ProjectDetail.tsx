import { useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
    ArrowLeft,
    Database,
    FileText,
    Settings2,
    FlaskConical,
    Calendar,
    Tag,
    Loader2,
    AlertCircle,
    Plus
} from 'lucide-react'
import { api, KnowledgeBaseCreate } from '@/api/client'
import { cn } from '@/lib/utils'
import { KBList } from '@/components/knowledge-bases/KBList'
import { CreateKBDialog } from '@/components/knowledge-bases/CreateKBDialog'

function KnowledgeBasesTab({ projectId }: { projectId: string }) {
    const [isDialogOpen, setIsDialogOpen] = useState(false)
    const queryClient = useQueryClient()

    const { data, isLoading } = useQuery({
        queryKey: ['knowledge-bases', projectId],
        queryFn: () => api.knowledgeBases.list(projectId),
        enabled: !!projectId,
    })

    const createMutation = useMutation({
        mutationFn: (newKB: KnowledgeBaseCreate) => api.knowledgeBases.create(projectId, newKB),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['knowledge-bases', projectId] })
        },
    })

    if (isLoading) {
        return (
            <div className="flex justify-center py-20">
                <Loader2 className="h-8 w-8 animate-spin text-primary/50" />
            </div>
        )
    }

    const kbs = data?.data?.items || []

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-xl font-semibold">Knowledge Bases</h2>
                    <p className="text-sm text-muted-foreground">Manage documents for retrieval and indexing.</p>
                </div>
                {kbs.length > 0 && (
                    <button
                        onClick={() => setIsDialogOpen(true)}
                        className="flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-all shadow-md"
                    >
                        <Plus className="h-4 w-4" />
                        New KB
                    </button>
                )}
            </div>

            <KBList
                knowledgeBases={kbs}
                onCreateClick={() => setIsDialogOpen(true)}
            />

            <CreateKBDialog
                isOpen={isDialogOpen}
                onClose={() => setIsDialogOpen(false)}
                onSubmit={async (kbData: KnowledgeBaseCreate) => {
                    await createMutation.mutateAsync(kbData)
                }}
            />
        </div>
    )
}
const TestSetsTab = () => <div className="py-10 text-center text-muted-foreground">Test Sets content coming soon...</div>
const RAGConfigsTab = () => <div className="py-10 text-center text-muted-foreground">RAG Configurations content coming soon...</div>
const EvaluationsTab = () => <div className="py-10 text-center text-muted-foreground">Evaluations content coming soon...</div>

const tabs = [
    { id: 'kb', name: 'Knowledge Bases', icon: Database },
    { id: 'tests', name: 'Test Sets', icon: FileText },
    { id: 'rags', name: 'RAG Configs', icon: Settings2 },
    { id: 'evals', name: 'Evaluations', icon: FlaskConical },
]

export function ProjectDetail() {
    const { id } = useParams<{ id: string }>()
    const navigate = useNavigate()
    const [activeTab, setActiveTab] = useState('kb')

    const { data: project, isLoading, isError } = useQuery({
        queryKey: ['project', id],
        queryFn: () => api.projects.get(id!),
        enabled: !!id,
    })

    if (isLoading) {
        return (
            <div className="flex h-[60vh] items-center justify-center">
                <Loader2 className="h-10 w-10 animate-spin text-primary" />
            </div>
        )
    }

    if (isError || !project) {
        return (
            <div className="flex h-[60vh] flex-col items-center justify-center space-y-4">
                <AlertCircle className="h-12 w-12 text-destructive" />
                <p className="text-lg font-medium">Project not found</p>
                <button
                    onClick={() => navigate('/projects')}
                    className="text-primary hover:underline"
                >
                    Back to Projects
                </button>
            </div>
        )
    }

    const p = project.data

    return (
        <div className="space-y-6 pb-10">
            {/* Breadcrumbs / Back */}
            <button
                onClick={() => navigate('/projects')}
                className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
                <ArrowLeft className="h-4 w-4" />
                Back to Projects
            </button>

            {/* Project Header */}
            <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
                <div className="space-y-1">
                    <h1 className="text-3xl font-bold tracking-tight">{p.name}</h1>
                    <p className="text-muted-foreground max-w-3xl">
                        {p.description || 'No description provided.'}
                    </p>
                    <div className="flex flex-wrap gap-2 pt-2">
                        <div className="flex items-center gap-1.5 rounded-full bg-muted px-2.5 py-0.5 text-xs font-medium text-muted-foreground">
                            <Calendar className="h-3 w-3" />
                            Created {new Date(p.created_at).toLocaleDateString()}
                        </div>
                        {p.tags.map(tag => (
                            <div key={tag} className="flex items-center gap-1.5 rounded-full bg-primary/10 px-2.5 py-0.5 text-xs font-medium text-primary">
                                <Tag className="h-3 w-3" />
                                {tag}
                            </div>
                        ))}
                    </div>
                </div>
                <div className="flex items-center gap-2">
                    <button className="rounded-lg border border-border bg-card px-4 py-2 text-sm font-medium hover:bg-accent transition-colors">
                        Edit Project
                    </button>
                    <button className="rounded-lg bg-destructive/10 text-destructive border border-destructive/20 px-4 py-2 text-sm font-medium hover:bg-destructive/20 transition-colors">
                        Archive
                    </button>
                </div>
            </div>

            {/* Tabs */}
            <div className="border-b border-border">
                <nav className="flex gap-8">
                    {tabs.map((tab) => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            className={cn(
                                "flex items-center gap-2 py-4 text-sm font-medium border-b-2 transition-all",
                                activeTab === tab.id
                                    ? "border-primary text-primary"
                                    : "border-transparent text-muted-foreground hover:text-foreground hover:border-border"
                            )}
                        >
                            <tab.icon className="h-4 w-4" />
                            {tab.name}
                        </button>
                    ))}
                </nav>
            </div>

            {/* Tab Content */}
            <div className="mt-6">
                {activeTab === 'kb' && <KnowledgeBasesTab projectId={p.id} />}
                {activeTab === 'tests' && <TestSetsTab />}
                {activeTab === 'rags' && <RAGConfigsTab />}
                {activeTab === 'evals' && <EvaluationsTab />}
            </div>
        </div>
    )
}
