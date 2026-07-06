import { useState, useEffect } from 'react'
import { useParams, useNavigate, useSearchParams } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { isAxiosError } from 'axios'
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
    Plus,
    FileUp,
    Play,
    ChevronRight,
    TrendingUp,
    GitCompare,
    Layers,
    CheckCircle2
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { api, KnowledgeBaseCreate, Evaluation, KnowledgeBaseIndex, Project, RAGConfig } from '@/api/client'
import { useToast } from '@/components/ui/toast-context'
import { TestSetList } from '@/components/test-sets/TestSetList'
import { TestSetDetail } from '@/components/test-sets/TestSetDetail'
import { CreateTestSetDialog } from '@/components/test-sets/CreateTestSetDialog'
import { ImportTestSetDialog } from '@/components/test-sets/ImportTestSetDialog'
import { RAGConfigList } from '@/components/rag-configs/RAGConfigList'
import { RAGConfigDialog } from '@/components/rag-configs/RAGConfigDialog'
import { KBList } from '@/components/knowledge-bases/KBList'
import { CreateKBDialog } from '@/components/knowledge-bases/CreateKBDialog'
import { StartEvaluationWizard } from '@/components/evaluations/StartEvaluationWizard'
import { TrendChart } from '@/components/trends/TrendChart'
import { EfficiencyMap } from '@/components/trends/EfficiencyMap'
import { ComparisonsTab } from '@/components/comparisons/ComparisonsTab'
import { EditProjectDialog } from '@/components/projects/EditProjectDialog'
import { IndexCard } from '@/components/indexes/IndexCard'
import { PaginationFooter } from '@/components/ui/PaginationFooter'

function ProjectOverviewTab({
    project,
    onSelectTab,
}: {
    project: Project
    onSelectTab: (tabId: string, params?: Record<string, string>) => void
}) {
    const { data: kbsData, isLoading: isLoadingKbs } = useQuery({
        queryKey: ['knowledge-bases', project.id],
        queryFn: () => api.knowledgeBases.list(project.id),
        enabled: !!project.id,
    })
    const { data: indexesData, isLoading: isLoadingIndexes } = useQuery({
        queryKey: ['indexes', project.id],
        queryFn: () => api.indexes.list({ project_id: project.id, limit: 100 }),
        enabled: !!project.id,
    })
    const { data: testSetsData, isLoading: isLoadingTestSets } = useQuery({
        queryKey: ['test-sets', project.id],
        queryFn: () => api.testSets.list(project.id),
        enabled: !!project.id,
    })
    const { data: ragConfigsData, isLoading: isLoadingConfigs } = useQuery({
        queryKey: ['rag-configs', project.id],
        queryFn: () => api.ragConfigs.list(project.id),
        enabled: !!project.id,
    })
    const { data: evaluationsData, isLoading: isLoadingEvaluations } = useQuery({
        queryKey: ['evaluations', project.id],
        queryFn: () => api.evaluations.list(project.id),
        enabled: !!project.id,
    })

    const kbs = kbsData?.data.items ?? []
    const indexes = indexesData?.data.items ?? []
    const testSets = testSetsData?.data.items ?? []
    const ragConfigs = ragConfigsData?.data.items ?? []
    const evaluations = evaluationsData?.data.items ?? []
    const documentCount = kbs.reduce((sum, kb) => sum + kb.document_count, 0)
    const readyIndexes = indexes.filter(index => index.status === 'ready')
    const hasBaseline = evaluations.some(evaluation => evaluation.is_baseline)
    const isLoading = isLoadingKbs || isLoadingIndexes || isLoadingTestSets || isLoadingConfigs || isLoadingEvaluations

    const readiness = [
        {
            id: 'kb',
            label: 'Knowledge bases',
            value: kbs.length,
            complete: kbs.length > 0,
            action: kbs.length > 0 ? 'Manage KBs' : 'Create KB',
            icon: Database,
        },
        {
            id: 'kb',
            label: 'Documents',
            value: documentCount,
            complete: documentCount > 0,
            action: documentCount > 0 ? 'Review documents' : 'Upload documents',
            icon: FileText,
        },
        {
            id: 'rags',
            label: 'RAG configs',
            value: ragConfigs.length,
            complete: ragConfigs.length > 0,
            action: ragConfigs.length > 0 ? 'Manage configs' : 'Create config',
            icon: Settings2,
        },
        {
            id: 'indexes',
            label: 'Ready indexes',
            value: readyIndexes.length,
            complete: readyIndexes.length > 0,
            action: readyIndexes.length > 0 ? 'View indexes' : 'Build index',
            icon: Layers,
        },
        {
            id: 'tests',
            label: 'Test sets',
            value: testSets.length,
            complete: testSets.length > 0,
            action: testSets.length > 0 ? 'Manage tests' : 'Create test set',
            icon: FileText,
        },
        {
            id: 'evals',
            label: 'Evaluations',
            value: evaluations.length,
            complete: evaluations.length > 0,
            action: evaluations.length > 0 ? 'View evaluations' : 'Launch evaluation',
            icon: FlaskConical,
        },
        {
            id: 'evals',
            label: 'Baseline',
            value: hasBaseline ? 1 : 0,
            complete: hasBaseline,
            action: hasBaseline ? 'View baseline' : 'Set baseline',
            icon: CheckCircle2,
        },
    ]

    const nextStep = readiness.find(item => !item.complete)

    if (isLoading) {
        return (
            <div className="flex justify-center py-20">
                <Loader2 className="h-8 w-8 animate-spin text-primary/50" />
            </div>
        )
    }

    return (
        <div className="space-y-6">
            <div className="rounded-xl border border-border bg-card p-6">
                <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                    <div>
                        <h2 className="text-xl font-semibold">Project Readiness</h2>
                        <p className="mt-1 text-sm text-muted-foreground">
                            Follow the setup path from documents and RAG config to an index, test set, evaluation, and baseline.
                        </p>
                    </div>
                    {nextStep ? (
                        <button
                            onClick={() => onSelectTab(nextStep.id)}
                            className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground hover:bg-primary/90"
                        >
                            <ChevronRight className="h-4 w-4" />
                            {nextStep.action}
                        </button>
                    ) : (
                        <div className="inline-flex items-center gap-2 rounded-lg bg-green-500/10 px-4 py-2 text-sm font-semibold text-green-700">
                            <CheckCircle2 className="h-4 w-4" />
                            Ready for comparison
                        </div>
                    )}
                </div>
            </div>

            <div className="grid gap-3">
                {readiness.map((item) => (
                    <button
                        key={`${item.label}-${item.id}`}
                        onClick={() => onSelectTab(item.id)}
                        className="flex items-center justify-between rounded-xl border border-border bg-card p-4 text-left transition-all hover:border-primary/50 hover:bg-accent/40"
                    >
                        <div className="flex items-center gap-4">
                            <div className={cn(
                                'flex h-10 w-10 items-center justify-center rounded-lg',
                                item.complete ? 'bg-green-500/10 text-green-600' : 'bg-muted text-muted-foreground'
                            )}>
                                <item.icon className="h-5 w-5" />
                            </div>
                            <div>
                                <p className="font-semibold">{item.label}</p>
                                <p className="text-xs text-muted-foreground">{item.action}</p>
                            </div>
                        </div>
                        <div className="flex items-center gap-3">
                            <span className="text-lg font-bold tabular-nums">{item.value}</span>
                            <ChevronRight className="h-4 w-4 text-muted-foreground" />
                        </div>
                    </button>
                ))}
            </div>
        </div>
    )
}

function ProjectIndexesTab({ projectId }: { projectId: string }) {
    const navigate = useNavigate()
    const queryClient = useQueryClient()

    const { data, isLoading } = useQuery({
        queryKey: ['indexes', projectId],
        queryFn: () => api.indexes.list({ project_id: projectId, limit: 100 }),
        enabled: !!projectId,
    })

    const runEvaluationFromIndex = (index: KnowledgeBaseIndex) => {
        const params = new URLSearchParams({
            tab: 'evals',
            startEval: '1',
            kbId: index.knowledge_base_id,
            indexId: index.id,
        })
        navigate(`/projects/${projectId}?${params.toString()}`)
    }

    if (isLoading) {
        return (
            <div className="flex justify-center py-20">
                <Loader2 className="h-8 w-8 animate-spin text-primary/50" />
            </div>
        )
    }

    const indexes = data?.data.items ?? []

    return (
        <div className="space-y-6">
            <div>
                <h2 className="text-xl font-semibold">Indexes</h2>
                <p className="text-sm text-muted-foreground">Review build status and launch evaluations from ready indexes.</p>
            </div>

            {indexes.length === 0 ? (
                <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-border bg-card/50 py-20">
                    <Layers className="h-10 w-10 text-muted-foreground/50" />
                    <h3 className="mt-5 text-xl font-semibold">No indexes yet</h3>
                    <p className="mt-2 max-w-sm text-center text-muted-foreground">
                        Create a knowledge base and RAG config, then build an index to make it available for evaluation.
                    </p>
                </div>
            ) : (
                <div className="grid grid-cols-1 gap-6 md:grid-cols-2 xl:grid-cols-3">
                    {indexes.map((index) => (
                        <IndexCard
                            key={index.id}
                            index={{ ...index, project_id: index.project_id ?? projectId }}
                            onDelete={() => queryClient.invalidateQueries({ queryKey: ['indexes', projectId] })}
                            onRunEvaluation={index.status === 'ready' ? () => runEvaluationFromIndex(index) : undefined}
                        />
                    ))}
                </div>
            )}
        </div>
    )
}

function KnowledgeBasesTab({ projectId }: { projectId: string }) {
    const [isDialogOpen, setIsDialogOpen] = useState(false)
    const queryClient = useQueryClient()
    const { success, error } = useToast()

    const { data, isLoading } = useQuery({
        queryKey: ['knowledge-bases', projectId],
        queryFn: () => api.knowledgeBases.list(projectId),
        enabled: !!projectId,
    })

    const createMutation = useMutation({
        mutationFn: (newKB: KnowledgeBaseCreate) => api.knowledgeBases.create(projectId, newKB),
        onSuccess: (response) => {
            queryClient.invalidateQueries({ queryKey: ['knowledge-bases', projectId] })
            success('Knowledge Base created', `"${response.data.name}" is ready for documents.`)
            setIsDialogOpen(false)
        },
        onError: () => {
            error('Failed to create KB', 'Please try again.')
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
                <button
                    onClick={() => setIsDialogOpen(true)}
                    className="flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-all shadow-md"
                >
                    <Plus className="h-4 w-4" />
                    New KB
                </button>
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

function TestSetsTab({ projectId }: { projectId: string }) {
    const navigate = useNavigate()
    const [selectedTestSetId, setSelectedTestSetId] = useState<string | null>(null)
    const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false)
    const [isImportDialogOpen, setIsImportDialogOpen] = useState(false)
    const queryClient = useQueryClient()
    const { success, error } = useToast()

    const { data, isLoading } = useQuery({
        queryKey: ['test-sets', projectId],
        queryFn: () => api.testSets.list(projectId),
        enabled: !!projectId,
    })

    const createMutation = useMutation({
        mutationFn: (data: unknown) => api.testSets.create(projectId, data as Parameters<typeof api.testSets.create>[1]),
        onSuccess: (response) => {
            queryClient.invalidateQueries({ queryKey: ['test-sets', projectId] })
            success('Test Set created', `"${response.data.name}" is ready for test cases.`)
            setIsCreateDialogOpen(false)
        },
        onError: () => {
            error('Failed to create test set', 'Please try again.')
        },
    })

    const importMutation = useMutation({
        mutationFn: (data: unknown) => api.testSets.import(projectId, data),
        onSuccess: (response) => {
            queryClient.invalidateQueries({ queryKey: ['test-sets', projectId] })
            const importedName = response.data && typeof response.data === 'object' && 'name' in response.data
                ? String(response.data.name)
                : 'Imported test set'
            success('Test Set imported', `"${importedName}" is ready for evaluation.`)
            setIsImportDialogOpen(false)
        },
        onError: () => {
            error('Failed to import test set', 'Check that the JSON file contains valid test cases.')
        },
    })

    const deleteMutation = useMutation({
        mutationFn: (id: string) => api.testSets.delete(id),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['test-sets', projectId] })
            success('Test Set deleted', 'The test set has been removed.')
        },
        onError: () => {
            error('Failed to delete', 'Please try again.')
        },
    })

    if (selectedTestSetId) {
        return <TestSetDetail testSetId={selectedTestSetId} projectId={projectId} onBack={() => setSelectedTestSetId(null)} />
    }

    if (isLoading) {
        return (
            <div className="flex justify-center py-20">
                <Loader2 className="h-8 w-8 animate-spin text-primary/50" />
            </div>
        )
    }

    const testSets = data?.data?.items || []

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-xl font-semibold">Test Sets</h2>
                    <p className="text-sm text-muted-foreground">Manage collections of questions and answers for evaluation.</p>
                </div>
                {testSets.length > 0 && (
                    <div className="flex items-center gap-2">
                        <button
                            onClick={() => setIsImportDialogOpen(true)}
                            className="flex items-center gap-2 rounded-lg border border-border bg-card px-4 py-2 text-sm font-semibold hover:bg-accent transition-all"
                        >
                            <FileUp className="h-4 w-4" />
                            Import JSON
                        </button>
                        <button
                            onClick={() => setIsCreateDialogOpen(true)}
                            className="flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-all shadow-md"
                        >
                            <Plus className="h-4 w-4" />
                            New Test Set
                        </button>
                    </div>
                )}
            </div>

            <TestSetList
                testSets={testSets}
                onCreateClick={() => setIsCreateDialogOpen(true)}
                onImportClick={() => setIsImportDialogOpen(true)}
                onViewDetail={(id) => navigate(`/projects/${projectId}/test-sets/${id}`)}
                onDelete={(id) => deleteMutation.mutate(id)}
                onExport={() => {
                    // Export logic handled in detail, or add here if needed for list
                }}
            />

            <CreateTestSetDialog
                isOpen={isCreateDialogOpen}
                onClose={() => setIsCreateDialogOpen(false)}
                onSubmit={async (data) => {
                    await createMutation.mutateAsync(data)
                }}
            />

            <ImportTestSetDialog
                isOpen={isImportDialogOpen}
                onClose={() => setIsImportDialogOpen(false)}
                onSubmit={async (data) => {
                    await importMutation.mutateAsync(data)
                }}
            />
        </div>
    )
}

function RAGConfigsTab({ projectId }: { projectId: string }) {
    const [isDialogOpen, setIsDialogOpen] = useState(false)
    const [editingConfig, setEditingConfig] = useState<RAGConfig | undefined>(undefined)
    const queryClient = useQueryClient()
    const { success, error } = useToast()

    const getApiDetail = (err: unknown): string => {
        if (isAxiosError(err)) {
            const detail = err.response?.data?.detail
            if (typeof detail === 'string' && detail.trim()) {
                return detail
            }
        }
        if (err instanceof Error && err.message) {
            return err.message
        }
        return 'Please try again.'
    }

    const { data, isLoading } = useQuery({
        queryKey: ['rag-configs', projectId],
        queryFn: () => api.ragConfigs.list(projectId),
        enabled: !!projectId,
    })

    const createMutation = useMutation({
        mutationFn: (data: Parameters<typeof api.ragConfigs.create>[1]) => api.ragConfigs.create(projectId, data),
        onSuccess: (response) => {
            queryClient.invalidateQueries({ queryKey: ['rag-configs', projectId] })
            success('RAG Config created', `"${response.data.name}" configuration saved.`)
            setIsDialogOpen(false)
        },
        onError: (err) => {
            error('Failed to create config', getApiDetail(err))
        },
    })

    const updateMutation = useMutation({
        mutationFn: ({ id, data }: { id: string, data: Parameters<typeof api.ragConfigs.update>[1] }) => api.ragConfigs.update(id, data),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['rag-configs', projectId] })
            success('Config updated', 'Configuration changes saved.')
            setIsDialogOpen(false)
        },
        onError: (err) => {
            error('Failed to update', getApiDetail(err))
        },
    })

    const deleteMutation = useMutation({
        mutationFn: (id: string) => api.ragConfigs.delete(id),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['rag-configs', projectId] })
            success('Config deleted', 'The configuration has been removed.')
        },
        onError: () => {
            error('Failed to delete', 'Please try again.')
        },
    })

    if (isLoading) {
        return (
            <div className="flex justify-center py-20">
                <Loader2 className="h-8 w-8 animate-spin text-primary/50" />
            </div>
        )
    }

    const configs = data?.data?.items || []

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-xl font-semibold">RAG Configurations</h2>
                    <p className="text-sm text-muted-foreground">Define different retrieval and generation settings.</p>
                </div>
                {configs.length > 0 && (
                    <button
                        onClick={() => {
                            setEditingConfig(undefined)
                            setIsDialogOpen(true)
                        }}
                        className="flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-all shadow-md"
                    >
                        <Plus className="h-4 w-4" />
                        New Config
                    </button>
                )}
            </div>

            <RAGConfigList
                configs={configs}
                onCreateClick={() => {
                    setEditingConfig(undefined)
                    setIsDialogOpen(true)
                }}
                onEdit={(config) => {
                    setEditingConfig(config)
                    setIsDialogOpen(true)
                }}
                onDelete={(id) => deleteMutation.mutate(id)}
            />

            <RAGConfigDialog
                isOpen={isDialogOpen}
                onClose={() => setIsDialogOpen(false)}
                config={editingConfig}
                onSubmit={async (data) => {
                    if (editingConfig) {
                        await updateMutation.mutateAsync({ id: editingConfig.id, data })
                    } else {
                        await createMutation.mutateAsync(data)
                    }
                }}
            />
        </div>
    )
}

function EvaluationsTab({
    projectId,
    launchWizard,
    initialKnowledgeBaseId,
    initialIndexId,
}: {
    projectId: string
    launchWizard?: boolean
    initialKnowledgeBaseId?: string
    initialIndexId?: string
}) {
    const navigate = useNavigate()
    const [searchParams, setSearchParams] = useSearchParams()
    const [isWizardOpen, setIsWizardOpen] = useState(false)
    const [hasAutoOpened, setHasAutoOpened] = useState(false)
    const queryClient = useQueryClient()
    const pageSize = 20
    const offsetParam = Number(searchParams.get('evalOffset') || '0')
    const offset = Number.isFinite(offsetParam) && offsetParam > 0 ? offsetParam : 0
    const statusFilter = searchParams.get('evalStatus') || ''
    const testSetFilter = searchParams.get('evalTestSet') || ''
    const ragConfigFilter = searchParams.get('evalRagConfig') || ''
    const indexFilter = searchParams.get('evalIndex') || ''

    const { data, isLoading } = useQuery({
        queryKey: ['evaluations', projectId, statusFilter, testSetFilter, ragConfigFilter, indexFilter, offset],
        queryFn: () => api.evaluations.list(projectId, {
            limit: pageSize,
            offset,
            status: statusFilter || undefined,
            test_set_id: testSetFilter || undefined,
            rag_config_id: ragConfigFilter || undefined,
            knowledge_base_index_id: indexFilter || undefined,
        }),
        enabled: !!projectId,
    })

    const { data: testSetsData } = useQuery({
        queryKey: ['test-sets', projectId, 'eval-filter-options'],
        queryFn: () => api.testSets.list(projectId, { limit: 100 }),
        enabled: !!projectId,
    })
    const { data: ragConfigsData } = useQuery({
        queryKey: ['rag-configs', projectId, 'eval-filter-options'],
        queryFn: () => api.ragConfigs.list(projectId, { limit: 100 }),
        enabled: !!projectId,
    })
    const { data: indexesData } = useQuery({
        queryKey: ['indexes', projectId, 'eval-filter-options'],
        queryFn: () => api.indexes.list({ project_id: projectId, limit: 100 }),
        enabled: !!projectId,
    })

    const evaluations = data?.data?.items || []
    const hasFilters = Boolean(statusFilter || testSetFilter || ragConfigFilter || indexFilter)
    const total = data?.data?.total ?? 0
    const testSets = testSetsData?.data.items ?? []
    const ragConfigs = ragConfigsData?.data.items ?? []
    const indexes = indexesData?.data.items ?? []

    const updateEvaluationParam = (key: string, value: string) => {
        const next = new URLSearchParams(searchParams)
        next.set('tab', 'evals')
        if (value) {
            next.set(key, value)
        } else {
            next.delete(key)
        }
        next.delete('evalOffset')
        setSearchParams(next)
    }

    const updateEvaluationOffset = (nextOffset: number) => {
        const next = new URLSearchParams(searchParams)
        next.set('tab', 'evals')
        if (nextOffset > 0) {
            next.set('evalOffset', String(nextOffset))
        } else {
            next.delete('evalOffset')
        }
        setSearchParams(next)
    }

    useEffect(() => {
        if (launchWizard && !hasAutoOpened) {
            setIsWizardOpen(true)
            setHasAutoOpened(true)
        }
    }, [launchWizard, hasAutoOpened])

    if (isLoading) {
        return (
            <div className="flex justify-center py-20">
                <Loader2 className="h-8 w-8 animate-spin text-primary/50" />
            </div>
        )
    }


    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-xl font-semibold">Evaluations</h2>
                    <p className="text-sm text-muted-foreground">Monitor RAG performance over time.</p>
                </div>
                <button
                    onClick={() => setIsWizardOpen(true)}
                    className="flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-all shadow-md"
                >
                    <Play className="h-4 w-4" />
                    Launch Eval
                </button>
            </div>

            <div className="grid gap-3 rounded-xl border border-border bg-card p-4 md:grid-cols-4">
                <select
                    value={testSetFilter}
                    onChange={(event) => updateEvaluationParam('evalTestSet', event.target.value)}
                    className="h-10 rounded-lg border border-input bg-background px-3 text-sm"
                >
                    <option value="">All test sets</option>
                    {testSets.map((testSet) => (
                        <option key={testSet.id} value={testSet.id}>{testSet.name}</option>
                    ))}
                </select>
                <select
                    value={ragConfigFilter}
                    onChange={(event) => updateEvaluationParam('evalRagConfig', event.target.value)}
                    className="h-10 rounded-lg border border-input bg-background px-3 text-sm"
                >
                    <option value="">All RAG configs</option>
                    {ragConfigs.map((config) => (
                        <option key={config.id} value={config.id}>{config.name}</option>
                    ))}
                </select>
                <select
                    value={indexFilter}
                    onChange={(event) => updateEvaluationParam('evalIndex', event.target.value)}
                    className="h-10 rounded-lg border border-input bg-background px-3 text-sm"
                >
                    <option value="">All indexes</option>
                    {indexes.map((index) => (
                        <option key={index.id} value={index.id}>{index.name}</option>
                    ))}
                </select>
                <select
                    value={statusFilter}
                    onChange={(event) => updateEvaluationParam('evalStatus', event.target.value)}
                    className="h-10 rounded-lg border border-input bg-background px-3 text-sm"
                >
                    <option value="">All statuses</option>
                    <option value="pending">Pending</option>
                    <option value="running">Running</option>
                    <option value="completed">Completed</option>
                    <option value="failed">Failed</option>
                    <option value="cancelled">Cancelled</option>
                    <option value="paused">Paused</option>
                </select>
            </div>

            {evaluations.length === 0 ? (
                <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-border py-20 bg-card/50">
                    <div className="rounded-full bg-primary/10 p-5 text-primary">
                        <FlaskConical className="h-10 w-10" />
                    </div>
                    <h3 className="mt-5 text-xl font-semibold">{hasFilters ? 'No matching evaluations' : 'No evaluations yet'}</h3>
                    <p className="mt-2 text-center text-muted-foreground max-w-sm">
                        {hasFilters
                            ? 'Clear filters or adjust the selected context to see more evaluations.'
                            : 'Start your first RAG evaluation to measure performance across different metrics.'}
                    </p>
                    {!hasFilters && (
                        <button
                            onClick={() => setIsWizardOpen(true)}
                            className="mt-6 flex items-center gap-2 rounded-lg bg-primary px-6 py-2.5 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-all shadow-md"
                        >
                            <Play className="h-4 w-4" />
                            Start First Evaluation
                        </button>
                    )}
                </div>
            ) : (
                <div className="overflow-hidden rounded-xl border border-border bg-card">
                    <div className="divide-y divide-border">
                        {evaluations.map((evalItem: Evaluation) => (
                            <div
                                key={evalItem.id}
                                className="group relative flex cursor-pointer flex-col gap-4 p-4 transition-all hover:bg-accent/40 lg:flex-row lg:items-center lg:justify-between"
                                onClick={() => navigate(`/projects/${projectId}/evaluations/${evalItem.id}`)}
                            >
                                <div className="flex items-start gap-4">
                                    <div className={cn(
                                        "flex h-10 w-10 shrink-0 items-center justify-center rounded-lg",
                                        evalItem.status === 'completed' ? "bg-green-500/10 text-green-600" :
                                            evalItem.status === 'failed' ? "bg-red-500/10 text-red-600" :
                                                evalItem.status === 'running' ? "bg-blue-500/10 text-blue-600 animate-pulse" :
                                                    "bg-muted text-muted-foreground"
                                    )}>
                                        <FlaskConical className="h-5 w-5" />
                                    </div>
                                    <div className="min-w-0">
                                        <div className="flex flex-wrap items-center gap-2">
                                            <p className="font-bold">{evalItem.name || `Evaluation #${evalItem.id.slice(0, 8)}`}</p>
                                            <span className={cn(
                                                "rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider",
                                                evalItem.status === 'completed' ? "bg-green-500/10 text-green-600" :
                                                    evalItem.status === 'failed' ? "bg-red-500/10 text-red-600" :
                                                        evalItem.status === 'running' ? "bg-blue-500/10 text-blue-600" :
                                                            "bg-muted text-muted-foreground"
                                            )}>
                                                {evalItem.status}
                                            </span>
                                            {evalItem.is_baseline && (
                                                <span className="rounded-full bg-primary/10 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-primary">
                                                    Baseline
                                                </span>
                                            )}
                                        </div>
                                        <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                                            <span className="flex items-center gap-1">
                                                <Calendar className="h-3 w-3" />
                                                {new Date(evalItem.created_at).toLocaleString()}
                                            </span>
                                            <span>{evalItem.result_count} results</span>
                                            {evalItem.test_set_name && <span>Test: {evalItem.test_set_name}</span>}
                                            {evalItem.index_name && <span>Index: {evalItem.index_name}</span>}
                                            {evalItem.rag_type && <span>RAG: {evalItem.rag_type}</span>}
                                        </div>
                                    </div>
                                </div>

                                <div className="flex items-center justify-between gap-6 lg:justify-end">
                                    <div className="flex items-center gap-6">
                                        {evalItem.pass_rate !== null && (
                                            <div className="text-right">
                                                <p className="text-[10px] font-bold uppercase text-muted-foreground">Pass Rate</p>
                                                <p className={cn(
                                                    "text-lg font-black",
                                                    evalItem.pass_rate >= 0.7 ? "text-green-500" :
                                                        evalItem.pass_rate >= 0.4 ? "text-amber-500" : "text-red-500"
                                                )}>
                                                    {(evalItem.pass_rate * 100).toFixed(0)}%
                                                </p>
                                            </div>
                                        )}
                                        {evalItem.summary_metrics?.overall_avg !== undefined && (
                                            <div className="text-right">
                                                <p className="text-[10px] font-bold uppercase text-muted-foreground">Avg Score</p>
                                                <p className="text-lg font-black text-primary">
                                                    {evalItem.summary_metrics.overall_avg.toFixed(2)}
                                                </p>
                                            </div>
                                        )}
                                    </div>
                                    <div className="rounded-full bg-muted/50 p-2 transition-all group-hover:bg-primary group-hover:text-primary-foreground">
                                        <ChevronRight className="h-4 w-4" />
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                    <PaginationFooter
                        total={total}
                        offset={data?.data?.offset ?? offset}
                        limit={data?.data?.limit ?? pageSize}
                        onPageChange={updateEvaluationOffset}
                        isLoading={isLoading}
                    />
                </div>
            )}

            <StartEvaluationWizard
                projectId={projectId}
                isOpen={isWizardOpen}
                onClose={() => setIsWizardOpen(false)}
                onStarted={(id) => {
                    queryClient.invalidateQueries({ queryKey: ['evaluations', projectId] })
                    navigate(`/projects/${projectId}/evaluations/${id}`)
                }}
                initialKnowledgeBaseId={initialKnowledgeBaseId}
                initialIndexId={initialIndexId}
            />
        </div>
    )
}

function TrendsTab({ projectId }: { projectId: string }) {
    const [activeTrendView, setActiveTrendView] = useState<'metrics' | 'efficiency'>('metrics')
    const { data, isLoading, isError } = useQuery({
        queryKey: ['project-trends', projectId],
        queryFn: () => api.trends.getProjectTrends(projectId),
        enabled: !!projectId,
    })

    if (isLoading) {
        return (
            <div className="flex justify-center py-20">
                <Loader2 className="h-8 w-8 animate-spin text-primary/50" />
            </div>
        )
    }

    if (isError || !data?.data) {
        return (
            <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-border py-20 bg-card/50">
                <AlertCircle className="h-10 w-10 text-destructive/50" />
                <h3 className="mt-5 text-xl font-semibold">Trend Analysis Error</h3>
                <p className="mt-2 text-center text-muted-foreground max-w-sm">
                    Could not load trend data for this project.
                </p>
            </div>
        )
    }

    return (
        <div className="space-y-6">
            <div>
                <h2 className="text-xl font-semibold">Trend Analysis</h2>
                <p className="text-sm text-muted-foreground">Visualize RAG performance improvements over time.</p>
            </div>
            <div className="flex gap-8 border-b border-border">
                <button
                    onClick={() => setActiveTrendView('metrics')}
                    className={cn(
                        "pb-4 text-sm font-bold border-b-2 transition-all",
                        activeTrendView === 'metrics'
                            ? "border-primary text-primary"
                            : "border-transparent text-muted-foreground hover:text-foreground"
                    )}
                >
                    Metric Trends
                </button>
                <button
                    onClick={() => setActiveTrendView('efficiency')}
                    className={cn(
                        "pb-4 text-sm font-bold border-b-2 transition-all",
                        activeTrendView === 'efficiency'
                            ? "border-primary text-primary"
                            : "border-transparent text-muted-foreground hover:text-foreground"
                    )}
                >
                    Efficiency Map
                </button>
            </div>
            {activeTrendView === 'metrics' ? (
                <TrendChart trends={data.data} />
            ) : (
                <EfficiencyMap trends={data.data} />
            )}
        </div>
    )
}

const tabs = [
    { id: 'overview', name: 'Overview', icon: CheckCircle2 },
    { id: 'kb', name: 'Knowledge Bases', icon: Database },
    { id: 'rags', name: 'RAG Configs', icon: Settings2 },
    { id: 'indexes', name: 'Indexes', icon: Layers },
    { id: 'tests', name: 'Test Sets', icon: FileText },
    { id: 'evals', name: 'Evaluations', icon: FlaskConical },
    { id: 'compare', name: 'Comparisons', icon: GitCompare },
    { id: 'trends', name: 'Trends', icon: TrendingUp },
]

export function ProjectDetail() {
    const { id } = useParams<{ id: string }>()
    const navigate = useNavigate()
    const [searchParams, setSearchParams] = useSearchParams()
    const [isEditOpen, setIsEditOpen] = useState(false)
    const queryClient = useQueryClient()
    const { success, error } = useToast()

    const { data: project, isLoading, isError } = useQuery({
        queryKey: ['project', id],
        queryFn: () => api.projects.get(id!),
        enabled: !!id,
    })

    const tabParam = searchParams.get('tab')
    const shouldLaunchEval = searchParams.get('startEval') === '1'
    const initialKnowledgeBaseId = searchParams.get('kbId') || undefined
    const initialIndexId = searchParams.get('indexId') || undefined
    const activeTab = tabParam && tabs.some(tab => tab.id === tabParam) ? tabParam : 'overview'

    const selectTab = (tabId: string) => {
        const next = new URLSearchParams(searchParams)
        next.set('tab', tabId)
        if (tabId !== 'evals') {
            next.delete('startEval')
            next.delete('kbId')
            next.delete('indexId')
        }
        setSearchParams(next)
    }

    const updateMutation = useMutation({
        mutationFn: (data: Parameters<typeof api.projects.update>[1]) => {
            if (!id) {
                return Promise.reject(new Error('Project id is missing'))
            }
            return api.projects.update(id, data)
        },
        onSuccess: (response) => {
            queryClient.invalidateQueries({ queryKey: ['project', id] })
            queryClient.invalidateQueries({ queryKey: ['projects'] })
            success('Project updated', `"${response.data.name}" has been updated.`)
            setIsEditOpen(false)
        },
        onError: () => {
            error('Failed to update project', 'Please try again.')
        },
    })

    const archiveMutation = useMutation({
        mutationFn: () => {
            if (!id) {
                return Promise.reject(new Error('Project id is missing'))
            }
            return api.projects.archive(id)
        },
        onSuccess: (response) => {
            queryClient.invalidateQueries({ queryKey: ['project', id] })
            queryClient.invalidateQueries({ queryKey: ['projects'] })
            success('Project archived', `"${response.data.name}" has been archived.`)
        },
        onError: () => {
            error('Failed to archive project', 'Please try again.')
        },
    })

    const handleArchive = () => {
        if (!confirm('Archive this project? It will be hidden from active project lists.')) return
        archiveMutation.mutate()
    }

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
                    <button
                        onClick={() => setIsEditOpen(true)}
                        className="rounded-lg border border-border bg-card px-4 py-2 text-sm font-medium hover:bg-accent transition-colors"
                    >
                        Edit Project
                    </button>
                    <button
                        onClick={handleArchive}
                        disabled={archiveMutation.isPending || p.status === 'archived'}
                        className="rounded-lg bg-destructive/10 text-destructive border border-destructive/20 px-4 py-2 text-sm font-medium hover:bg-destructive/20 transition-colors disabled:cursor-not-allowed disabled:opacity-50"
                    >
                        {p.status === 'archived' ? 'Archived' : archiveMutation.isPending ? 'Archiving...' : 'Archive'}
                    </button>
                </div>
            </div>

            {/* Tabs */}
            <div className="border-b border-border">
                <nav className="flex gap-8">
                    {tabs.map((tab) => (
                        <button
                            key={tab.id}
                            onClick={() => selectTab(tab.id)}
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
                {activeTab === 'overview' && <ProjectOverviewTab project={p} onSelectTab={selectTab} />}
                {activeTab === 'kb' && <KnowledgeBasesTab projectId={p.id} />}
                {activeTab === 'tests' && <TestSetsTab projectId={p.id} />}
                {activeTab === 'rags' && <RAGConfigsTab projectId={p.id} />}
                {activeTab === 'indexes' && <ProjectIndexesTab projectId={p.id} />}
                {activeTab === 'evals' && (
                    <EvaluationsTab
                        projectId={p.id}
                        launchWizard={shouldLaunchEval}
                        initialKnowledgeBaseId={initialKnowledgeBaseId}
                        initialIndexId={initialIndexId}
                    />
                )}
                {activeTab === 'compare' && <ComparisonsTab projectId={p.id} />}
                {activeTab === 'trends' && <TrendsTab projectId={p.id} />}
            </div>

            <EditProjectDialog
                isOpen={isEditOpen}
                project={p}
                onClose={() => setIsEditOpen(false)}
                onSubmit={async (data) => {
                    await updateMutation.mutateAsync(data)
                }}
            />
        </div>
    )
}
