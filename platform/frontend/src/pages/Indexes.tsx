import { useNavigate, useSearchParams } from 'react-router-dom'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { api, KnowledgeBaseIndex } from '../api/client'
import { IndexCard } from '../components/indexes/IndexCard'
import { AlertCircle, Filter, Loader2, Search } from 'lucide-react'
import { PaginationFooter } from '@/components/ui/PaginationFooter'

export function Indexes() {
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const queryClient = useQueryClient()
  const pageSize = 20

  const statusFilter = searchParams.get('status') || ''
  const projectFilter = searchParams.get('project') || ''
  const search = searchParams.get('search') || ''
  const offsetParam = Number(searchParams.get('offset') || '0')
  const offset = Number.isFinite(offsetParam) && offsetParam > 0 ? offsetParam : 0

  const indexesQuery = useQuery({
    queryKey: ['indexes', 'global', statusFilter, projectFilter, offset],
    queryFn: () =>
      api.indexes.list({
        status: statusFilter || undefined,
        project_id: projectFilter || undefined,
        limit: pageSize,
        offset,
      }),
  })

  const projectsQuery = useQuery({
    queryKey: ['projects', 'index-filter-options'],
    queryFn: () => api.projects.list({ limit: 100 }),
  })

  const updateParam = (key: string, value: string) => {
    const next = new URLSearchParams(searchParams)
    if (value) next.set(key, value)
    else next.delete(key)
    next.delete('offset')
    setSearchParams(next)
  }

  const handleSearchChange = (value: string) => {
    updateParam('search', value.trim())
  }

  const indexes = indexesQuery.data?.data.items ?? []
  const filteredIndexes = indexes.filter((index) => {
    const query = search.trim().toLowerCase()
    if (!query) return true

    return [
      index.name,
      index.project_name,
      index.knowledge_base_name,
      index.rag_config_name,
      index.status,
    ].some((value) => value?.toLowerCase().includes(query))
  })

  const handlePageChange = (nextOffset: number) => {
    const next = new URLSearchParams(searchParams)
    if (nextOffset > 0) next.set('offset', String(nextOffset))
    else next.delete('offset')
    setSearchParams(next)
  }

  const runEvaluationFromIndex = (index: KnowledgeBaseIndex) => {
    if (!index.project_id) return

    const params = new URLSearchParams({
      tab: 'evals',
      startEval: '1',
      kbId: index.knowledge_base_id,
      indexId: index.id,
    })
    navigate(`/projects/${index.project_id}?${params.toString()}`)
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Indexes</h1>
          <p className="text-muted-foreground mt-1">Monitor index build status across projects.</p>
        </div>
      </div>

      <div className="rounded-xl border border-border bg-card p-4 grid gap-4 lg:grid-cols-[1fr_14rem_14rem]">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search indexes..."
            className="h-10 w-full rounded-lg border border-input bg-background pl-9 pr-4 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
            value={search}
            onChange={(event) => handleSearchChange(event.target.value)}
          />
        </div>
        <div className="relative">
          <Filter className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <select
            value={projectFilter}
            onChange={(e) => updateParam('project', e.target.value)}
            className="h-10 w-full appearance-none rounded-lg border border-input bg-background pl-9 pr-4 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            <option value="">All projects</option>
            {(projectsQuery.data?.data.items ?? []).map((project) => (
              <option key={project.id} value={project.id}>{project.name}</option>
            ))}
          </select>
        </div>
        <div className="relative">
          <Filter className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <select
            value={statusFilter}
            onChange={(e) => updateParam('status', e.target.value)}
            className="h-10 w-full appearance-none rounded-lg border border-input bg-background pl-9 pr-4 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            <option value="">All Statuses</option>
            <option value="ready">Ready</option>
            <option value="building">Building</option>
            <option value="failed">Failed</option>
            <option value="pending">Pending</option>
          </select>
        </div>
      </div>

      {indexesQuery.isLoading ? (
        <div className="flex justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </div>
      ) : indexesQuery.isError ? (
        <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-border bg-card/50 py-12 text-center">
          <AlertCircle className="h-10 w-10 text-destructive" />
          <p className="mt-3 font-semibold text-foreground">Failed to load indexes</p>
          <p className="mt-1 text-sm text-muted-foreground">Check the API connection and try again.</p>
        </div>
      ) : filteredIndexes.length === 0 ? (
        <div className="rounded-xl border border-dashed border-border bg-card/50 py-12 text-center">
          <p className="text-muted-foreground">No indexes found.</p>
          <p className="text-sm text-muted-foreground/80 mt-1">
            {search ? 'Clear search or filters to see more indexes.' : 'Go to a Knowledge Base to create an index.'}
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredIndexes.map(index => (
              <IndexCard
                key={index.id}
                index={index}
                onDelete={() => queryClient.invalidateQueries({ queryKey: ['indexes'] })}
                onRunEvaluation={index.project_id ? () => runEvaluationFromIndex(index) : undefined}
              />
            ))}
          </div>
          <div className="overflow-hidden rounded-xl border border-border bg-card">
            <PaginationFooter
              total={indexesQuery.data?.data.total ?? 0}
              offset={indexesQuery.data?.data.offset ?? offset}
              limit={indexesQuery.data?.data.limit ?? pageSize}
              onPageChange={handlePageChange}
              isLoading={indexesQuery.isFetching}
            />
          </div>
        </div>
      )}
    </div>
  )
}
