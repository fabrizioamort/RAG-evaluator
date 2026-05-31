import { useState, useEffect, useCallback } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { api, KnowledgeBaseIndex } from '../api/client'
import { IndexCard } from '../components/indexes/IndexCard'
import { Search, Filter, Loader2 } from 'lucide-react'

export function Indexes() {
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const [indexes, setIndexes] = useState<KnowledgeBaseIndex[]>([])
  const [loading, setLoading] = useState(true)

  const statusFilter = searchParams.get('status') || ''
  const search = searchParams.get('search') || ''

  const fetchIndexes = useCallback(async () => {
    setLoading(true)
    try {
      const response = await api.indexes.list({
        status: statusFilter || undefined,
        limit: 50
      })
      setIndexes(response.data.items)
    } catch (err) {
      console.error('Failed to fetch indexes', err)
    } finally {
      setLoading(false)
    }
  }, [statusFilter])

  useEffect(() => {
    fetchIndexes()
  }, [fetchIndexes])

  const handleFilterChange = (status: string) => {
    const next = new URLSearchParams(searchParams)
    if (status) next.set('status', status)
    else next.delete('status')
    setSearchParams(next)
  }

  const handleSearchChange = (value: string) => {
    const next = new URLSearchParams(searchParams)
    if (value.trim()) next.set('search', value)
    else next.delete('search')
    setSearchParams(next)
  }

  const filteredIndexes = indexes.filter((index) => {
    const query = search.trim().toLowerCase()
    if (!query) return true

    return [
      index.name,
      index.knowledge_base_name,
      index.rag_config_name,
      index.status,
    ].some((value) => value?.toLowerCase().includes(query))
  })

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
          <h1 className="text-2xl font-bold text-gray-900">Indexes</h1>
          <p className="text-gray-500 mt-1">Manage your knowledge base indexes and their build status</p>
        </div>
      </div>

      <div className="bg-white p-4 rounded-lg border shadow-sm flex gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search indexes..."
            className="w-full pl-9 pr-4 py-2 border rounded-md focus:ring-2 focus:ring-blue-500 outline-none"
            value={search}
            onChange={(event) => handleSearchChange(event.target.value)}
          />
        </div>
        <div className="relative w-48">
          <Filter className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
          <select
            value={statusFilter}
            onChange={(e) => handleFilterChange(e.target.value)}
            className="w-full pl-9 pr-4 py-2 border rounded-md focus:ring-2 focus:ring-blue-500 outline-none appearance-none bg-white"
          >
            <option value="">All Statuses</option>
            <option value="ready">Ready</option>
            <option value="building">Building</option>
            <option value="failed">Failed</option>
            <option value="pending">Pending</option>
          </select>
        </div>
      </div>

      {loading ? (
        <div className="flex justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
        </div>
      ) : filteredIndexes.length === 0 ? (
        <div className="text-center py-12 bg-gray-50 rounded-lg border border-dashed">
          <p className="text-gray-500">No indexes found.</p>
          <p className="text-sm text-gray-400 mt-1">
            {search ? 'Clear search or filters to see more indexes.' : 'Go to a Knowledge Base to create an index.'}
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredIndexes.map(index => (
            <IndexCard
              key={index.id}
              index={index}
              onDelete={fetchIndexes}
              onRunEvaluation={index.project_id ? () => runEvaluationFromIndex(index) : undefined}
            />
          ))}
        </div>
      )}
    </div>
  )
}
