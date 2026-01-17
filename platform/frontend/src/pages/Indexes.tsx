import { useState, useEffect, useCallback } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api, KnowledgeBaseIndex } from '../api/client'
import { IndexCard } from '../components/indexes/IndexCard'
import { Search, Filter, Loader2 } from 'lucide-react'

export function Indexes() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [indexes, setIndexes] = useState<KnowledgeBaseIndex[]>([])
  const [loading, setLoading] = useState(true)

  const statusFilter = searchParams.get('status') || ''
  const search = searchParams.get('search') || '' // Not supported by API yet

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
    if (status) {
      setSearchParams({ status })
    } else {
      setSearchParams({})
    }
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
            onChange={() => {
              // client side filtering or update query
            }}
            disabled // Disabled for now as API doesn't support search text
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
      ) : indexes.length === 0 ? (
        <div className="text-center py-12 bg-gray-50 rounded-lg border border-dashed">
          <p className="text-gray-500">No indexes found.</p>
          <p className="text-sm text-gray-400 mt-1">Go to a Knowledge Base to create an index.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {indexes.map(index => (
            <IndexCard
              key={index.id}
              index={index}
              onDelete={fetchIndexes}
              onRunEvaluation={() => {
                // Navigate to new eval with this index pre-selected?
                // Or open dialog?
                // For now just console log
                console.log('Run eval for', index.id)
              }}
            />
          ))}
        </div>
      )}
    </div>
  )
}
