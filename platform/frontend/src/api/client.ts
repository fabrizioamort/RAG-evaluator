import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export const apiClient = axios.create({
  baseURL: `${API_BASE_URL}/api/v1`,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor for adding request ID
apiClient.interceptors.request.use((config) => {
  config.headers['X-Request-ID'] = crypto.randomUUID()
  return config
})

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      // Server responded with error status
      const message = error.response.data?.detail || 'An error occurred'
      console.error('API Error:', message)
    } else if (error.request) {
      // Request made but no response
      console.error('Network Error: No response received')
    } else {
      // Error in request setup
      console.error('Request Error:', error.message)
    }
    return Promise.reject(error)
  }
)

// API types
export interface HealthResponse {
  status: string
  database: string
  version: string
}

export interface Project {
  id: string
  name: string
  description: string | null
  status: 'active' | 'archived'
  tags: string[]
  created_at: string
  updated_at: string
  knowledge_base_count: number
  test_set_count: number
  rag_config_count: number
  evaluation_count: number
}

export interface ProjectCreate {
  name: string
  description?: string
  tags?: string[]
}

export interface ProjectUpdate {
  name?: string
  description?: string
  status?: 'active' | 'archived'
  tags?: string[]
}

export interface PaginatedList<T> {
  items: T[]
  total: number
  offset: number
  limit: number
}

export interface Document {
  id: string
  knowledge_base_id: string
  filename: string
  file_path: string
  content_type: string
  size_bytes: number
  checksum: string
  status: 'uploaded' | 'processed' | 'failed'
  created_at: string
}

export interface KnowledgeBase {
  id: string
  project_id: string
  name: string
  description: string | null
  metadata: Record<string, any>
  status: 'pending' | 'ready' | 'indexing' | 'error'
  current_version: number
  storage_path: string
  index_path: string
  document_count: number
  created_at: string
  documents?: Document[]
}

export interface KnowledgeBaseCreate {
  name: string
  description?: string
  metadata?: Record<string, any>
}

export interface KnowledgeBaseUpdate {
  name?: string
  description?: string
  metadata?: Record<string, any>
}

export interface DocumentUploadResponse {
  uploaded: Document[]
  failed: { filename: string; error: string }[]
  total_size_bytes: number
}

// API functions
export const api = {
  health: {
    check: () => apiClient.get<HealthResponse>('/health'),
    detail: () => apiClient.get('/health/detail'),
  },
  projects: {
    list: (params?: { limit?: number; offset?: number; status?: string }) =>
      apiClient.get<PaginatedList<Project>>('/projects', { params }),
    get: (id: string) => apiClient.get<Project>(`/projects/${id}`),
    create: (data: ProjectCreate) => apiClient.post<Project>('/projects', data),
    update: (id: string, data: ProjectUpdate) => apiClient.put<Project>(`/projects/${id}`, data),
    delete: (id: string) => apiClient.delete(`/projects/${id}`),
    archive: (id: string) => apiClient.post<Project>(`/projects/${id}/archive`),
  },
  knowledgeBases: {
    list: (projectId: string, params?: { limit?: number; offset?: number }) =>
      apiClient.get<PaginatedList<KnowledgeBase>>(`/projects/${projectId}/knowledge-bases`, { params }),
    get: (id: string) => apiClient.get<KnowledgeBase>(`/knowledge-bases/${id}`),
    create: (projectId: string, data: KnowledgeBaseCreate) =>
      apiClient.post<KnowledgeBase>(`/projects/${projectId}/knowledge-bases`, data),
    update: (id: string, data: KnowledgeBaseUpdate) =>
      apiClient.put<KnowledgeBase>(`/knowledge-bases/${id}`, data),
    delete: (id: string) => apiClient.delete(`/knowledge-bases/${id}`),
    uploadDocuments: (id: string, files: File[]) => {
      const formData = new FormData()
      files.forEach((file) => formData.append('files', file))
      return apiClient.post<DocumentUploadResponse>(`/knowledge-bases/${id}/documents`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
    },
    deleteDocument: (kbId: string, docId: string) =>
      apiClient.delete(`/knowledge-bases/${kbId}/documents/${docId}`),
    getStatus: (id: string) => apiClient.get(`/knowledge-bases/${id}/status`),
  },
}
