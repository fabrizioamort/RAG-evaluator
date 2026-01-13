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

export interface TestCase {
  id: string
  test_set_id: string
  template_id: string | null
  question: string
  expected_answer: string
  ground_truth_context: string[]
  difficulty: 'easy' | 'medium' | 'hard'
  category: string | null
  question_type: 'factual' | 'reasoning' | 'comparison' | 'multi_hop'
  is_generated: boolean
  is_reviewed: boolean
  quality_score: number | null
  provenance_artifact_id: string | null
  created_at: string
}

export interface TestSet {
  id: string
  project_id: string
  name: string
  description: string | null
  tags: string[]
  test_case_count: number
  created_at: string
  test_cases?: TestCase[]
}

export interface TestSetCreate {
  name: string
  description?: string
  tags?: string[]
}

export interface TestCaseCreate {
  question: string
  expected_answer: string
  ground_truth_context?: string[]
  difficulty?: 'easy' | 'medium' | 'hard'
  category?: string
  question_type?: 'factual' | 'reasoning' | 'comparison' | 'multi_hop'
}

export interface RAGConfig {
  id: string
  project_id: string
  name: string
  rag_type: string
  parameters: Record<string, any>
  llm_provider: string
  llm_model: string
  llm_base_url: string | null
  created_at: string
}

export interface RAGConfigCreate {
  name: string
  rag_type: string
  parameters?: Record<string, any>
  llm_provider?: string
  llm_model?: string
  llm_base_url?: string
}

export interface RAGTypeParameter {
  name: string
  type: 'string' | 'integer' | 'float' | 'boolean'
  description: string
  required: boolean
  default: any
  min_value?: number
  max_value?: number
  choices?: string[]
}

export interface RAGTypeInfo {
  name: string
  display_name: string
  description: string
  parameters: RAGTypeParameter[]
  requires_index: boolean
}

export interface LLMProviderInfo {
  name: string
  display_name: string
  models: string[]
  requires_api_key: boolean
  supports_base_url: boolean
}

export interface SummaryMetrics {
  faithfulness_avg?: number
  relevancy_avg?: number
  precision_avg?: number
  recall_avg?: number
  overall_avg?: number
}

export interface Evaluation {
  id: string
  project_id: string
  knowledge_base_id: string | null
  test_set_id: string | null
  rag_config_id: string | null
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | 'paused'
  started_at: string | null
  completed_at: string | null
  summary_metrics: SummaryMetrics | null
  pass_rate: number | null
  error_message: string | null
  result_count: number
  created_at: string
}

export interface EvaluationCreate {
  knowledge_base_id: string
  test_set_id: string
  rag_config_id: string
  notes?: string
  tags?: string[]
}

export interface EvaluationResult {
  id: string
  evaluation_id: string
  test_case_id: string | null
  generated_answer: string | null
  faithfulness_score: number | null
  faithfulness_reason: string | null
  relevancy_score: number | null
  relevancy_reason: string | null
  precision_score: number | null
  precision_reason: string | null
  recall_score: number | null
  recall_reason: string | null
  latency_seconds: number | null
  cost_usd: string | null
  prompt_tokens: number | null
  completion_tokens: number | null
  created_at: string
  // Artifacts
  retrieved_context_artifact_id?: string | null
  retrieval_trace_artifact_id?: string | null
  raw_metrics_artifact_id?: string | null
  // Augmented from joined test case
  question?: string
  expected_answer?: string
  difficulty?: string
  category?: string
}

export interface ProgressEvent {
  event_type: 'started' | 'progress' | 'completed' | 'error' | 'paused' | 'resumed'
  evaluation_id: string
  timestamp: string
  total_test_cases?: number
  completed?: number
  total?: number
  current_question?: string
  last_result?: EvaluationResult
  summary_metrics?: SummaryMetrics
  pass_rate?: number
  error_message?: string
  resuming_from?: number
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
  testSets: {
    list: (projectId: string, params?: { limit?: number; offset?: number }) =>
      apiClient.get<PaginatedList<TestSet>>(`/projects/${projectId}/test-sets`, { params }),
    get: (id: string) => apiClient.get<TestSet>(`/test-sets/${id}`),
    create: (projectId: string, data: TestSetCreate) =>
      apiClient.post<TestSet>(`/projects/${projectId}/test-sets`, data),
    update: (id: string, data: Partial<TestSetCreate>) =>
      apiClient.put<TestSet>(`/test-sets/${id}`, data),
    delete: (id: string) => apiClient.delete(`/test-sets/${id}`),
    import: (id: string, data: any) => apiClient.post(`/test-sets/${id}/import`, data),
    export: (id: string) => apiClient.get(`/test-sets/${id}/export`, { responseType: 'blob' }),
    addCase: (testSetId: string, data: TestCaseCreate) =>
      apiClient.post<TestCase>(`/test-sets/${testSetId}/cases`, data),
    updateCase: (testSetId: string, caseId: string, data: Partial<TestCaseCreate>) =>
      apiClient.put<TestCase>(`/test-sets/${testSetId}/cases/${caseId}`, data),
    deleteCase: (testSetId: string, caseId: string) =>
      apiClient.delete(`/test-sets/${testSetId}/cases/${caseId}`),
  },
  ragConfigs: {
    list: (projectId: string, params?: { limit?: number; offset?: number }) =>
      apiClient.get<PaginatedList<RAGConfig>>(`/projects/${projectId}/rag-configs`, { params }),
    get: (id: string) => apiClient.get<RAGConfig>(`/rag-configs/${id}`),
    create: (projectId: string, data: RAGConfigCreate) =>
      apiClient.post<RAGConfig>(`/projects/${projectId}/rag-configs`, data),
    update: (id: string, data: Partial<RAGConfigCreate>) =>
      apiClient.put<RAGConfig>(`/rag-configs/${id}`, data),
    delete: (id: string) => apiClient.delete(`/rag-configs/${id}`),
    getTypes: () => apiClient.get<RAGTypeInfo[]>('/rag-types'),
    getParameters: (type: string) => apiClient.get<RAGTypeParameter[]>(`/rag-types/${type}/parameters`),
    getLLMProviders: () => apiClient.get<LLMProviderInfo[]>('/llm-providers'),
  },
  evaluations: {
    list: (projectId: string, params?: { limit?: number; offset?: number }) =>
      apiClient.get<PaginatedList<Evaluation>>(`/projects/${projectId}/evaluations`, { params }),
    get: (id: string) => apiClient.get<Evaluation>(`/evaluations/${id}`),
    create: (data: EvaluationCreate) => apiClient.post<Evaluation>('/evaluations', data),
    getResults: (id: string, params?: { limit?: number; offset?: number }) =>
      apiClient.get<PaginatedList<EvaluationResult>>(`/evaluations/${id}/results`, { params }),
    cancel: (id: string) => apiClient.post(`/evaluations/${id}/cancel`),
    pause: (id: string) => apiClient.post(`/evaluations/${id}/pause`),
    resume: (id: string) => apiClient.post(`/evaluations/${id}/resume`),
    retry: (id: string) => apiClient.post<Evaluation>(`/evaluations/${id}/retry`),
    getStreamUrl: (id: string) => `${API_BASE_URL}/api/v1/evaluations/${id}/stream`,
  },
}
