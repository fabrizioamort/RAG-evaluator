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

export interface DashboardStats {
  projects: number
  knowledge_bases: number
  test_sets: number
  evaluations: number
  completed_evaluations: number
  running_evaluations: number
}

export interface RecentActivityItem {
  id: string
  type: 'project' | 'evaluation' | 'knowledge_base' | 'test_set'
  action: string
  name: string
  timestamp: string
  metadata?: Record<string, unknown>
}

export interface RecentActivityResponse {
  items: RecentActivityItem[]
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
  metadata: Record<string, unknown>
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
  metadata?: Record<string, unknown>
}

export interface KnowledgeBaseUpdate {
  name?: string
  description?: string
  metadata?: Record<string, unknown>
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

export interface KnowledgeBaseIndex {
  id: string
  knowledge_base_id: string
  kb_version_id: string | null
  rag_config_id: string
  name: string
  description: string | null
  status: 'pending' | 'building' | 'ready' | 'failed' | 'archived'
  physical_id: string
  storage_type: string
  config_snapshot: Record<string, unknown>
  document_count: number
  chunk_count: number
  embedding_model: string | null
  build_started_at: string | null
  build_completed_at: string | null
  build_duration_seconds: number | null
  error_message: string | null
  created_at: string
  // Denormalized
  knowledge_base_name?: string
  rag_config_name?: string
  project_id?: string
}

export interface KnowledgeBaseIndexCreate {
  rag_config_id: string
  name?: string
  description?: string
}

export interface KnowledgeBaseIndexList {
  items: KnowledgeBaseIndex[]
  total: number
  offset: number
  limit: number
}

export interface IndexArchiveRequest {
  reason?: string
}

export interface IndexRetryRequest {
  force?: boolean
}

export interface RAGConfig {
  id: string
  project_id: string
  name: string
  rag_type: string
  parameters: Record<string, unknown>
  llm_provider: string
  llm_model: string
  llm_base_url: string | null
  llm_reasoning_effort: string | null
  embedding_model: string
  embedding_provider: string
  embedding_base_url: string | null
  created_at: string
}

export interface RAGConfigCreate {
  name: string
  rag_type: string
  parameters?: Record<string, unknown>
  llm_provider?: string
  llm_model?: string
  llm_base_url?: string
  llm_reasoning_effort?: string | null
  embedding_model?: string
  embedding_provider?: string
  embedding_base_url?: string | null
}

export interface RAGTypeParameter {
  name: string
  type: 'string' | 'integer' | 'float' | 'boolean'
  description: string
  phase: 'build' | 'query'
  required: boolean
  default: unknown
  min_value?: number
  max_value?: number
  choices?: string[]
  platform_managed?: boolean
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
  model_capabilities: Record<string, { supports_reasoning_effort: boolean }>
  requires_api_key: boolean
  supports_base_url: boolean
  supports_embeddings: boolean
}

export interface SummaryMetrics {
  faithfulness_avg?: number
  relevancy_avg?: number
  precision_avg?: number
  recall_avg?: number
  g_eval_avg?: number
  overall_avg?: number
}

export interface Evaluation {
  id: string
  name?: string
  project_id: string
  knowledge_base_id: string | null
  knowledge_base_index_id: string | null
  test_set_id: string | null
  rag_config_id: string | null
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | 'paused'
  started_at: string | null
  completed_at: string | null
  summary_metrics: SummaryMetrics | null
  performance_metrics: Record<string, unknown> | null
  cost_metrics: Record<string, unknown> | null
  pass_rate: number | null
  is_baseline: boolean
  baseline_reason: string | null
  error_message: string | null
  metric_config?: { metrics: string[] } | null
  query_overrides: QueryOverrides
  eval_judge_model: string | null
  eval_judge_provider: string | null
  result_count: number
  created_at: string
}

export interface QueryOverrides {
  llm_model?: string
  llm_provider?: string
  llm_base_url?: string
  llm_reasoning_effort?: string
  top_k?: number
  parameters?: Record<string, unknown>
}

export interface EvaluationCreate {
  name?: string
  knowledge_base_index_id: string
  test_set_id: string
  metric_names?: string[]
  include_reason?: boolean
  query_overrides?: QueryOverrides
  eval_judge_model?: string
  eval_judge_provider?: string
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
  g_eval_score: number | null
  g_eval_reason: string | null
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

export interface RetrievalTraceStep {
  type: string
  input: unknown
  output_refs?: string[]
  duration_ms?: number
  metadata?: Record<string, unknown>
}

export interface RetrievalTraceChunk {
  content: string
  document_id: string
  chunk_id: string
  score: number
  rank: number
  source: string
  metadata: Record<string, unknown>
}

export interface RetrievalTrace {
  strategy: string
  steps: RetrievalTraceStep[]
  retrieved_chunks: RetrievalTraceChunk[]
  fusion_details?: Record<string, unknown>
  total_duration_ms: number
}

export interface RunManifest {
  id: string
  rag_config_snapshot: Record<string, unknown>
  build_config_snapshot: Record<string, unknown>
  query_overrides: Record<string, unknown>
  effective_config_snapshot: Record<string, unknown>
  kb_version_snapshot: Record<string, unknown>
  generation_model: string | null
  eval_judge_model: string | null
  prompt_templates: Record<string, unknown>
  rag_evaluator_version: string | null
  platform_version: string | null
  created_at: string
}

export interface TrendDataPoint {
  timestamp: string
  evaluation_id: string
  metrics: Record<string, number | null>
  pass_rate: number | null
}

export interface RAGConfigTrend {
  rag_config_id: string | null
  rag_config_name: string | null
  data_points: TrendDataPoint[]
}

export interface ProjectTrends {
  project_id: string
  trends: RAGConfigTrend[]
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

// Test Template types
export interface TestTemplate {
  id: string
  name: string
  description: string | null
  category: string
  question_template: string
  answer_template: string | null
  entity_types: string[]
  complexity_level: 'easy' | 'medium' | 'hard'
  is_builtin: boolean
  created_at: string
}

// Test Generation types
export interface TestGenerationConfig {
  knowledge_base_id: string
  target_count: number
  questions_per_chunk?: number
  difficulty_distribution?: Record<string, number>
  template_ids?: string[]
  llm_model?: string
  skip_semantic_check?: boolean
}

export interface TestGenerationJob {
  id: string
  test_set_id: string
  knowledge_base_id: string | null
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  config: Record<string, unknown>
  questions_generated: number
  questions_total: number
  questions_rejected: number
  started_at: string | null
  completed_at: string | null
  error_message: string | null
  created_at: string
}

export interface TestGenerationStatus {
  job_id: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  progress: number
  questions_generated: number
  questions_total: number
  questions_rejected: number
  started_at: string | null
  completed_at: string | null
  error_message: string | null
}

export interface BulkReviewRequest {
  test_case_ids: string[]
  action: 'approve' | 'reject'
}

// Playground types
export interface PlaygroundIndexInfo {
  id: string
  name: string
  rag_type: string
  knowledge_base_id: string
  knowledge_base_name: string
  project_id: string
  project_name: string
  document_count: number
  chunk_count: number
  status: string
}

export interface PlaygroundIndexList {
  indexes: PlaygroundIndexInfo[]
}

export interface RetrievedChunkDetail {
  content: string
  document_id: string
  chunk_id: string
  score: number
  rank: number
  source: string
  metadata: Record<string, unknown>
}

export interface RetrievalTraceStepDetail {
  step_type: string
  duration_ms: number
  input_data: unknown
  output_summary: string | null
  metadata: Record<string, unknown>
}

export interface RetrievalTraceDetail {
  strategy: string
  steps: RetrievalTraceStepDetail[]
  total_duration_ms: number
  fusion_details: Record<string, unknown> | null
}

export interface RetrievedContextDetail {
  chunks: string[]
  chunk_details: RetrievedChunkDetail[]
}

export interface QueryMetrics {
  retrieval_time_ms: number
  generation_time_ms: number
  total_time_ms: number
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  cost_usd: string | null
}

export interface PlaygroundQueryResult {
  index_id: string
  index_name: string
  rag_type: string
  knowledge_base_name: string
  answer: string | null
  retrieved_context: RetrievedContextDetail | null
  trace: RetrievalTraceDetail | null
  metrics: QueryMetrics | null
  effective_config_snapshot?: Record<string, unknown> | null
  error: string | null
  success: boolean
}

export interface PlaygroundQueryResponse {
  query_id: string
  question: string
  results: PlaygroundQueryResult[]
  created_at: string
}

export interface PlaygroundQueryRequest {
  question: string
  index_ids: string[]
  top_k?: number
  query_overrides?: QueryOverrides
}

export interface PlaygroundQueryHistoryItem {
  id: string
  created_at: string
  question: string
  index_count: number
  index_names: string[]
  success_count: number
  total_time_ms: number | null
}

export interface PlaygroundQueryHistoryList {
  items: PlaygroundQueryHistoryItem[]
  total: number
  offset: number
  limit: number
}

export interface PlaygroundQueryDetail {
  id: string
  created_at: string
  question: string
  top_k: number
  query_overrides: Record<string, unknown>
  results: PlaygroundQueryResult[]
}

// Comparisons
export interface CostMetrics {
  total_cost_usd?: number | string
  total_prompt_tokens?: number
  total_completion_tokens?: number
  avg_cost_per_query?: number | string | null
}

export interface PerformanceMetrics {
  avg_latency_seconds?: number | null
  min_latency_seconds?: number | null
  max_latency_seconds?: number | null
  p95_latency_seconds?: number | null
}

export interface MetricDelta {
  baseline_value: number | null
  compared_value: number | null
  absolute_delta: number | null
  percentage_delta: number | null
  improved: boolean | null
}

export interface EvaluationComparisonResult {
  evaluation_id: string
  evaluation_name?: string | null
  rag_config_name?: string | null
  summary_metrics?: SummaryMetrics | null
  cost_metrics?: CostMetrics | null
  performance_metrics?: PerformanceMetrics | null
  pass_rate?: number | null
}

export interface PerQuestionDelta {
  test_case_id: string
  question?: string | null
  baseline_result?: Record<string, unknown> | null
  compared_results: Record<string, Record<string, unknown>>
}

export interface AggregateMetrics {
  baseline_evaluation_id: string
  baseline_evaluation_name?: string | null
  baseline_rag_config_name?: string | null
  baseline_summary?: SummaryMetrics | null
  baseline_cost?: CostMetrics | null
  baseline_performance?: PerformanceMetrics | null
  baseline_pass_rate?: number | null
  comparison_results: EvaluationComparisonResult[]
}

export interface ComparisonResponse {
  id: string
  project_id: string
  name?: string | null
  description?: string | null
  baseline_evaluation_id: string
  compared_evaluation_ids: string[]
  aggregate_metrics?: AggregateMetrics | null
  created_at: string
}

export interface ComparisonDetail extends ComparisonResponse {
  per_question_deltas?: PerQuestionDelta[] | null
}

export interface ComparisonCreate {
  name?: string
  description?: string
  baseline_evaluation_id: string
  compared_evaluation_ids: string[]
}

// API functions
export const api = {
  health: {
    check: () => apiClient.get<HealthResponse>('/health'),
    detail: () => apiClient.get('/health/detail'),
  },
  stats: {
    get: () => apiClient.get<DashboardStats>('/stats'),
    recentActivity: (limit?: number) => apiClient.get<RecentActivityResponse>('/recent-activity', { params: { limit } }),
  },
  projects: {
    list: (params?: { limit?: number; offset?: number; status?: string }) =>
      apiClient.get<PaginatedList<Project>>('/projects', { params }),
    get: (id: string) => apiClient.get<Project>(`/projects/${id}`),
    create: (data: ProjectCreate) => apiClient.post<Project>('/projects', data),
    update: (id: string, data: ProjectUpdate) => apiClient.put<Project>(`/projects/${id}`, data),
    delete: (id: string) => apiClient.delete(`/projects/${id}`),
    archive: (id: string) => apiClient.post<Project>(`/projects/${id}/archive`),
    getBaseline: (projectId: string) => apiClient.get<Evaluation>(`/projects/${projectId}/baseline`),
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
    // index method is deprecated/removed in favor of api.indexes.create
  },
  indexes: {
    list: (params?: { kb_id?: string; project_id?: string; status?: string; limit?: number; offset?: number }) =>
      apiClient.get<KnowledgeBaseIndexList>('/indexes', { params }),
    get: (id: string) => apiClient.get<KnowledgeBaseIndex>(`/indexes/${id}`),
    create: (kbId: string, data: KnowledgeBaseIndexCreate) =>
      apiClient.post<KnowledgeBaseIndex>(`/knowledge-bases/${kbId}/indexes`, data),
    delete: (id: string) => apiClient.delete(`/indexes/${id}`),
    archive: (id: string, data?: IndexArchiveRequest) =>
      apiClient.post<KnowledgeBaseIndex>(`/indexes/${id}/archive`, data),
    retry: (id: string, data?: IndexRetryRequest) =>
      apiClient.post(`/indexes/${id}/retry`, data),
    getStreamUrl: (id: string) => `${API_BASE_URL}/api/v1/indexes/${id}/stream`,
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
    import: (projectId: string, data: unknown) => apiClient.post(`/projects/${projectId}/test-sets/import`, data),
    export: (id: string) => apiClient.get(`/test-sets/${id}/export`, { responseType: 'blob' }),
    addCase: (testSetId: string, data: TestCaseCreate) =>
      apiClient.post<TestCase>(`/test-sets/${testSetId}/cases`, data),
    updateCase: (testSetId: string, caseId: string, data: Partial<TestCaseCreate>) =>
      apiClient.put<TestCase>(`/test-sets/${testSetId}/cases/${caseId}`, data),
    deleteCase: (testSetId: string, caseId: string) =>
      apiClient.delete(`/test-sets/${testSetId}/cases/${caseId}`),
    // Test Generation
    generate: (testSetId: string, config: TestGenerationConfig) =>
      apiClient.post<TestGenerationJob>(`/test-sets/${testSetId}/generate`, config),
    getGenerationStatus: (testSetId: string) =>
      apiClient.get<TestGenerationStatus>(`/test-sets/${testSetId}/generation-status`),
    cancelGeneration: (testSetId: string) =>
      apiClient.delete(`/test-sets/${testSetId}/generation`),
    listGenerationJobs: (testSetId: string) =>
      apiClient.get<TestGenerationJob[]>(`/test-sets/${testSetId}/generation-jobs`),
    // Bulk review
    bulkReview: (testSetId: string, data: BulkReviewRequest) =>
      apiClient.post(`/test-sets/${testSetId}/cases/bulk-review`, data),
  },
  testTemplates: {
    list: (params?: { limit?: number; offset?: number }) =>
      apiClient.get<PaginatedList<TestTemplate>>('/test-templates', { params }),
    get: (id: string) => apiClient.get<TestTemplate>(`/test-templates/${id}`),
    create: (data: { name: string; description?: string; category: string; question_template: string; answer_template?: string; entity_types?: string[]; complexity_level?: string }) =>
      apiClient.post<TestTemplate>('/test-templates', data),
    update: (id: string, data: Partial<{ name: string; description: string; category: string; question_template: string; answer_template: string; entity_types: string[]; complexity_level: string }>) =>
      apiClient.put<TestTemplate>(`/test-templates/${id}`, data),
    delete: (id: string) => apiClient.delete(`/test-templates/${id}`),
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
    getTrace: (evaluationId: string, resultId: string) =>
      apiClient.get<RetrievalTrace>(`/evaluations/${evaluationId}/trace/${resultId}`),
    getManifest: (evaluationId: string) =>
      apiClient.get<RunManifest>(`/evaluations/${evaluationId}/manifest`),
    getStreamUrl: (id: string) => `${API_BASE_URL}/api/v1/evaluations/${id}/stream`,
    setBaseline: (id: string, reason: string) =>
      apiClient.post<Evaluation>(`/evaluations/${id}/set-baseline`, { reason }),
  },
  trends: {
    getProjectTrends: (projectId: string) =>
      apiClient.get<ProjectTrends>(`/projects/${projectId}/trends`),
    getRagConfigTrends: (ragConfigId: string) =>
      apiClient.get<RAGConfigTrend>(`/rag-configs/${ragConfigId}/trends`),
  },
  comparisons: {
    list: (projectId: string, params?: { limit?: number; offset?: number }) =>
      apiClient.get<PaginatedList<ComparisonResponse>>(`/projects/${projectId}/comparisons`, { params }),
    get: (id: string) => apiClient.get<ComparisonDetail>(`/comparisons/${id}`),
    create: (data: ComparisonCreate) => apiClient.post<ComparisonResponse>('/comparisons', data),
    delete: (id: string) => apiClient.delete(`/comparisons/${id}`),
    listForEvaluation: (evaluationId: string, params?: { limit?: number; offset?: number }) =>
      apiClient.get<PaginatedList<ComparisonResponse>>(`/evaluations/${evaluationId}/comparisons`, { params }),
  },
  playground: {
    getIndexes: (params?: { project_id?: string; kb_id?: string }) =>
      apiClient.get<PlaygroundIndexList>('/playground/indexes', { params }),
    executeQuery: (data: PlaygroundQueryRequest) =>
      apiClient.post<PlaygroundQueryResponse>('/playground/query', data),
    getHistory: (params?: { limit?: number; offset?: number }) =>
      apiClient.get<PlaygroundQueryHistoryList>('/playground/history', { params }),
    getQueryDetail: (queryId: string) =>
      apiClient.get<PlaygroundQueryDetail>(`/playground/history/${queryId}`),
    deleteQuery: (queryId: string) =>
      apiClient.delete(`/playground/history/${queryId}`),
  },
}
