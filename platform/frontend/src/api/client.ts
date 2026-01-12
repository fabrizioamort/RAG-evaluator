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

// API functions
export const api = {
  health: {
    check: () => apiClient.get<HealthResponse>('/health'),
    detail: () => apiClient.get('/health/detail'),
  },
}
