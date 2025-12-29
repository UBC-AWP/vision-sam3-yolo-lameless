import axios, { InternalAxiosRequestConfig } from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Add auth token to all requests
apiClient.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const token = localStorage.getItem('access_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error: unknown) => Promise.reject(error)
)

// Video endpoints
export const videosApi = {
  upload: async (file: File, label?: number, onProgress?: (progress: number) => void) => {
    const formData = new FormData()
    formData.append('file', file)
    if (label !== undefined && label !== null) {
      formData.append('label', String(label))
    }
    const response = await apiClient.post('/api/videos/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total)
          onProgress(progress)
        }
      },
    })
    return response.data
  },
  uploadMultiple: async (
    files: Array<{ file: File; label?: number }>,
    onFileProgress?: (index: number, progress: number) => void,
    onFileComplete?: (index: number, result: any) => void
  ) => {
    const results = []
    for (let i = 0; i < files.length; i++) {
      const { file, label } = files[i]
      try {
        const result = await videosApi.upload(file, label, (progress) => {
          onFileProgress?.(i, progress)
        })
        onFileComplete?.(i, { success: true, data: result })
        results.push({ success: true, data: result })
      } catch (error: any) {
        onFileComplete?.(i, { success: false, error: error.response?.data?.detail || 'Upload failed' })
        results.push({ success: false, error: error.response?.data?.detail || 'Upload failed' })
      }
    }
    return results
  },
  get: async (videoId: string) => {
    const response = await apiClient.get(`/api/videos/${videoId}`)
    return response.data
  },
  list: async (skip = 0, limit = 100) => {
    const response = await apiClient.get('/api/videos', { params: { skip, limit } })
    return response.data
  },
  getStreamUrl: (videoId: string) => {
    return `${API_BASE_URL}/api/videos/${videoId}/stream`
  },
  getAnnotatedUrl: (videoId: string) => {
    return `${API_BASE_URL}/api/videos/${videoId}/annotated`
  },
  getFrameUrl: (videoId: string, frameNum: number, annotated = false) => {
    return `${API_BASE_URL}/api/videos/${videoId}/frame/${frameNum}?annotated=${annotated}`
  },
  triggerAnnotation: async (videoId: string, options?: {
    include_yolo?: boolean
    include_pose?: boolean
    show_confidence?: boolean
    show_labels?: boolean
  }) => {
    const response = await apiClient.post(`/api/videos/${videoId}/annotate`, null, {
      params: options || {}
    })
    return response.data
  },
  getAnnotationStatus: async (videoId: string) => {
    const response = await apiClient.get(`/api/videos/${videoId}/annotation-status`)
    return response.data
  },
  deleteAnnotation: async (videoId: string) => {
    const response = await apiClient.delete(`/api/videos/${videoId}/annotation`)
    return response.data
  },
  getDetections: async (videoId: string) => {
    const response = await apiClient.get(`/api/videos/${videoId}/detections`)
    return response.data
  },
  getPose: async (videoId: string) => {
    const response = await apiClient.get(`/api/videos/${videoId}/pose`)
    return response.data
  },
}

// Analysis endpoints
export const analysisApi = {
  get: async (videoId: string) => {
    const response = await apiClient.get(`/api/analysis/${videoId}`)
    return response.data
  },
  getSummary: async (videoId: string) => {
    const response = await apiClient.get(`/api/analysis/${videoId}/summary`)
    return response.data
  },
  getSimilarityMap: async () => {
    const response = await apiClient.get('/api/analysis/similarity-map')
    return response.data
  },
  getAllVideoEmbeddings: async () => {
    const response = await apiClient.get('/api/analysis/embeddings')
    return response.data
  },
}

// Training endpoints
export const trainingApi = {
  label: async (videoId: string, label: number, confidence = 'certain') => {
    const response = await apiClient.post(`/api/training/videos/${videoId}/label`, {
      label,
      confidence,
    })
    return response.data
  },
  getQueue: async () => {
    const response = await apiClient.get('/api/training/queue')
    return response.data
  },
  getStats: async () => {
    const response = await apiClient.get('/api/training/stats')
    return response.data
  },
  getStatus: async () => {
    const response = await apiClient.get('/api/training/status')
    return response.data
  },
  getModels: async () => {
    const response = await apiClient.get('/api/training/models')
    return response.data
  },
  startMLTraining: async () => {
    const response = await apiClient.post('/api/training/ml/start')
    return response.data
  },
  startYOLOTraining: async () => {
    const response = await apiClient.post('/api/training/yolo/start')
    return response.data
  },
  // Pairwise comparison endpoints
  submitPairwise: async (videoId1: string, videoId2: string, winner: number, confidence = 'confident', rawScore?: number) => {
    const response = await apiClient.post('/api/training/pairwise', {
      video_id_1: videoId1,
      video_id_2: videoId2,
      winner,
      confidence,
      raw_score: rawScore,
    })
    return response.data
  },
  getNextPairwise: async (excludeCompleted = true) => {
    const response = await apiClient.get('/api/training/pairwise/next', {
      params: { exclude_completed: excludeCompleted }
    })
    return response.data
  },
  getPairwiseStats: async () => {
    const response = await apiClient.get('/api/training/pairwise/stats')
    return response.data
  },
  getPairwiseRanking: async () => {
    const response = await apiClient.get('/api/training/pairwise/ranking')
    return response.data
  },
  // Triplet comparison endpoints
  getNextTriplet: async () => {
    const response = await apiClient.get('/api/training/triplet/next')
    return response.data
  },
  submitTriplet: async (
    referenceId: string, 
    comparisonAId: string, 
    comparisonBId: string, 
    selectedAnswer: 'A' | 'B',
    confidence: string,
    taskType: string
  ) => {
    const response = await apiClient.post('/api/training/triplet', {
      reference_id: referenceId,
      comparison_a_id: comparisonAId,
      comparison_b_id: comparisonBId,
      selected_answer: selectedAnswer,
      confidence,
      task_type: taskType,
    })
    return response.data
  },
  getTripletStats: async () => {
    const response = await apiClient.get('/api/training/triplet/stats')
    return response.data
  },
  // Rater reliability endpoints
  getRaterStats: async () => {
    const response = await apiClient.get('/api/training/raters')
    return response.data
  },
  getRaterTier: async (raterId?: string) => {
    const response = await apiClient.get('/api/training/rater/tier', {
      params: { rater_id: raterId }
    })
    return response.data
  },
}

// Model endpoints
export const modelsApi = {
  getParameters: async () => {
    const response = await apiClient.get('/api/models/parameters')
    return response.data
  },
  updateParameters: async (parameters: any) => {
    const response = await apiClient.post('/api/models/parameters', parameters)
    return response.data
  },
  getComparison: async () => {
    const response = await apiClient.get('/api/models/comparison')
    return response.data
  },
}

// SHAP endpoints
export const shapApi = {
  getLocal: async (videoId: string) => {
    const response = await apiClient.get(`/api/shap/${videoId}/local`)
    return response.data
  },
  getForcePlot: async (videoId: string) => {
    const response = await apiClient.get(`/api/shap/${videoId}/force-plot`)
    return response.data
  },
  getGlobal: async () => {
    const response = await apiClient.get('/api/shap/global')
    return response.data
  },
}

// ML Configuration endpoints
export const mlConfigApi = {
  getConfig: async () => {
    const response = await apiClient.get('/api/ml-config/')
    return response.data
  },
  updateConfig: async (config: any) => {
    const response = await apiClient.put('/api/ml-config/', config)
    return response.data
  },
  getCatBoostConfig: async () => {
    const response = await apiClient.get('/api/ml-config/catboost')
    return response.data
  },
  updateCatBoostConfig: async (config: any) => {
    const response = await apiClient.put('/api/ml-config/catboost', config)
    return response.data
  },
  getXGBoostConfig: async () => {
    const response = await apiClient.get('/api/ml-config/xgboost')
    return response.data
  },
  updateXGBoostConfig: async (config: any) => {
    const response = await apiClient.put('/api/ml-config/xgboost', config)
    return response.data
  },
  getLightGBMConfig: async () => {
    const response = await apiClient.get('/api/ml-config/lightgbm')
    return response.data
  },
  updateLightGBMConfig: async (config: any) => {
    const response = await apiClient.put('/api/ml-config/lightgbm', config)
    return response.data
  },
  getEnsembleConfig: async () => {
    const response = await apiClient.get('/api/ml-config/ensemble')
    return response.data
  },
  updateEnsembleConfig: async (config: any) => {
    const response = await apiClient.put('/api/ml-config/ensemble', config)
    return response.data
  },
  getTrainingConfig: async () => {
    const response = await apiClient.get('/api/ml-config/training')
    return response.data
  },
  updateTrainingConfig: async (config: any) => {
    const response = await apiClient.put('/api/ml-config/training', config)
    return response.data
  },
  resetToDefaults: async () => {
    const response = await apiClient.post('/api/ml-config/reset')
    return response.data
  },
  getSchema: async () => {
    const response = await apiClient.get('/api/ml-config/schema')
    return response.data
  },
  getModelsStatus: async () => {
    const response = await apiClient.get('/api/ml-config/models/status')
    return response.data
  },
  getParameterDescriptions: async () => {
    const response = await apiClient.get('/api/ml-config/parameter-descriptions')
    return response.data
  },
}

// Analysis pipeline endpoints (for VideoResults)
export const pipelineResultsApi = {
  getAll: async (videoId: string) => {
    const response = await apiClient.get(`/api/analysis/${videoId}/all`)
    return response.data
  },
  getPipeline: async (videoId: string, pipeline: string) => {
    const response = await apiClient.get(`/api/analysis/${videoId}/${pipeline}`)
    return response.data
  },
}

// Elo Ranking / Hierarchy endpoints (based on research paper methodology)
export const eloRankingApi = {
  // Submit a pairwise comparison with Elo update
  submitComparison: async (
    videoId1: string,
    videoId2: string,
    winner: number,
    degree: number = 1,
    confidence: string = 'confident',
    rawScore?: number
  ) => {
    const response = await apiClient.post('/api/elo/comparison', {
      video_id_1: videoId1,
      video_id_2: videoId2,
      winner,
      degree,
      confidence,
      raw_score: rawScore,
    })
    return response.data
  },

  // Get full hierarchy with Elo ratings and David's Scores
  getHierarchy: async () => {
    const response = await apiClient.get('/api/elo/hierarchy')
    return response.data
  },

  // Get next pair to compare (intelligent selection)
  getNextPair: async () => {
    const response = await apiClient.get('/api/elo/next-pair')
    return response.data
  },

  // Get comprehensive stats
  getStats: async () => {
    const response = await apiClient.get('/api/elo/stats')
    return response.data
  },

  // Create a hierarchy snapshot
  createSnapshot: async (name: string, description?: string) => {
    const response = await apiClient.post('/api/elo/snapshot', null, {
      params: { name, description }
    })
    return response.data
  },

  // List all snapshots
  getSnapshots: async () => {
    const response = await apiClient.get('/api/elo/snapshots')
    return response.data
  },

  // Get a specific snapshot
  getSnapshot: async (snapshotId: string) => {
    const response = await apiClient.get(`/api/elo/snapshot/${snapshotId}`)
    return response.data
  },

  // Get video Elo history
  getVideoHistory: async (videoId: string) => {
    const response = await apiClient.get(`/api/elo/video/${videoId}/history`)
    return response.data
  },

  // Recalculate all ratings from scratch
  recalculateRatings: async () => {
    const response = await apiClient.post('/api/elo/recalculate')
    return response.data
  },
}

// User types
export interface User {
  id: string
  email: string
  username: string
  role: 'admin' | 'researcher' | 'rater'
  is_active: boolean
  rater_tier?: 'gold' | 'silver' | 'bronze' | null
  created_at: string
  last_login?: string | null
}

export interface CreateUserData {
  email: string
  username: string
  password: string
  role: 'admin' | 'researcher' | 'rater'
  rater_tier?: 'gold' | 'silver' | 'bronze'
}

// User Management endpoints (Admin only)
export const usersApi = {
  // List all users
  list: async (skip = 0, limit = 100): Promise<User[]> => {
    const response = await apiClient.get('/api/auth/users', { params: { skip, limit } })
    return response.data
  },

  // Get a specific user
  get: async (userId: string): Promise<User> => {
    const response = await apiClient.get(`/api/auth/users/${userId}`)
    return response.data
  },

  // Create a new user (admin only)
  create: async (userData: CreateUserData): Promise<User> => {
    const response = await apiClient.post('/api/auth/users', userData)
    return response.data
  },

  // Update user role
  updateRole: async (userId: string, role: string) => {
    const response = await apiClient.put(`/api/auth/users/${userId}/role`, null, {
      params: { role }
    })
    return response.data
  },

  // Update user tier (raters only)
  updateTier: async (userId: string, tier: string) => {
    const response = await apiClient.put(`/api/auth/users/${userId}/tier`, null, {
      params: { tier }
    })
    return response.data
  },

  // Enable/disable user
  updateStatus: async (userId: string, isActive: boolean) => {
    const response = await apiClient.put(`/api/auth/users/${userId}/status`, null, {
      params: { is_active: isActive }
    })
    return response.data
  },

  // Delete user
  delete: async (userId: string) => {
    const response = await apiClient.delete(`/api/auth/users/${userId}`)
    return response.data
  },
}

// Tutorial / Gold Task endpoints
export interface TutorialExample {
  id: string
  video_id_1: string
  video_id_2: string
  description: string
  hint: string
  correct_answer: number  // -3 to 3 (7-point scale)
  difficulty: string
  order: number
}

export interface GoldTask {
  id: string
  video_id_1: string
  video_id_2: string
  correct_winner: number
  correct_degree: number
  difficulty: string
  description?: string
  hint?: string
  is_tutorial: boolean
  tutorial_order?: number
  is_active: boolean
  created_at: string
}

export interface CreateGoldTaskData {
  video_id_1: string
  video_id_2: string
  correct_winner: number
  correct_degree?: number
  difficulty?: string
  description?: string
  hint?: string
  is_tutorial?: boolean
  tutorial_order?: number
}

export const tutorialApi = {
  // Get tutorial examples for pairwise tutorial
  getExamples: async (): Promise<{ examples: TutorialExample[]; total: number }> => {
    const response = await apiClient.get('/api/tutorial/examples')
    return response.data
  },

  // Auto-generate tutorial examples from random videos (admin only)
  autoGenerate: async (count = 3) => {
    const response = await apiClient.post('/api/tutorial/examples/auto-generate', null, {
      params: { count }
    })
    return response.data
  },

  // List all gold tasks / tutorials
  listTasks: async (params?: { is_tutorial?: boolean; is_active?: boolean }) => {
    const response = await apiClient.get('/api/tutorial/tasks', { params })
    return response.data
  },

  // Create a gold task / tutorial
  createTask: async (taskData: CreateGoldTaskData): Promise<GoldTask> => {
    const response = await apiClient.post('/api/tutorial/tasks', taskData)
    return response.data
  },

  // Update a gold task / tutorial
  updateTask: async (taskId: string, updates: Partial<GoldTask>) => {
    const response = await apiClient.put(`/api/tutorial/tasks/${taskId}`, updates)
    return response.data
  },

  // Delete a gold task / tutorial
  deleteTask: async (taskId: string) => {
    const response = await apiClient.delete(`/api/tutorial/tasks/${taskId}`)
    return response.data
  },

  // Get tutorial stats
  getStats: async () => {
    const response = await apiClient.get('/api/tutorial/stats')
    return response.data
  },
}

// Auth endpoints
export const authApi = {
  login: async (email: string, password: string) => {
    const response = await apiClient.post('/api/auth/login', { email, password })
    return response.data
  },

  logout: async () => {
    const response = await apiClient.post('/api/auth/logout')
    return response.data
  },

  register: async (email: string, username: string, password: string) => {
    const response = await apiClient.post('/api/auth/register', {
      email,
      username,
      password,
    })
    return response.data
  },

  refresh: async (refreshToken: string) => {
    const response = await apiClient.post('/api/auth/refresh', {
      refresh_token: refreshToken
    })
    return response.data
  },

  me: async (): Promise<User> => {
    const response = await apiClient.get('/api/auth/me')
    return response.data
  },

  changePassword: async (currentPassword: string, newPassword: string) => {
    const response = await apiClient.put('/api/auth/password', {
      current_password: currentPassword,
      new_password: newPassword,
    })
    return response.data
  },
}
