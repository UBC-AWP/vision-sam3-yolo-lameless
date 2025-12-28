import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

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
