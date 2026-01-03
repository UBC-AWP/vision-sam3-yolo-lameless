/**
 * Video Results Page
 * Comprehensive view of all pipeline results for a video
 */
import { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  Activity,
  Brain,
  Eye,
  Network,
  Cpu,
  GitMerge,
  CheckCircle,
  XCircle,
  AlertCircle,
  ChevronDown,
  ChevronUp,
  ArrowLeft,
  Play,
  Loader2,
  BarChart3,
  Users,
  Microscope,
  Sparkles
} from 'lucide-react'
import ShapExplanation from '@/components/ShapExplanation'
import LLMExplanation from '@/components/LLMExplanation'

const API_BASE = '/api'

interface PipelineResult {
  status: 'success' | 'error' | 'pending' | 'not_available'
  data: any
  error?: string
}

interface AllResults {
  yolo: PipelineResult
  sam3: PipelineResult
  dinov3: PipelineResult
  tleap: PipelineResult
  tcn: PipelineResult
  transformer: PipelineResult
  gnn: PipelineResult
  ml: PipelineResult
  fusion: PipelineResult
}

export default function VideoResults() {
  const { videoId } = useParams<{ videoId: string }>()
  const navigate = useNavigate()
  const [loading, setLoading] = useState(true)
  const [videoInfo, setVideoInfo] = useState<any>(null)
  const [results, setResults] = useState<AllResults | null>(null)
  const [expandedPipelines, setExpandedPipelines] = useState<Set<string>>(new Set(['fusion']))
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (videoId) {
      loadAllResults()
    }
  }, [videoId])

  const loadAllResults = async () => {
    setLoading(true)
    setError(null)

    try {
      // Load video info
      const videoResponse = await fetch(`${API_BASE}/videos/${videoId}`)
      if (videoResponse.ok) {
        setVideoInfo(await videoResponse.json())
      }

      // Load all pipeline results
      const pipelines = ['yolo', 'sam3', 'dinov3', 'tleap', 'tcn', 'transformer', 'gnn', 'ml', 'fusion']
      const resultPromises = pipelines.map(async (pipeline) => {
        try {
          const response = await fetch(`${API_BASE}/analysis/${videoId}/${pipeline}`)
          if (response.ok) {
            const data = await response.json()
            return { pipeline, result: { status: 'success' as const, data } }
          } else if (response.status === 404) {
            return { pipeline, result: { status: 'not_available' as const, data: null } }
          } else {
            return { pipeline, result: { status: 'error' as const, data: null, error: 'Failed to load' } }
          }
        } catch (err) {
          return { pipeline, result: { status: 'error' as const, data: null, error: String(err) } }
        }
      })

      const allResults = await Promise.all(resultPromises)
      const resultsMap: any = {}
      allResults.forEach(({ pipeline, result }) => {
        resultsMap[pipeline] = result
      })
      setResults(resultsMap)
    } catch (err) {
      setError('Failed to load results')
    } finally {
      setLoading(false)
    }
  }

  const togglePipeline = (pipeline: string) => {
    setExpandedPipelines(prev => {
      const newSet = new Set(prev)
      if (newSet.has(pipeline)) {
        newSet.delete(pipeline)
      } else {
        newSet.add(pipeline)
      }
      return newSet
    })
  }

  const getPipelineIcon = (pipeline: string) => {
    switch (pipeline) {
      case 'yolo': return <Eye className="h-5 w-5" />
      case 'sam3': return <Activity className="h-5 w-5" />
      case 'dinov3': return <Brain className="h-5 w-5" />
      case 'tleap': return <Users className="h-5 w-5" />
      case 'tcn': return <BarChart3 className="h-5 w-5" />
      case 'transformer': return <Cpu className="h-5 w-5" />
      case 'gnn': return <Network className="h-5 w-5" />
      case 'ml': return <GitMerge className="h-5 w-5" />
      case 'fusion': return <GitMerge className="h-5 w-5 text-primary" />
      default: return <Activity className="h-5 w-5" />
    }
  }

  const getPipelineDescription = (pipeline: string) => {
    switch (pipeline) {
      case 'yolo': return 'Object Detection - Cow detection with bounding boxes'
      case 'sam3': return 'Segmentation - Instance segmentation masks'
      case 'dinov3': return 'Visual Embeddings - Feature extraction & similarity'
      case 'tleap': return 'Pose Estimation - 20-keypoint tracking'
      case 'tcn': return 'Temporal CNN - Sequence classification'
      case 'transformer': return 'Gait Transformer - Attention-based analysis'
      case 'gnn': return 'Graph Neural Network - Relational learning'
      case 'ml': return 'ML Ensemble - Gradient boosting models'
      case 'fusion': return 'Final Fusion - Combined prediction'
      default: return ''
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'success': return <CheckCircle className="h-5 w-5 text-success" />
      case 'error': return <XCircle className="h-5 w-5 text-destructive" />
      case 'pending': return <Loader2 className="h-5 w-5 text-warning animate-spin" />
      case 'not_available': return <AlertCircle className="h-5 w-5 text-muted-foreground" />
      default: return null
    }
  }

  const renderYoloResults = (data: any) => (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-muted/50 rounded-lg p-4">
          <div className="text-sm text-muted-foreground">Detections</div>
          <div className="text-2xl font-bold">{data.features?.num_detections || data.detections?.length || 0}</div>
        </div>
        <div className="bg-muted/50 rounded-lg p-4">
          <div className="text-sm text-muted-foreground">Avg Confidence</div>
          <div className="text-2xl font-bold">{((data.features?.avg_confidence || 0) * 100).toFixed(1)}%</div>
        </div>
        <div className="bg-muted/50 rounded-lg p-4">
          <div className="text-sm text-muted-foreground">Position Stability</div>
          <div className="text-2xl font-bold">{(data.features?.position_stability || 0).toFixed(3)}</div>
        </div>
        <div className="bg-muted/50 rounded-lg p-4">
          <div className="text-sm text-muted-foreground">Detection Rate</div>
          <div className="text-2xl font-bold">{((data.features?.detection_rate || 0) * 100).toFixed(1)}%</div>
        </div>
      </div>
    </div>
  )

  const renderTleapResults = (data: any) => (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-muted/50 rounded-lg p-4">
          <div className="text-sm text-muted-foreground">Frames Processed</div>
          <div className="text-2xl font-bold">{data.frames_processed || 0}</div>
        </div>
        <div className="bg-muted/50 rounded-lg p-4">
          <div className="text-sm text-muted-foreground">Lameness Score</div>
          <div className="text-2xl font-bold">
            {data.locomotion_features?.lameness_score?.toFixed(3) || 'N/A'}
          </div>
        </div>
        <div className="bg-muted/50 rounded-lg p-4">
          <div className="text-sm text-muted-foreground">Head Bob</div>
          <div className="text-2xl font-bold">
            {data.locomotion_features?.head_bob_magnitude?.toFixed(3) || 'N/A'}
          </div>
        </div>
        <div className="bg-muted/50 rounded-lg p-4">
          <div className="text-sm text-muted-foreground">Back Arch</div>
          <div className="text-2xl font-bold">
            {data.locomotion_features?.back_arch_mean?.toFixed(3) || 'N/A'}
          </div>
        </div>
      </div>
      {data.pose_sequences && data.pose_sequences.length > 0 && (
        <div className="text-sm text-muted-foreground">
          {data.pose_sequences[0].keypoints?.length || 0} keypoints tracked per frame
        </div>
      )}
    </div>
  )

  const renderTcnResults = (data: any) => (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-muted/50 rounded-lg p-4">
          <div className="text-sm text-muted-foreground">Severity Score</div>
          <div className={`text-2xl font-bold ${data.severity_score > 0.5 ? 'text-destructive' : 'text-success'}`}>
            {(data.severity_score * 100).toFixed(1)}%
          </div>
        </div>
        <div className="bg-muted/50 rounded-lg p-4">
          <div className="text-sm text-muted-foreground">Uncertainty</div>
          <div className="text-2xl font-bold">{(data.uncertainty * 100).toFixed(2)}%</div>
        </div>
        <div className="bg-muted/50 rounded-lg p-4">
          <div className="text-sm text-muted-foreground">Prediction</div>
          <div className={`text-2xl font-bold ${data.prediction === 1 ? 'text-destructive' : 'text-success'}`}>
            {data.prediction === 1 ? 'Lame' : 'Healthy'}
          </div>
        </div>
        <div className="bg-muted/50 rounded-lg p-4">
          <div className="text-sm text-muted-foreground">Input Frames</div>
          <div className="text-2xl font-bold">{data.input_frames || 0}</div>
        </div>
      </div>
    </div>
  )

  const renderGnnResults = (data: any) => (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        <div className="bg-muted/50 rounded-lg p-4">
          <div className="text-sm text-muted-foreground">Severity Score</div>
          <div className={`text-2xl font-bold ${data.severity_score > 0.5 ? 'text-destructive' : 'text-success'}`}>
            {(data.severity_score * 100).toFixed(1)}%
          </div>
        </div>
        <div className="bg-muted/50 rounded-lg p-4">
          <div className="text-sm text-muted-foreground">Uncertainty</div>
          <div className="text-2xl font-bold">{((data.uncertainty || 0) * 100).toFixed(2)}%</div>
        </div>
        <div className="bg-muted/50 rounded-lg p-4">
          <div className="text-sm text-muted-foreground">Graph Nodes</div>
          <div className="text-2xl font-bold">{data.graph_info?.num_nodes || 'N/A'}</div>
        </div>
      </div>
      {data.neighbor_influence && data.neighbor_influence.length > 0 && (
        <div>
          <div className="text-sm font-medium mb-2">Similar Videos (Neighbors)</div>
          <div className="space-y-1">
            {data.neighbor_influence.slice(0, 5).map((neighbor: any, idx: number) => (
              <div key={idx} className="flex justify-between text-sm bg-muted/30 px-3 py-2 rounded">
                <span className="font-mono text-xs">{neighbor.video_id.slice(0, 8)}...</span>
                <span className={neighbor.score > 0.5 ? 'text-destructive' : 'text-success'}>
                  {(neighbor.score * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )

  const renderDinov3Results = (data: any) => (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        <div className="bg-muted/50 rounded-lg p-4">
          <div className="text-sm text-muted-foreground">Neighbor Evidence</div>
          <div className={`text-2xl font-bold ${data.neighbor_evidence > 0.5 ? 'text-destructive' : 'text-success'}`}>
            {(data.neighbor_evidence * 100).toFixed(1)}%
          </div>
        </div>
        <div className="bg-muted/50 rounded-lg p-4">
          <div className="text-sm text-muted-foreground">Similar Cases</div>
          <div className="text-2xl font-bold">{data.similar_cases?.length || 0}</div>
        </div>
        <div className="bg-muted/50 rounded-lg p-4">
          <div className="text-sm text-muted-foreground">Embedding Dim</div>
          <div className="text-2xl font-bold">{data.embedding_dim || 768}</div>
        </div>
      </div>
    </div>
  )

  const renderMlResults = (data: any) => {
    const predictions = data.predictions || {}
    const ensemble = predictions.ensemble || data.ensemble || {}
    const catboost = predictions.catboost || {}
    const xgboost = predictions.xgboost || {}
    const lightgbm = predictions.lightgbm || {}
    const features = data.features || []
    const featureNames = data.feature_names || []

    return (
      <div className="space-y-6">
        {/* Ensemble Result - Featured */}
        <div className="bg-gradient-to-br from-primary/10 to-primary/5 rounded-xl p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-lg">Ensemble Prediction</h3>
            <span className={`px-3 py-1 rounded-full text-sm font-medium ${
              ensemble.prediction === 1
                ? 'bg-destructive/10 text-destructive'
                : 'bg-success/10 text-success'
            }`}>
              {ensemble.prediction === 1 ? 'LAME' : 'HEALTHY'}
            </span>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex-1">
              <div className="h-6 bg-muted rounded-full overflow-hidden">
                <div
                  className={`h-full transition-all ${
                    (ensemble.probability || 0) > 0.5 ? 'bg-destructive' : 'bg-success'
                  }`}
                  style={{ width: `${(ensemble.probability || 0) * 100}%` }}
                />
              </div>
            </div>
            <div className="text-2xl font-bold w-24 text-right">
              {((ensemble.probability || 0) * 100).toFixed(1)}%
            </div>
          </div>
        </div>

        {/* Individual Model Results */}
        <div>
          <h3 className="font-semibold mb-3">Individual Model Predictions</h3>
          <div className="grid md:grid-cols-3 gap-4">
            {/* CatBoost */}
            <div className="border border-border rounded-lg p-4 bg-card">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-3 h-3 bg-info rounded" />
                <span className="font-medium">CatBoost</span>
                {ensemble.weights?.catboost && (
                  <span className="text-xs bg-info/10 text-info px-2 py-0.5 rounded ml-auto">
                    {((ensemble.weights.catboost) * 100).toFixed(0)}% weight
                  </span>
                )}
              </div>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Probability</span>
                  <span className={`font-medium ${
                    (catboost.probability || 0) > 0.5 ? 'text-destructive' : 'text-success'
                  }`}>
                    {catboost.probability !== undefined
                      ? `${(catboost.probability * 100).toFixed(1)}%`
                      : 'N/A'}
                  </span>
                </div>
                <div className="h-2 bg-muted rounded-full overflow-hidden">
                  <div
                    className={`h-full ${(catboost.probability || 0) > 0.5 ? 'bg-destructive/80' : 'bg-success/80'}`}
                    style={{ width: `${(catboost.probability || 0) * 100}%` }}
                  />
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Prediction</span>
                  <span className={`font-medium ${
                    catboost.prediction === 1 ? 'text-destructive' : 'text-success'
                  }`}>
                    {catboost.prediction !== undefined
                      ? (catboost.prediction === 1 ? 'Lame' : 'Healthy')
                      : 'N/A'}
                  </span>
                </div>
              </div>
            </div>

            {/* XGBoost */}
            <div className="border border-border rounded-lg p-4 bg-card">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-3 h-3 bg-success rounded" />
                <span className="font-medium">XGBoost</span>
                {ensemble.weights?.xgboost && (
                  <span className="text-xs bg-success/10 text-success px-2 py-0.5 rounded ml-auto">
                    {((ensemble.weights.xgboost) * 100).toFixed(0)}% weight
                  </span>
                )}
              </div>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Probability</span>
                  <span className={`font-medium ${
                    (xgboost.probability || 0) > 0.5 ? 'text-destructive' : 'text-success'
                  }`}>
                    {xgboost.probability !== undefined
                      ? `${(xgboost.probability * 100).toFixed(1)}%`
                      : 'N/A'}
                  </span>
                </div>
                <div className="h-2 bg-muted rounded-full overflow-hidden">
                  <div
                    className={`h-full ${(xgboost.probability || 0) > 0.5 ? 'bg-destructive/80' : 'bg-success/80'}`}
                    style={{ width: `${(xgboost.probability || 0) * 100}%` }}
                  />
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Prediction</span>
                  <span className={`font-medium ${
                    xgboost.prediction === 1 ? 'text-destructive' : 'text-success'
                  }`}>
                    {xgboost.prediction !== undefined
                      ? (xgboost.prediction === 1 ? 'Lame' : 'Healthy')
                      : 'N/A'}
                  </span>
                </div>
              </div>
            </div>

            {/* LightGBM */}
            <div className="border border-border rounded-lg p-4 bg-card">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-3 h-3 bg-purple-500 rounded" />
                <span className="font-medium">LightGBM</span>
                {ensemble.weights?.lightgbm && (
                  <span className="text-xs bg-purple-500/10 text-purple-500 px-2 py-0.5 rounded ml-auto">
                    {((ensemble.weights.lightgbm) * 100).toFixed(0)}% weight
                  </span>
                )}
              </div>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Probability</span>
                  <span className={`font-medium ${
                    (lightgbm.probability || 0) > 0.5 ? 'text-destructive' : 'text-success'
                  }`}>
                    {lightgbm.probability !== undefined
                      ? `${(lightgbm.probability * 100).toFixed(1)}%`
                      : 'N/A'}
                  </span>
                </div>
                <div className="h-2 bg-muted rounded-full overflow-hidden">
                  <div
                    className={`h-full ${(lightgbm.probability || 0) > 0.5 ? 'bg-destructive/80' : 'bg-success/80'}`}
                    style={{ width: `${(lightgbm.probability || 0) * 100}%` }}
                  />
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Prediction</span>
                  <span className={`font-medium ${
                    lightgbm.prediction === 1 ? 'text-destructive' : 'text-success'
                  }`}>
                    {lightgbm.prediction !== undefined
                      ? (lightgbm.prediction === 1 ? 'Lame' : 'Healthy')
                      : 'N/A'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Model Agreement Visualization */}
        <div className="border border-border rounded-lg p-4 bg-card">
          <h3 className="font-semibold mb-3">Model Agreement</h3>
          <div className="flex items-center gap-4">
            <div className="flex-1 h-8 flex rounded overflow-hidden">
              <div
                className="bg-info flex items-center justify-center text-primary-foreground text-xs font-medium"
                style={{
                  width: `${(ensemble.weights?.catboost || 0.33) * 100}%`,
                  opacity: catboost.prediction === ensemble.prediction ? 1 : 0.3
                }}
              >
                {catboost.prediction === ensemble.prediction ? 'Agrees' : 'Disagrees'}
              </div>
              <div
                className="bg-success flex items-center justify-center text-primary-foreground text-xs font-medium"
                style={{
                  width: `${(ensemble.weights?.xgboost || 0.33) * 100}%`,
                  opacity: xgboost.prediction === ensemble.prediction ? 1 : 0.3
                }}
              >
                {xgboost.prediction === ensemble.prediction ? 'Agrees' : 'Disagrees'}
              </div>
              <div
                className="bg-purple-500 flex items-center justify-center text-primary-foreground text-xs font-medium"
                style={{
                  width: `${(ensemble.weights?.lightgbm || 0.34) * 100}%`,
                  opacity: lightgbm.prediction === ensemble.prediction ? 1 : 0.3
                }}
              >
                {lightgbm.prediction === ensemble.prediction ? 'Agrees' : 'Disagrees'}
              </div>
            </div>
          </div>
          <div className="mt-2 text-sm text-muted-foreground">
            {[catboost, xgboost, lightgbm].filter(m => m.prediction === ensemble.prediction).length} of 3 models agree with ensemble
          </div>
        </div>

        {/* Feature Values */}
        {features.length > 0 && featureNames.length > 0 && (
          <div className="border border-border rounded-lg p-4 bg-card">
            <h3 className="font-semibold mb-3">Input Features</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {featureNames.map((name: string, idx: number) => (
                <div key={name} className="bg-muted/50 rounded px-3 py-2">
                  <div className="text-xs text-muted-foreground truncate" title={name}>
                    {name.replace(/_/g, ' ')}
                  </div>
                  <div className="font-mono text-sm font-medium">
                    {typeof features[idx] === 'number'
                      ? features[idx].toFixed(4)
                      : features[idx]}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Pipeline Results Availability */}
        {data.pipeline_results_available && (
          <div className="text-sm text-muted-foreground">
            <span className="font-medium">Pipeline data used: </span>
            {Object.entries(data.pipeline_results_available)
              .filter(([_, available]) => available)
              .map(([pipeline]) => pipeline)
              .join(', ') || 'None'}
          </div>
        )}
      </div>
    )
  }

  const renderFusionResults = (data: any) => {
    const fusion = data.fusion_result || data
    return (
      <div className="space-y-6">
        {/* Main Result */}
        <div className="text-center p-6 bg-gradient-to-br from-primary/10 to-primary/5 rounded-xl">
          <div className="text-sm text-muted-foreground mb-2">Final Lameness Prediction</div>
          <div className={`text-5xl font-bold mb-2 ${
            fusion.final_prediction === 1 ? 'text-destructive' : 'text-success'
          }`}>
            {fusion.final_prediction === 1 ? 'LAME' : 'HEALTHY'}
          </div>
          <div className="text-lg text-muted-foreground">
            Confidence: {((fusion.final_probability || 0) * 100).toFixed(1)}%
          </div>
          {/* Confidence bar */}
          <div className="mt-4 max-w-md mx-auto">
            <div className="h-4 bg-gradient-to-r from-success via-warning to-destructive rounded-full relative">
              <div
                className="absolute top-1/2 w-1 h-6 bg-foreground rounded -translate-y-1/2 shadow-lg"
                style={{ left: `${(fusion.final_probability || 0.5) * 100}%` }}
              />
            </div>
            <div className="flex justify-between text-xs text-muted-foreground mt-1">
              <span>Healthy (0%)</span>
              <span>Lame (100%)</span>
            </div>
          </div>
        </div>

        {/* Pipeline Contributions */}
        {fusion.pipeline_contributions && (
          <div>
            <div className="text-sm font-medium mb-3">Pipeline Contributions</div>
            <div className="space-y-2">
              {Object.entries(fusion.pipeline_contributions).map(([pipeline, value]) => {
                if (value === null) return null
                const score = typeof value === 'object' ? (value as any).probability : value
                if (score === null || score === undefined) return null
                return (
                  <div key={pipeline} className="flex items-center gap-3">
                    <div className="w-24 text-sm capitalize">{pipeline}</div>
                    <div className="flex-1 h-6 bg-muted rounded-full overflow-hidden">
                      <div
                        className={`h-full transition-all ${
                          score > 0.5 ? 'bg-destructive/80' : 'bg-success/80'
                        }`}
                        style={{ width: `${score * 100}%` }}
                      />
                    </div>
                    <div className="w-16 text-right text-sm font-medium">
                      {(score * 100).toFixed(1)}%
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        )}
      </div>
    )
  }

  const renderPipelineResults = (pipeline: string, data: any) => {
    switch (pipeline) {
      case 'yolo': return renderYoloResults(data)
      case 'tleap': return renderTleapResults(data)
      case 'tcn': return renderTcnResults(data)
      case 'transformer': return renderTcnResults(data) // Same structure
      case 'gnn': return renderGnnResults(data)
      case 'dinov3': return renderDinov3Results(data)
      case 'ml': return renderMlResults(data)
      case 'fusion': return renderFusionResults(data)
      default:
        return (
          <pre className="text-xs bg-muted p-4 rounded overflow-auto max-h-64">
            {JSON.stringify(data, null, 2)}
          </pre>
        )
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4 text-primary" />
          <p className="text-muted-foreground">Loading pipeline results...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="text-center py-12">
        <XCircle className="h-12 w-12 text-destructive mx-auto mb-4" />
        <h2 className="text-xl font-bold mb-2">Failed to load results</h2>
        <p className="text-muted-foreground mb-4">{error}</p>
        <button
          onClick={() => navigate('/')}
          className="px-4 py-2 bg-primary text-primary-foreground rounded-lg"
        >
          Back to Dashboard
        </button>
      </div>
    )
  }

  const pipelineOrder = ['fusion', 'yolo', 'sam3', 'dinov3', 'tleap', 'tcn', 'transformer', 'gnn', 'ml']

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button
            onClick={() => navigate('/')}
            className="p-2 hover:bg-muted rounded-lg transition-colors"
          >
            <ArrowLeft className="h-5 w-5" />
          </button>
          <div>
            <h1 className="text-2xl font-bold">Pipeline Results</h1>
            <p className="text-sm text-muted-foreground font-mono">{videoId}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => navigate(`/video/${videoId}`)}
            className="flex items-center gap-2 px-4 py-2 bg-muted hover:bg-muted/80 rounded-lg transition-colors"
          >
            <Play className="h-4 w-4" />
            View Video
          </button>
          <button
            onClick={() => navigate(`/pipeline-analysis/${videoId}`)}
            className="flex items-center gap-2 px-4 py-2 bg-info text-primary-foreground hover:bg-info/90 rounded-lg transition-colors"
          >
            <Microscope className="h-4 w-4" />
            Deep Analysis
          </button>
          <button
            onClick={loadAllResults}
            className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
          >
            Refresh
          </button>
        </div>
      </div>

      {/* Video Info */}
      {videoInfo && (
        <div className="bg-card border rounded-lg p-4">
          <div className="flex items-center gap-6 text-sm">
            <div>
              <span className="text-muted-foreground">Filename:</span>{' '}
              <span className="font-medium">{videoInfo.filename}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Size:</span>{' '}
              <span className="font-medium">{(videoInfo.file_size / 1024 / 1024).toFixed(2)} MB</span>
            </div>
            {videoInfo.metadata && (
              <>
                <div>
                  <span className="text-muted-foreground">Duration:</span>{' '}
                  <span className="font-medium">{videoInfo.metadata.duration?.toFixed(1)}s</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Resolution:</span>{' '}
                  <span className="font-medium">{videoInfo.metadata.width}x{videoInfo.metadata.height}</span>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* AI Explanations Section */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* LLM Explanation */}
        <LLMExplanation videoId={videoId!} />
        
        {/* SHAP Explanation */}
        <div className="border border-border rounded-xl bg-card overflow-hidden">
          <div className="p-4 border-b border-border bg-muted/30">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-purple-500/20 flex items-center justify-center">
                <Sparkles className="h-5 w-5 text-purple-500" />
              </div>
              <div>
                <h3 className="font-semibold text-foreground">Feature Importance</h3>
                <p className="text-sm text-muted-foreground">SHAP-based explanation</p>
              </div>
            </div>
          </div>
          <div className="p-4">
            <ShapExplanation videoId={videoId!} />
          </div>
        </div>
      </div>

      {/* Pipeline Results */}
      <div className="space-y-4">
        {results && pipelineOrder.map(pipeline => {
          const result = results[pipeline as keyof AllResults]
          if (!result) return null

          const isExpanded = expandedPipelines.has(pipeline)

          return (
            <div key={pipeline} className={`border rounded-lg overflow-hidden ${
              pipeline === 'fusion' ? 'border-primary/50 bg-primary/5' : 'bg-card'
            }`}>
              {/* Header */}
              <button
                onClick={() => togglePipeline(pipeline)}
                className="w-full flex items-center justify-between p-4 hover:bg-muted/50 transition-colors"
              >
                <div className="flex items-center gap-3">
                  {getPipelineIcon(pipeline)}
                  <div className="text-left">
                    <div className="font-semibold uppercase">{pipeline}</div>
                    <div className="text-xs text-muted-foreground">
                      {getPipelineDescription(pipeline)}
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  {getStatusIcon(result.status)}
                  {isExpanded ? (
                    <ChevronUp className="h-5 w-5 text-muted-foreground" />
                  ) : (
                    <ChevronDown className="h-5 w-5 text-muted-foreground" />
                  )}
                </div>
              </button>

              {/* Content */}
              {isExpanded && (
                <div className="border-t p-4">
                  {result.status === 'success' && result.data ? (
                    renderPipelineResults(pipeline, result.data)
                  ) : result.status === 'not_available' ? (
                    <div className="text-center py-8 text-muted-foreground">
                      <AlertCircle className="h-8 w-8 mx-auto mb-2" />
                      <p>No results available for this pipeline</p>
                      <p className="text-sm">The pipeline may not have processed this video yet</p>
                    </div>
                  ) : (
                    <div className="text-center py-8 text-red-500">
                      <XCircle className="h-8 w-8 mx-auto mb-2" />
                      <p>Failed to load results</p>
                      {result.error && <p className="text-sm">{result.error}</p>}
                    </div>
                  )}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
