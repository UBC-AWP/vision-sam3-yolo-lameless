import { useEffect, useState, useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { videosApi, analysisApi, trainingApi } from '@/api/client'
import VideoPlayer from '@/components/VideoPlayer'
import ShapExplanation from '@/components/ShapExplanation'
import LLMExplanation from '@/components/LLMExplanation'

interface DetectionFrame {
  frame: number
  time: number
  detections: Array<{
    bbox: number[]
    confidence: number
    class: string
  }>
}

export default function VideoAnalysis() {
  const { videoId } = useParams()
  const navigate = useNavigate()
  const [video, setVideo] = useState<any>(null)
  const [analysis, setAnalysis] = useState<any>(null)
  const [detections, setDetections] = useState<DetectionFrame[]>([])
  const [currentFrame, setCurrentFrame] = useState(0)
  const [loading, setLoading] = useState(true)
  const [labeling, setLabeling] = useState(false)
  const [currentLabel, setCurrentLabel] = useState<number | null>(null)

  useEffect(() => {
    if (videoId) {
      loadData()
    }
  }, [videoId])

  const loadData = async () => {
    try {
      const [videoData, analysisData] = await Promise.all([
        videosApi.get(videoId!),
        analysisApi.getSummary(videoId!).catch(() => null),
      ])
      setVideo(videoData)
      setAnalysis(analysisData)
      setCurrentLabel(videoData.label)

      // Load detections
      try {
        const detectionsData = await videosApi.getDetections(videoId!)
        setDetections(detectionsData.detections || [])
      } catch (err) {
        // No detections available
      }
    } catch (error) {
      console.error('Failed to load data:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleLabel = async (label: number) => {
    if (!videoId) return

    setLabeling(true)
    try {
      await trainingApi.label(videoId, label)
      setCurrentLabel(label)
    } catch (error) {
      alert('Failed to save label')
    } finally {
      setLabeling(false)
    }
  }

  const handleFrameChange = useCallback((frame: number) => {
    setCurrentFrame(frame)
  }, [])

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return
      }
      
      switch (e.key.toLowerCase()) {
        case 's':
          handleLabel(0) // Sound
          break
        case 'l':
          handleLabel(1) // Lame
          break
      }
    }

    window.addEventListener('keypress', handleKeyPress)
    return () => window.removeEventListener('keypress', handleKeyPress)
  }, [videoId])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <div className="text-muted-foreground">Loading video analysis...</div>
        </div>
      </div>
    )
  }

  if (!video) {
    return (
      <div className="text-center py-12">
        <h2 className="text-2xl font-bold mb-4">Video not found</h2>
        <button
          onClick={() => navigate('/')}
          className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90"
        >
          Back to Dashboard
        </button>
      </div>
    )
  }

  const probability = analysis?.final_probability || 0.5
  const prediction = analysis?.final_prediction || (probability > 0.5 ? 1 : 0)

  // Find detections for current frame
  const currentDetections = detections.find(d => d.frame === currentFrame)

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <h2 className="text-3xl font-bold">Video Analysis</h2>
          <p className="text-muted-foreground mt-1">
            {video.filename} • {(video.file_size / 1024 / 1024).toFixed(2)} MB
          </p>
        </div>
        <div className="flex items-center gap-2">
          {currentLabel !== null && (
            <span className={`px-3 py-1 rounded-full text-sm font-medium ${
              currentLabel === 0
                ? 'bg-success/20 text-success'
                : 'bg-destructive/20 text-destructive'
            }`}>
              Labeled: {currentLabel === 0 ? 'Sound' : 'Lame'}
            </span>
          )}
          <button
            onClick={() => navigate(`/results/${videoId}`)}
            className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 text-sm font-medium"
          >
            View All Results
          </button>
          <button
            onClick={() => navigate('/')}
            className="px-3 py-1 text-sm text-muted-foreground hover:text-foreground"
          >
            ← Back
          </button>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Video Player - 2 columns */}
        <div className="lg:col-span-2 space-y-4">
          <VideoPlayer
            videoId={videoId!}
            showAnnotations={false}
            onFrameChange={handleFrameChange}
          />

          {/* Detection Timeline */}
          {detections.length > 0 && (
            <div className="border border-border rounded-lg p-4 bg-card">
              <h3 className="text-sm font-semibold mb-3 text-foreground">Detection Timeline</h3>
              <div className="relative h-8 bg-muted rounded overflow-hidden">
                {/* Detection markers */}
                {detections.map((det, idx) => {
                  const totalFrames = video.metadata?.frame_count || 1
                  const position = (det.frame / totalFrames) * 100
                  const hasDetection = det.detections.length > 0
                  
                  return (
                    <div
                      key={idx}
                      className={`absolute top-0 w-1 h-full ${
                        hasDetection ? 'bg-green-500' : 'bg-gray-300'
                      }`}
                      style={{ left: `${position}%` }}
                      title={`Frame ${det.frame}: ${det.detections.length} detections`}
                    />
                  )
                })}
                
                {/* Current frame indicator */}
                <div
                  className="absolute top-0 w-0.5 h-full bg-red-500 z-10"
                  style={{ left: `${(currentFrame / (video.metadata?.frame_count || 1)) * 100}%` }}
                />
              </div>
              <div className="flex justify-between text-xs text-muted-foreground mt-1">
                <span>Frame 0</span>
                <span>Frame {video.metadata?.frame_count || '?'}</span>
              </div>
            </div>
          )}

          {/* Current Frame Detections */}
          {currentDetections && currentDetections.detections.length > 0 && (
            <div className="border border-border rounded-lg p-4 bg-card">
              <h3 className="text-sm font-semibold mb-2 text-foreground">
                Frame {currentFrame} Detections ({currentDetections.detections.length})
              </h3>
              <div className="space-y-2">
                {currentDetections.detections.map((det, idx) => (
                  <div key={idx} className="flex justify-between text-sm">
                    <span className="capitalize">{det.class}</span>
                    <span className="text-muted-foreground">
                      {(det.confidence * 100).toFixed(1)}% confidence
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right sidebar */}
        <div className="space-y-6">
          {/* Prediction Card */}
          <div className="border border-border rounded-lg p-6 bg-card">
            <h3 className="text-lg font-semibold mb-4 text-foreground">Prediction</h3>
            
            <div className="text-center mb-6">
              <div className={`text-4xl font-bold ${
                prediction === 1 ? 'text-destructive' : 'text-success'
              }`}>
                {prediction === 1 ? 'Lame' : 'Sound'}
              </div>
              <div className="text-muted-foreground mt-1">
                {(probability * 100).toFixed(1)}% confidence
              </div>
            </div>

            {/* Probability bar */}
            <div className="mb-6">
              <div className="flex justify-between text-xs text-muted-foreground mb-1">
                <span>Sound</span>
                <span>Lame</span>
              </div>
              <div className="h-4 bg-gradient-to-r from-green-500 via-yellow-500 to-red-500 rounded-full relative">
                <div
                  className="absolute top-1/2 w-1 h-6 bg-black rounded -translate-y-1/2"
                  style={{ left: `${probability * 100}%` }}
                />
              </div>
            </div>

            {/* Pipeline contributions */}
            {analysis?.pipeline_contributions && (
              <div className="space-y-2">
                <h4 className="text-sm font-medium">Pipeline Scores</h4>
                {Object.entries(analysis.pipeline_contributions).map(([pipeline, value]) => (
                  <div key={pipeline} className="flex justify-between text-sm">
                    <span className="capitalize">{pipeline.replace('_', ' ')}</span>
                    <span className="text-muted-foreground">
                      {value !== null ? `${(Number(value) * 100).toFixed(1)}%` : 'N/A'}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* SHAP Explanation */}
          <div className="border border-border rounded-lg p-6 bg-card">
            <ShapExplanation videoId={videoId!} />
          </div>

          {/* LLM AI Explanation */}
          <LLMExplanation videoId={videoId!} />

          {/* Labeling Controls */}
          <div className="border border-border rounded-lg p-6 bg-card">
            <h3 className="text-lg font-semibold mb-4 text-foreground">Label Video</h3>
            <p className="text-sm text-muted-foreground mb-4">
              Review the video and provide your expert label. 
              Keyboard shortcuts: <kbd className="px-1 bg-muted rounded">S</kbd> for Sound, 
              <kbd className="px-1 bg-muted rounded ml-1">L</kbd> for Lame.
            </p>
            <div className="flex gap-3">
              <button
                onClick={() => handleLabel(0)}
                disabled={labeling}
                className={`flex-1 px-4 py-3 rounded-lg font-medium transition-colors ${
                  currentLabel === 0
                    ? 'bg-success text-white'
                    : 'bg-success/20 text-success hover:bg-success/30'
                } disabled:opacity-50`}
              >
                {labeling ? '...' : '✓ Sound'}
              </button>
              <button
                onClick={() => handleLabel(1)}
                disabled={labeling}
                className={`flex-1 px-4 py-3 rounded-lg font-medium transition-colors ${
                  currentLabel === 1
                    ? 'bg-destructive text-white'
                    : 'bg-destructive/20 text-destructive hover:bg-destructive/30'
                } disabled:opacity-50`}
              >
                {labeling ? '...' : '✗ Lame'}
              </button>
            </div>
          </div>

          {/* Video Metadata */}
          <div className="border border-border rounded-lg p-6 bg-card">
            <h3 className="text-sm font-semibold mb-3 text-foreground">Video Details</h3>
            <dl className="space-y-2 text-sm">
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Video ID</dt>
                <dd className="font-mono text-xs">{videoId?.slice(0, 8)}...</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Duration</dt>
                <dd>{video.metadata?.duration?.toFixed(1) || '?'}s</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Resolution</dt>
                <dd>{video.metadata?.width || '?'}x{video.metadata?.height || '?'}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Frame Rate</dt>
                <dd>{video.metadata?.fps?.toFixed(1) || '?'} fps</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Total Frames</dt>
                <dd>{video.metadata?.frame_count || '?'}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Status</dt>
                <dd className={`capitalize ${
                  video.has_analysis ? 'text-success' : 'text-warning'
                }`}>
                  {video.status}
                </dd>
              </div>
            </dl>
          </div>
        </div>
      </div>
    </div>
  )
}
