import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { trainingApi } from '@/api/client'

export default function TrainingQueue() {
  const [queue, setQueue] = useState<any[]>([])
  const [stats, setStats] = useState<any>(null)
  const [status, setStatus] = useState<any>(null)
  const [models, setModels] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [training, setTraining] = useState(false)

  useEffect(() => {
    loadData()
    // Refresh status every 10 seconds
    const interval = setInterval(loadStatus, 10000)
    return () => clearInterval(interval)
  }, [])

  const loadData = async () => {
    try {
      const [queueData, statsData, statusData, modelsData] = await Promise.all([
        trainingApi.getQueue().catch(() => ({ videos: [] })),
        trainingApi.getStats().catch(() => null),
        trainingApi.getStatus().catch(() => null),
        trainingApi.getModels().catch(() => ({ models: [] })),
      ])
      setQueue(queueData.videos || [])
      setStats(statsData)
      setStatus(statusData)
      setModels(modelsData.models || [])
    } catch (error) {
      console.error('Failed to load data:', error)
    } finally {
      setLoading(false)
    }
  }

  const loadStatus = async () => {
    try {
      const [statusData, modelsData] = await Promise.all([
        trainingApi.getStatus().catch(() => null),
        trainingApi.getModels().catch(() => ({ models: [] })),
      ])
      setStatus(statusData)
      setModels(modelsData.models || [])
    } catch (error) {
      console.error('Failed to load status:', error)
    }
  }

  const handleStartTraining = async () => {
    setTraining(true)
    try {
      await trainingApi.startMLTraining()
      alert('Training started! Check back in a few moments.')
      // Refresh status after a delay
      setTimeout(loadStatus, 2000)
    } catch (error: any) {
      alert(error.response?.data?.detail || 'Failed to start training')
    } finally {
      setTraining(false)
    }
  }

  if (loading) {
    return <div className="text-center py-8">Loading...</div>
  }

  const isTraining = status?.status === 'training'
  const canTrain = stats?.ready_for_training && !isTraining

  return (
    <div>
      <h2 className="text-3xl font-bold mb-6">Training</h2>

      {/* Training Stats */}
      {stats && (
        <div className="grid gap-4 md:grid-cols-5 mb-6">
          <div className="border border-border rounded-lg p-4 bg-card">
            <p className="text-sm text-muted-foreground">Total Labels</p>
            <p className="text-2xl font-bold text-foreground">{stats.total_labels}</p>
          </div>
          <div className="border border-success/30 rounded-lg p-4 bg-success/10">
            <p className="text-sm text-success">Sound</p>
            <p className="text-2xl font-bold text-success">{stats.sound_count}</p>
          </div>
          <div className="border border-destructive/30 rounded-lg p-4 bg-destructive/10">
            <p className="text-sm text-destructive">Lame</p>
            <p className="text-2xl font-bold text-destructive">{stats.lame_count}</p>
          </div>
          <div className="border border-border rounded-lg p-4 bg-card">
            <p className="text-sm text-muted-foreground">Balance</p>
            <p className="text-2xl font-bold text-foreground">
              {stats.balance_ratio ? stats.balance_ratio.toFixed(2) : 'N/A'}
            </p>
          </div>
          <div className="border border-border rounded-lg p-4 bg-card">
            <p className="text-sm text-muted-foreground">Status</p>
            <p className={`text-2xl font-bold ${
              stats.ready_for_training ? 'text-success' : 'text-warning'
            }`}>
              {stats.ready_for_training ? 'Ready' : 'Need Data'}
            </p>
          </div>
        </div>
      )}

      {/* Training Status & Controls */}
      <div className="grid gap-6 md:grid-cols-2 mb-6">
        <div className="border border-border rounded-lg p-6 bg-card">
          <h3 className="text-xl font-semibold mb-4 text-foreground">Training Status</h3>
          
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Status:</span>
              <span className={`font-medium ${
                status?.status === 'completed' ? 'text-success' :
                status?.status === 'training' ? 'text-primary' :
                status?.status === 'failed' ? 'text-destructive' :
                'text-muted-foreground'
              }`}>
                {status?.status || 'idle'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Last Trained:</span>
              <span className="font-medium">
                {status?.last_trained 
                  ? new Date(status.last_trained).toLocaleString()
                  : 'Never'
                }
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Samples Used:</span>
              <span className="font-medium">{status?.samples_used || 0}</span>
            </div>
          </div>

          <div className="mt-6">
            <button
              onClick={handleStartTraining}
              disabled={!canTrain || training}
              className="w-full px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 disabled:opacity-50"
            >
              {training ? 'Starting...' : isTraining ? 'Training in Progress...' : 'Start Training'}
            </button>
            {!stats?.ready_for_training && (
              <p className="text-sm text-muted-foreground mt-2">
                Need at least 10 samples with both Sound and Lame labels
              </p>
            )}
          </div>
        </div>

        {/* Trained Models */}
        <div className="border border-border rounded-lg p-6 bg-card">
          <h3 className="text-xl font-semibold mb-4 text-foreground">Trained Models</h3>
          
          {models.length === 0 ? (
            <p className="text-muted-foreground">No models trained yet</p>
          ) : (
            <div className="space-y-2">
              {models.map((model) => (
                <div key={model.name} className="flex justify-between p-2 bg-secondary/50 rounded">
                  <span className="font-medium capitalize">{model.name.replace('_model', '')}</span>
                  <span className="text-sm text-muted-foreground">
                    {model.size_kb.toFixed(1)} KB
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Model Metrics */}
          {status?.metrics && Object.keys(status.metrics).length > 0 && (
            <div className="mt-4 pt-4 border-t border-border">
              <h4 className="font-medium mb-2 text-foreground">Model Performance</h4>
              <div className="space-y-1 text-sm">
                {Object.entries(status.metrics).map(([model, metrics]: [string, any]) => (
                  !metrics.error && (
                    <div key={model} className="flex justify-between">
                      <span className="capitalize">{model}:</span>
                      <span>
                        {metrics.cv_accuracy_mean 
                          ? `CV: ${(metrics.cv_accuracy_mean * 100).toFixed(1)}%`
                          : `Acc: ${(metrics.train_accuracy * 100).toFixed(1)}%`
                        }
                      </span>
                    </div>
                  )
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Training Info */}
      <div className="border rounded-lg p-4 mb-6 bg-warning/10 border-warning/30">
        <div className="flex items-start gap-3">
          <div className="text-warning text-xl">📋</div>
          <div>
            <h4 className="font-medium text-warning">Manual Training</h4>
            <p className="text-sm text-warning/80">
              Click "Start Training" when you're ready to train models. You need at least 10 labeled videos 
              with both Sound and Lame samples. Training uses CatBoost, XGBoost, LightGBM, and creates an ensemble model.
            </p>
          </div>
        </div>
      </div>

      {/* Videos Needing Labels */}
      <div className="border border-border rounded-lg bg-card">
        <div className="p-4 border-b border-border">
          <h3 className="font-semibold text-foreground">Videos Needing Labels ({queue.length})</h3>
          <p className="text-sm text-muted-foreground">
            Prioritized by prediction uncertainty (most uncertain first)
          </p>
        </div>
        <div className="divide-y divide-border max-h-96 overflow-y-auto">
          {queue.length === 0 ? (
            <div className="p-8 text-center text-muted-foreground">
              No videos in queue. Upload and analyze videos first.
            </div>
          ) : (
            queue.map((video) => (
              <Link
                key={video.video_id}
                to={`/video/${video.video_id}`}
                className="block p-4 hover:bg-accent transition-colors"
              >
                <div className="flex justify-between items-center">
                  <div>
                    <p className="font-medium">Video {video.video_id.slice(0, 8)}...</p>
                    <p className="text-sm text-muted-foreground">
                      Predicted: {(video.predicted_probability * 100).toFixed(1)}% lame
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`px-2 py-1 text-xs rounded ${
                      video.uncertainty < 0.2 ? 'bg-warning/20 text-warning' : 'bg-muted text-muted-foreground'
                    }`}>
                      {video.uncertainty < 0.2 ? 'Uncertain' : 'Confident'}
                    </span>
                    <span className="text-muted-foreground">→</span>
                  </div>
                </div>
              </Link>
            ))
          )}
        </div>
      </div>
    </div>
  )
}
