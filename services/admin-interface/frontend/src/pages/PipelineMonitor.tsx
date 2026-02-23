/**
 * Pipeline Monitor Page
 * Real-time status and control of all ML pipeline services
 */
import { useState, useEffect, useCallback } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { usePipelineWebSocket } from '../hooks/useWebSocket'
import { cn } from '@/lib/utils'
import {
  Activity,
  CheckCircle,
  XCircle,
  AlertTriangle,
  HelpCircle,
  Play,
  RefreshCw,
  Loader2,
  ChevronDown,
  ChevronUp,
  Terminal,
  Clock,
  X
} from 'lucide-react'

interface PipelineStatus {
  service_name: string
  description: string
  status: 'healthy' | 'degraded' | 'down' | 'unknown'
  last_heartbeat: string | null
  active_jobs: number
  success_count: number
  error_count: number
  success_rate: number
  last_error: string | null
}

interface LogEntry {
  timestamp: string
  level: string
  service: string
  message: string
}

const API_URL = import.meta.env.VITE_API_URL || (import.meta.env.PROD ? '' : 'http://localhost:8000')

export default function PipelineMonitor() {
  const { getAccessToken, hasRole } = useAuth()
  const [pipelines, setPipelines] = useState<PipelineStatus[]>([])
  const [selectedPipeline, setSelectedPipeline] = useState<string | null>(null)
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [isLoadingLogs, setIsLoadingLogs] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [triggerVideoId, setTriggerVideoId] = useState('')
  const [triggerDialogOpen, setTriggerDialogOpen] = useState(false)
  const [triggerPipeline, setTriggerPipeline] = useState<string | null>(null)
  const [isTriggering, setIsTriggering] = useState(false)

  // WebSocket for real-time updates
  const { lastMessage, isConnected } = usePipelineWebSocket({
    onMessage: (message) => {
      if (message.type === 'pipeline_status') {
        setPipelines(prev => prev.map(p =>
          p.service_name === message.service
            ? { ...p, status: message.status as PipelineStatus['status'] }
            : p
        ))
      }
    }
  })

  const fetchPipelines = useCallback(async () => {
    try {
      const token = getAccessToken()
      const response = await fetch(`${API_URL}/api/pipeline/status`, {
        headers: { Authorization: `Bearer ${token}` }
      })

      if (!response.ok) throw new Error('Failed to fetch pipeline status')

      const data = await response.json()
      setPipelines(data)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load pipelines')
    } finally {
      setIsLoading(false)
    }
  }, [getAccessToken])

  const fetchLogs = useCallback(async (serviceName: string) => {
    setIsLoadingLogs(true)
    try {
      const token = getAccessToken()
      const response = await fetch(`${API_URL}/api/pipeline/${serviceName}/logs?limit=50`, {
        headers: { Authorization: `Bearer ${token}` }
      })

      if (!response.ok) throw new Error('Failed to fetch logs')

      const data = await response.json()
      setLogs(data)
    } catch (err) {
      console.error('Failed to fetch logs:', err)
      setLogs([])
    } finally {
      setIsLoadingLogs(false)
    }
  }, [getAccessToken])

  useEffect(() => {
    fetchPipelines()
    const interval = setInterval(fetchPipelines, 30000)
    return () => clearInterval(interval)
  }, [fetchPipelines])

  useEffect(() => {
    if (selectedPipeline) {
      fetchLogs(selectedPipeline)
    }
  }, [selectedPipeline, fetchLogs])

  const handleTrigger = async () => {
    if (!triggerPipeline || !triggerVideoId) return

    setIsTriggering(true)
    try {
      const token = getAccessToken()
      const response = await fetch(
        `${API_URL}/api/pipeline/${triggerPipeline}/trigger/${triggerVideoId}`,
        {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        }
      )

      if (!response.ok) throw new Error('Failed to trigger pipeline')

      setTriggerDialogOpen(false)
      setTriggerVideoId('')
      setTriggerPipeline(null)
      fetchPipelines()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to trigger pipeline')
    } finally {
      setIsTriggering(false)
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle className="h-5 w-5 text-emerald-500" />
      case 'degraded':
        return <AlertTriangle className="h-5 w-5 text-amber-500" />
      case 'down':
        return <XCircle className="h-5 w-5 text-red-500" />
      default:
        return <HelpCircle className="h-5 w-5 text-muted-foreground" />
    }
  }

  const getStatusStyles = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'border-emerald-500/50 bg-emerald-500/5'
      case 'degraded':
        return 'border-amber-500/50 bg-amber-500/5'
      case 'down':
        return 'border-red-500/50 bg-red-500/5'
      default:
        return 'border-border bg-muted/30'
    }
  }

  const formatTime = (timestamp: string | null) => {
    if (!timestamp) return 'Never'
    const date = new Date(timestamp)
    return date.toLocaleTimeString()
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="text-center animate-fade-in">
          <div className="relative inline-flex">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary to-primary/60 flex items-center justify-center animate-pulse-soft">
              <Activity className="h-8 w-8 text-primary-foreground" />
            </div>
            <div className="absolute -inset-2 bg-primary/20 rounded-3xl blur-xl animate-pulse-soft" />
          </div>
          <p className="mt-4 text-muted-foreground">Loading pipelines...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between animate-slide-in-up">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-primary to-primary/60 flex items-center justify-center shadow-lg shadow-primary/20">
            <Activity className="h-6 w-6 text-primary-foreground" />
          </div>
          <div>
            <h1 className="text-2xl font-bold">Pipeline Monitor</h1>
            <p className="text-muted-foreground">Real-time status of all ML pipeline services</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-sm">
            <div className={cn("w-2 h-2 rounded-full", isConnected ? 'bg-emerald-500 animate-pulse' : 'bg-red-500')} />
            <span className="text-muted-foreground">
              {isConnected ? 'Live updates' : 'Disconnected'}
            </span>
          </div>
          <button
            onClick={() => fetchPipelines()}
            className="flex items-center gap-2 px-3 py-2 rounded-xl hover:bg-accent/50 transition-colors text-sm"
          >
            <RefreshCw className="h-4 w-4" />
            Refresh
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-destructive/10 border border-destructive/30 text-destructive animate-scale-in">
          {error}
        </div>
      )}

      {/* Pipeline Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
        {pipelines.map((pipeline, i) => (
          <div
            key={pipeline.service_name}
            className={cn(
              "premium-card border-2 cursor-pointer transition-all hover:shadow-lg animate-slide-in-up",
              getStatusStyles(pipeline.status),
              selectedPipeline === pipeline.service_name && 'ring-2 ring-primary'
            )}
            style={{ animationDelay: `${i * 0.05}s`, animationFillMode: 'backwards' }}
            onClick={() => setSelectedPipeline(
              selectedPipeline === pipeline.service_name ? null : pipeline.service_name
            )}
          >
            <div className="flex items-start justify-between mb-2">
              {getStatusIcon(pipeline.status)}
              {hasRole(['admin', 'researcher']) && (
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    setTriggerPipeline(pipeline.service_name)
                    setTriggerDialogOpen(true)
                  }}
                  className="p-1.5 hover:bg-accent/50 rounded-lg transition-colors"
                  title="Trigger pipeline"
                >
                  <Play className="h-4 w-4 text-muted-foreground" />
                </button>
              )}
            </div>

            <h3 className="font-semibold text-sm truncate">
              {pipeline.service_name}
            </h3>
            <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
              {pipeline.description}
            </p>

            <div className="mt-3 pt-3 border-t border-border/50 space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-muted-foreground">Active Jobs</span>
                <span className="font-medium">{pipeline.active_jobs}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-muted-foreground">Success Rate</span>
                <span className="font-medium">
                  {(pipeline.success_rate * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-muted-foreground">Last Heartbeat</span>
                <span className="font-medium">{formatTime(pipeline.last_heartbeat)}</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Logs Panel */}
      {selectedPipeline && (
        <div className="premium-card p-0 overflow-hidden animate-scale-in">
          <div className="flex items-center justify-between p-4 border-b border-border/50">
            <div className="flex items-center gap-2">
              <Terminal className="h-5 w-5 text-muted-foreground" />
              <h3 className="font-semibold">{selectedPipeline} Logs</h3>
            </div>
            <button
              onClick={() => setSelectedPipeline(null)}
              className="p-2 hover:bg-accent/50 rounded-lg transition-colors"
            >
              <ChevronUp className="h-5 w-5" />
            </button>
          </div>

          <div className="p-4 bg-card rounded-b-xl max-h-80 overflow-auto font-mono text-sm">
            {isLoadingLogs ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            ) : logs.length > 0 ? (
              logs.map((log, index) => (
                <div key={index} className="py-1">
                  <span className="text-muted-foreground">
                    {new Date(log.timestamp).toLocaleTimeString()}
                  </span>
                  <span className={cn(
                    "ml-2",
                    log.level === 'ERROR' ? 'text-red-500' :
                    log.level === 'WARNING' ? 'text-amber-500' :
                    'text-emerald-500'
                  )}>
                    [{log.level}]
                  </span>
                  <span className="ml-2 text-foreground">{log.message}</span>
                </div>
              ))
            ) : (
              <p className="text-muted-foreground text-center py-8">No logs available</p>
            )}
          </div>
        </div>
      )}

      {/* Trigger Dialog */}
      {triggerDialogOpen && (
        <>
          <div
            className="fixed inset-0 bg-background/80 backdrop-blur-sm z-50 animate-fade-in"
            onClick={() => {
              setTriggerDialogOpen(false)
              setTriggerVideoId('')
              setTriggerPipeline(null)
            }}
          />
          <div className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-md z-50 animate-scale-in mx-4">
            <div className="bg-card rounded-2xl shadow-2xl border border-border/50 p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold">
                  Trigger Pipeline: {triggerPipeline}
                </h3>
                <button
                  onClick={() => {
                    setTriggerDialogOpen(false)
                    setTriggerVideoId('')
                    setTriggerPipeline(null)
                  }}
                  className="p-2 hover:bg-accent/50 rounded-lg transition-colors"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>
              <div className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Video ID</label>
                  <input
                    type="text"
                    value={triggerVideoId}
                    onChange={(e) => setTriggerVideoId(e.target.value)}
                    placeholder="Enter video ID"
                    className="input-premium"
                  />
                </div>
                <div className="flex justify-end gap-3 pt-2">
                  <button
                    onClick={() => {
                      setTriggerDialogOpen(false)
                      setTriggerVideoId('')
                      setTriggerPipeline(null)
                    }}
                    className="px-4 py-2 rounded-xl border border-border hover:bg-accent/50 transition-colors font-medium"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleTrigger}
                    disabled={!triggerVideoId || isTriggering}
                    className="btn-premium flex items-center gap-2"
                  >
                    {isTriggering ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Play className="h-4 w-4" />
                    )}
                    Trigger
                  </button>
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* Stats Summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: 'Healthy', value: pipelines.filter(p => p.status === 'healthy').length, icon: CheckCircle, color: 'text-emerald-500', bgColor: 'bg-emerald-500/10' },
          { label: 'Degraded', value: pipelines.filter(p => p.status === 'degraded').length, icon: AlertTriangle, color: 'text-amber-500', bgColor: 'bg-amber-500/10' },
          { label: 'Down', value: pipelines.filter(p => p.status === 'down').length, icon: XCircle, color: 'text-red-500', bgColor: 'bg-red-500/10' },
          { label: 'Active Jobs', value: pipelines.reduce((sum, p) => sum + p.active_jobs, 0), icon: Activity, color: 'text-muted-foreground', bgColor: 'bg-muted' },
        ].map((stat, i) => (
          <div
            key={stat.label}
            className="premium-card animate-slide-in-up"
            style={{ animationDelay: `${0.5 + i * 0.05}s`, animationFillMode: 'backwards' }}
          >
            <div className={cn("flex items-center gap-2 mb-2", stat.color)}>
              <div className={cn("w-8 h-8 rounded-lg flex items-center justify-center", stat.bgColor)}>
                <stat.icon className="h-4 w-4" />
              </div>
              <span className="text-sm font-medium">{stat.label}</span>
            </div>
            <p className="text-2xl font-bold">{stat.value}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
