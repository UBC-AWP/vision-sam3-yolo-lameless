/**
 * System Health Page
 * Infrastructure monitoring dashboard for Docker, NATS, databases, and disk usage
 */
import { useState, useEffect, useCallback } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { useHealthWebSocket } from '../hooks/useWebSocket'
import { cn } from '@/lib/utils'
import {
  Server,
  Database,
  HardDrive,
  Activity,
  CheckCircle,
  XCircle,
  AlertTriangle,
  RefreshCw,
  Loader2,
  Cpu,
  MemoryStick,
  Clock,
  Zap,
  TrendingUp,
  Heart
} from 'lucide-react'

interface HealthOverview {
  status: 'healthy' | 'degraded' | 'critical'
  timestamp: string
  components: Record<string, string>
  issues: string[]
}

interface ContainerHealth {
  name: string
  status: string
  cpu_percent: number | null
  memory_mb: number | null
  memory_percent: number | null
  uptime: string | null
}

interface NATSHealth {
  status: string
  connections: number
  subscriptions: number
  messages_in: number
  messages_out: number
  bytes_in: number
  bytes_out: number
}

interface DatabaseHealth {
  status: string
  connection_count: number
  database_size_mb: number
  response_time_ms: number
}

interface DiskUsage {
  path: string
  total_gb: number
  used_gb: number
  free_gb: number
  percent_used: number
  status: string
}

interface ThroughputMetrics {
  videos_processed_24h: number
  videos_processed_7d: number
  avg_processing_time_s: number
  success_rate: number
  queue_depth: number
}

const API_URL = import.meta.env.VITE_API_URL || (import.meta.env.PROD ? '' : 'http://localhost:8000')

export default function SystemHealth() {
  const { getAccessToken } = useAuth()
  const [overview, setOverview] = useState<HealthOverview | null>(null)
  const [containers, setContainers] = useState<ContainerHealth[]>([])
  const [nats, setNats] = useState<NATSHealth | null>(null)
  const [postgres, setPostgres] = useState<DatabaseHealth | null>(null)
  const [qdrant, setQdrant] = useState<DatabaseHealth | null>(null)
  const [disk, setDisk] = useState<DiskUsage[]>([])
  const [throughput, setThroughput] = useState<ThroughputMetrics | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // WebSocket for real-time health updates
  const { isConnected } = useHealthWebSocket({
    onMessage: (message) => {
      if (message.type === 'health_update') {
        fetchAll()
      }
    }
  })

  const fetchAll = useCallback(async () => {
    const token = getAccessToken()
    const headers = { Authorization: `Bearer ${token}` }

    try {
      const [overviewRes, containersRes, natsRes, postgresRes, qdrantRes, diskRes, throughputRes] = await Promise.all([
        fetch(`${API_URL}/api/health/overview`, { headers }),
        fetch(`${API_URL}/api/health/docker`, { headers }),
        fetch(`${API_URL}/api/health/nats`, { headers }),
        fetch(`${API_URL}/api/health/postgres`, { headers }),
        fetch(`${API_URL}/api/health/qdrant`, { headers }),
        fetch(`${API_URL}/api/health/disk`, { headers }),
        fetch(`${API_URL}/api/health/throughput`, { headers })
      ])

      if (overviewRes.ok) setOverview(await overviewRes.json())
      if (containersRes.ok) setContainers(await containersRes.json())
      if (natsRes.ok) setNats(await natsRes.json())
      if (postgresRes.ok) setPostgres(await postgresRes.json())
      if (qdrantRes.ok) setQdrant(await qdrantRes.json())
      if (diskRes.ok) setDisk(await diskRes.json())
      if (throughputRes.ok) setThroughput(await throughputRes.json())

      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load health data')
    } finally {
      setIsLoading(false)
    }
  }, [getAccessToken])

  useEffect(() => {
    fetchAll()
    const interval = setInterval(fetchAll, 30000)
    return () => clearInterval(interval)
  }, [fetchAll])

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle className="h-5 w-5 text-emerald-500" />
      case 'warning':
      case 'degraded':
        return <AlertTriangle className="h-5 w-5 text-amber-500" />
      case 'critical':
      case 'down':
        return <XCircle className="h-5 w-5 text-red-500" />
      default:
        return <Activity className="h-5 w-5 text-muted-foreground" />
    }
  }

  const getStatusBg = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'bg-emerald-500'
      case 'warning':
      case 'degraded':
        return 'bg-amber-500'
      case 'critical':
      case 'down':
        return 'bg-red-500'
      default:
        return 'bg-muted-foreground'
    }
  }

  const formatBytes = (bytes: number) => {
    if (bytes >= 1073741824) return `${(bytes / 1073741824).toFixed(2)} GB`
    if (bytes >= 1048576) return `${(bytes / 1048576).toFixed(2)} MB`
    if (bytes >= 1024) return `${(bytes / 1024).toFixed(2)} KB`
    return `${bytes} B`
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="text-center animate-fade-in">
          <div className="relative inline-flex">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary to-primary/60 flex items-center justify-center animate-pulse-soft">
              <Heart className="h-8 w-8 text-primary-foreground" />
            </div>
            <div className="absolute -inset-2 bg-primary/20 rounded-3xl blur-xl animate-pulse-soft" />
          </div>
          <p className="mt-4 text-muted-foreground">Loading system health...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between animate-slide-in-up">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-emerald-500 to-emerald-600 flex items-center justify-center shadow-lg shadow-emerald-500/20">
            <Heart className="h-6 w-6 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold">System Health</h1>
            <p className="text-muted-foreground">Infrastructure monitoring and system status</p>
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
            onClick={() => fetchAll()}
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

      {/* Overview Card */}
      {overview && (
        <div className={cn(
          "premium-card border-2 animate-slide-in-up",
          overview.status === 'healthy' ? 'border-emerald-500/50 bg-emerald-500/5' :
          overview.status === 'degraded' ? 'border-amber-500/50 bg-amber-500/5' :
          'border-red-500/50 bg-red-500/5'
        )} style={{ animationDelay: '0.1s', animationFillMode: 'backwards' }}>
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div className="flex items-center gap-3">
              {getStatusIcon(overview.status)}
              <div>
                <h2 className="text-lg font-semibold capitalize">{overview.status}</h2>
                <p className="text-sm text-muted-foreground">
                  Last updated: {new Date(overview.timestamp).toLocaleString()}
                </p>
              </div>
            </div>
            <div className="flex gap-4 flex-wrap">
              {Object.entries(overview.components).map(([name, status]) => (
                <div key={name} className="flex items-center gap-2">
                  <div className={cn("w-2 h-2 rounded-full", getStatusBg(status))} />
                  <span className="text-sm capitalize">{name}</span>
                </div>
              ))}
            </div>
          </div>
          {overview.issues.length > 0 && (
            <div className="mt-4 pt-4 border-t border-border/50">
              <h4 className="text-sm font-medium mb-2">Issues</h4>
              <ul className="space-y-1">
                {overview.issues.map((issue, i) => (
                  <li key={i} className="text-sm text-muted-foreground flex items-center gap-2">
                    <AlertTriangle className="h-3 w-3 text-amber-500" />
                    {issue}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Services Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* NATS Card */}
        <div className="premium-card animate-slide-in-up" style={{ animationDelay: '0.2s', animationFillMode: 'backwards' }}>
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-violet-500/10 flex items-center justify-center">
                <Zap className="h-4 w-4 text-violet-500" />
              </div>
              <h3 className="font-semibold">NATS Messaging</h3>
            </div>
            {nats && getStatusIcon(nats.status)}
          </div>
          {nats && (
            <div className="space-y-3">
              {[
                { label: 'Connections', value: nats.connections },
                { label: 'Subscriptions', value: nats.subscriptions },
                { label: 'Messages In', value: nats.messages_in.toLocaleString() },
                { label: 'Messages Out', value: nats.messages_out.toLocaleString() },
                { label: 'Bytes In', value: formatBytes(nats.bytes_in) },
              ].map((item) => (
                <div key={item.label} className="flex justify-between text-sm">
                  <span className="text-muted-foreground">{item.label}</span>
                  <span className="font-medium">{item.value}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* PostgreSQL Card */}
        <div className="premium-card animate-slide-in-up" style={{ animationDelay: '0.3s', animationFillMode: 'backwards' }}>
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-blue-500/10 flex items-center justify-center">
                <Database className="h-4 w-4 text-blue-500" />
              </div>
              <h3 className="font-semibold">PostgreSQL</h3>
            </div>
            {postgres && getStatusIcon(postgres.status)}
          </div>
          {postgres && (
            <div className="space-y-3">
              {[
                { label: 'Connections', value: postgres.connection_count },
                { label: 'Database Size', value: `${postgres.database_size_mb.toFixed(2)} MB` },
                { label: 'Response Time', value: `${postgres.response_time_ms.toFixed(2)} ms` },
              ].map((item) => (
                <div key={item.label} className="flex justify-between text-sm">
                  <span className="text-muted-foreground">{item.label}</span>
                  <span className="font-medium">{item.value}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Qdrant Card */}
        <div className="premium-card animate-slide-in-up" style={{ animationDelay: '0.4s', animationFillMode: 'backwards' }}>
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-orange-500/10 flex items-center justify-center">
                <Database className="h-4 w-4 text-orange-500" />
              </div>
              <h3 className="font-semibold">Qdrant Vector DB</h3>
            </div>
            {qdrant && getStatusIcon(qdrant.status)}
          </div>
          {qdrant && (
            <div className="space-y-3">
              {[
                { label: 'Status', value: qdrant.status, capitalize: true },
                { label: 'Est. Size', value: `${qdrant.database_size_mb.toFixed(2)} MB` },
                { label: 'Response Time', value: `${qdrant.response_time_ms.toFixed(2)} ms` },
              ].map((item) => (
                <div key={item.label} className="flex justify-between text-sm">
                  <span className="text-muted-foreground">{item.label}</span>
                  <span className={cn("font-medium", item.capitalize && "capitalize")}>{item.value}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Disk Usage */}
      <div className="premium-card animate-slide-in-up" style={{ animationDelay: '0.5s', animationFillMode: 'backwards' }}>
        <div className="flex items-center gap-2 mb-4">
          <div className="w-8 h-8 rounded-lg bg-muted flex items-center justify-center">
            <HardDrive className="h-4 w-4 text-muted-foreground" />
          </div>
          <h3 className="font-semibold">Disk Usage</h3>
        </div>
        <div className="space-y-4">
          {disk.map((d) => (
            <div key={d.path}>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">{d.path}</span>
                <span className="text-sm text-muted-foreground">
                  {d.used_gb.toFixed(2)} / {d.total_gb.toFixed(2)} GB ({d.percent_used.toFixed(1)}%)
                </span>
              </div>
              <div className="h-2 bg-muted rounded-full overflow-hidden">
                <div
                  className={cn(
                    "h-full rounded-full transition-all",
                    d.status === 'critical' ? 'bg-red-500' :
                    d.status === 'warning' ? 'bg-amber-500' :
                    'bg-emerald-500'
                  )}
                  style={{ width: `${Math.min(d.percent_used, 100)}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Throughput Metrics */}
      {throughput && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { label: '24h Processed', value: throughput.videos_processed_24h, icon: TrendingUp, color: 'text-blue-500', bgColor: 'bg-blue-500/10' },
            { label: '7d Processed', value: throughput.videos_processed_7d, icon: TrendingUp, color: 'text-violet-500', bgColor: 'bg-violet-500/10' },
            { label: 'Success Rate', value: `${(throughput.success_rate * 100).toFixed(1)}%`, icon: CheckCircle, color: 'text-emerald-500', bgColor: 'bg-emerald-500/10' },
            { label: 'Queue Depth', value: throughput.queue_depth, icon: Clock, color: 'text-orange-500', bgColor: 'bg-orange-500/10' },
          ].map((metric, i) => (
            <div
              key={metric.label}
              className="premium-card animate-slide-in-up"
              style={{ animationDelay: `${0.6 + i * 0.05}s`, animationFillMode: 'backwards' }}
            >
              <div className={cn("flex items-center gap-2 mb-2", metric.color)}>
                <div className={cn("w-8 h-8 rounded-lg flex items-center justify-center", metric.bgColor)}>
                  <metric.icon className="h-4 w-4" />
                </div>
                <span className="text-sm font-medium">{metric.label}</span>
              </div>
              <p className="text-2xl font-bold">{metric.value}</p>
            </div>
          ))}
        </div>
      )}

      {/* Containers Table */}
      <div className="premium-card p-0 overflow-hidden animate-slide-in-up" style={{ animationDelay: '0.8s', animationFillMode: 'backwards' }}>
        <div className="p-4 border-b border-border/50">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-muted flex items-center justify-center">
              <Server className="h-4 w-4 text-muted-foreground" />
            </div>
            <h3 className="font-semibold">Docker Containers</h3>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="premium-table">
            <thead>
              <tr>
                <th>Container</th>
                <th>Status</th>
                <th>CPU</th>
                <th>Memory</th>
                <th>Uptime</th>
              </tr>
            </thead>
            <tbody>
              {containers.map((container, i) => (
                <tr
                  key={container.name}
                  className="animate-fade-in"
                  style={{ animationDelay: `${i * 0.03}s`, animationFillMode: 'backwards' }}
                >
                  <td className="font-medium">{container.name}</td>
                  <td>
                    <span className={cn(
                      "inline-flex items-center gap-1.5 px-2 py-1 rounded-lg text-xs font-medium",
                      container.status === 'running' ? 'bg-emerald-500/15 text-emerald-500' :
                      container.status === 'paused' ? 'bg-amber-500/15 text-amber-500' :
                      'bg-red-500/15 text-red-500'
                    )}>
                      <div className={cn(
                        "w-1.5 h-1.5 rounded-full",
                        container.status === 'running' ? 'bg-emerald-500' :
                        container.status === 'paused' ? 'bg-amber-500' :
                        'bg-red-500'
                      )} />
                      {container.status}
                    </span>
                  </td>
                  <td className="text-muted-foreground">
                    {container.cpu_percent !== null ? `${container.cpu_percent.toFixed(1)}%` : '-'}
                  </td>
                  <td className="text-muted-foreground">
                    {container.memory_mb !== null ? `${container.memory_mb.toFixed(0)} MB` : '-'}
                  </td>
                  <td className="text-muted-foreground">
                    {container.uptime || '-'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
