/**
 * Dashboard Page
 * Premium overview with modern metric cards and data visualization
 */
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { cn } from '@/lib/utils'
import { videosApi, trainingApi } from '@/api/client'
import {
  TrendingUp,
  TrendingDown,
  Video,
  BarChart3,
  Tag,
  CheckCircle2,
  XCircle,
  Clock,
  ArrowRight,
  Sparkles,
  Activity,
  Target,
  Zap,
  Loader2,
  Upload,
  PlayCircle,
  AlertCircle,
  ChevronRight
} from 'lucide-react'

export default function Dashboard() {
  const [videos, setVideos] = useState<any[]>([])
  const [stats, setStats] = useState<any>(null)
  const [trainingStatus, setTrainingStatus] = useState<any>(null)
  const [pairwiseStats, setPairwiseStats] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    try {
      const [videoData, statsData, statusData, pairwiseData] = await Promise.all([
        videosApi.list(0, 1000),
        trainingApi.getStats().catch(() => null),
        trainingApi.getStatus().catch(() => null),
        trainingApi.getPairwiseStats().catch(() => null)
      ])
      setVideos(videoData.videos || [])
      setStats(statsData)
      setTrainingStatus(statusData)
      setPairwiseStats(pairwiseData)
    } catch (error) {
      console.error('Failed to load data:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="text-center animate-fade-in">
          <div className="relative inline-flex">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary to-primary/60 flex items-center justify-center animate-pulse-soft">
              <Activity className="h-8 w-8 text-primary-foreground" />
            </div>
            <div className="absolute -inset-2 bg-primary/20 rounded-3xl blur-xl animate-pulse-soft" />
          </div>
          <p className="mt-4 text-muted-foreground">Loading dashboard...</p>
        </div>
      </div>
    )
  }

  const labeledCount = videos.filter(v => v.has_label).length
  const soundCount = videos.filter(v => v.label === 0).length
  const lameCount = videos.filter(v => v.label === 1).length
  const analyzedCount = videos.filter(v => v.has_analysis).length
  const annotatedCount = videos.filter(v => v.has_annotated).length
  const pendingCount = videos.filter(v => !v.has_label).length

  const metrics = [
    {
      label: 'Total Videos',
      value: videos.length,
      icon: Video,
      trend: null,
      color: 'from-blue-500 to-blue-600',
      bgColor: 'bg-blue-500/10',
      textColor: 'text-blue-500',
    },
    {
      label: 'Analyzed',
      value: analyzedCount,
      icon: BarChart3,
      trend: videos.length > 0 ? Math.round((analyzedCount / videos.length) * 100) : 0,
      trendLabel: '% complete',
      color: 'from-violet-500 to-violet-600',
      bgColor: 'bg-violet-500/10',
      textColor: 'text-violet-500',
    },
    {
      label: 'Labeled',
      value: labeledCount,
      icon: Tag,
      trend: videos.length > 0 ? Math.round((labeledCount / videos.length) * 100) : 0,
      trendLabel: '% complete',
      color: 'from-amber-500 to-amber-600',
      bgColor: 'bg-amber-500/10',
      textColor: 'text-amber-500',
    },
    {
      label: 'Sound',
      value: soundCount,
      icon: CheckCircle2,
      trend: null,
      color: 'from-emerald-500 to-emerald-600',
      bgColor: 'bg-emerald-500/10',
      textColor: 'text-emerald-500',
    },
    {
      label: 'Lame',
      value: lameCount,
      icon: XCircle,
      trend: null,
      color: 'from-rose-500 to-rose-600',
      bgColor: 'bg-rose-500/10',
      textColor: 'text-rose-500',
    },
    {
      label: 'Pending',
      value: pendingCount,
      icon: Clock,
      trend: null,
      color: 'from-slate-500 to-slate-600',
      bgColor: 'bg-slate-500/10',
      textColor: 'text-slate-500',
    },
  ]

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start gap-4">
        <div className="animate-slide-in-up">
          <h1 className="text-3xl font-bold flex items-center gap-3">
            Dashboard
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-lg bg-primary/10 text-primary text-sm font-medium">
              <Sparkles className="h-3.5 w-3.5" />
              Live
            </span>
          </h1>
          <p className="text-muted-foreground mt-1">
            Overview of your lameness detection research pipeline
          </p>
        </div>
        <Link
          to="/upload"
          className="btn-premium inline-flex items-center gap-2 animate-slide-in-up"
          style={{ animationDelay: '0.1s' }}
        >
          <Upload className="h-4 w-4" />
          Upload Videos
        </Link>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        {metrics.map((metric, i) => (
          <div
            key={metric.label}
            className="premium-card group animate-slide-in-up"
            style={{ animationDelay: `${i * 0.05}s`, animationFillMode: 'backwards' }}
          >
            <div className="flex items-start justify-between mb-3">
              <div className={cn(
                "w-10 h-10 rounded-xl flex items-center justify-center",
                metric.bgColor
              )}>
                <metric.icon className={cn("h-5 w-5", metric.textColor)} />
              </div>
              {metric.trend !== null && (
                <span className="text-xs text-muted-foreground flex items-center gap-1">
                  <TrendingUp className="h-3 w-3 text-emerald-500" />
                  {metric.trend}{metric.trendLabel}
                </span>
              )}
            </div>
            <p className="text-3xl font-bold gradient-text">{metric.value}</p>
            <p className="text-sm text-muted-foreground mt-0.5">{metric.label}</p>
          </div>
        ))}
      </div>

      {/* Main Content Grid */}
      <div className="grid lg:grid-cols-3 gap-6">
        {/* Pairwise Comparison Progress */}
        <div className="premium-card animate-slide-in-up" style={{ animationDelay: '0.3s', animationFillMode: 'backwards' }}>
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center shadow-lg shadow-blue-500/20">
              <Target className="h-5 w-5 text-white" />
            </div>
            <div>
              <h3 className="font-semibold">Pairwise Comparisons</h3>
              <p className="text-xs text-muted-foreground">Human annotation progress</p>
            </div>
          </div>
          
          {pairwiseStats ? (
            <>
              <div className="mb-4">
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-muted-foreground">Progress</span>
                  <span className="font-medium">
                    {pairwiseStats.pairs_compared} / {pairwiseStats.total_possible_pairs}
                  </span>
                </div>
                <div className="h-2 bg-muted rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-blue-500 to-blue-600 rounded-full transition-all duration-500"
                    style={{ width: `${pairwiseStats.completion_rate * 100}%` }}
                  />
                </div>
              </div>
              <p className="text-sm text-muted-foreground mb-4">
                <span className="font-medium text-foreground">{pairwiseStats.total_comparisons}</span> total comparisons made
              </p>
              <Link
                to="/pairwise"
                className="flex items-center justify-between p-3 rounded-xl bg-muted/50 hover:bg-muted transition-colors group"
              >
                <span className="text-sm font-medium">Continue Comparing</span>
                <ArrowRight className="h-4 w-4 text-muted-foreground group-hover:translate-x-1 transition-transform" />
              </Link>
            </>
          ) : (
            <div className="text-center py-6">
              <div className="w-12 h-12 rounded-full bg-muted mx-auto flex items-center justify-center mb-3">
                <Target className="h-6 w-6 text-muted-foreground" />
              </div>
              <p className="text-sm text-muted-foreground">
                Upload at least 2 videos to start pairwise comparisons
              </p>
            </div>
          )}
        </div>

        {/* Training Status */}
        <div className="premium-card animate-slide-in-up" style={{ animationDelay: '0.4s', animationFillMode: 'backwards' }}>
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-violet-600 flex items-center justify-center shadow-lg shadow-violet-500/20">
              <Zap className="h-5 w-5 text-white" />
            </div>
            <div>
              <h3 className="font-semibold">Training Status</h3>
              <p className="text-xs text-muted-foreground">ML model training</p>
            </div>
          </div>
          
          {trainingStatus ? (
            <>
              <div className="space-y-3 mb-4">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-muted-foreground">Status</span>
                  <span className={cn(
                    "px-2.5 py-1 rounded-lg text-xs font-medium capitalize",
                    trainingStatus.status === 'completed' 
                      ? 'bg-emerald-500/15 text-emerald-500' 
                      : trainingStatus.status === 'training' 
                        ? 'bg-blue-500/15 text-blue-500' 
                        : 'bg-muted text-muted-foreground'
                  )}>
                    {trainingStatus.status}
                  </span>
                </div>
                {trainingStatus.last_trained && (
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Last Trained</span>
                    <span className="font-medium">
                      {new Date(trainingStatus.last_trained).toLocaleDateString()}
                    </span>
                  </div>
                )}
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Samples Used</span>
                  <span className="font-medium">{trainingStatus.samples_used}</span>
                </div>
              </div>
              <Link
                to="/training"
                className="flex items-center justify-between p-3 rounded-xl bg-muted/50 hover:bg-muted transition-colors group"
              >
                <span className="text-sm font-medium">Manage Training</span>
                <ArrowRight className="h-4 w-4 text-muted-foreground group-hover:translate-x-1 transition-transform" />
              </Link>
            </>
          ) : (
            <div className="text-center py-6">
              <div className="w-12 h-12 rounded-full bg-muted mx-auto flex items-center justify-center mb-3">
                <Zap className="h-6 w-6 text-muted-foreground" />
              </div>
              <p className="text-sm text-muted-foreground">
                Training status unavailable
              </p>
            </div>
          )}
        </div>

        {/* Needs Attention */}
        <div className="premium-card animate-slide-in-up" style={{ animationDelay: '0.5s', animationFillMode: 'backwards' }}>
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-500 to-amber-600 flex items-center justify-center shadow-lg shadow-amber-500/20">
              <AlertCircle className="h-5 w-5 text-white" />
            </div>
            <div>
              <h3 className="font-semibold">Needs Attention</h3>
              <p className="text-xs text-muted-foreground">Videos requiring labels</p>
            </div>
          </div>
          
          {pendingCount > 0 ? (
            <>
              <p className="text-sm text-muted-foreground mb-3">
                <span className="font-semibold text-foreground">{pendingCount}</span> videos need labeling
              </p>
              <div className="space-y-2">
                {videos.filter(v => !v.has_label).slice(0, 3).map((video) => (
                  <Link
                    key={video.video_id}
                    to={`/video/${video.video_id}`}
                    className="flex items-center gap-3 p-2.5 rounded-xl bg-muted/50 hover:bg-muted transition-colors group"
                  >
                    <div className="w-8 h-8 rounded-lg bg-background flex items-center justify-center flex-shrink-0">
                      <PlayCircle className="h-4 w-4 text-muted-foreground" />
                    </div>
                    <span className="text-sm truncate flex-1">{video.original_filename || video.filename}</span>
                    <ChevronRight className="h-4 w-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
                  </Link>
                ))}
              </div>
              {pendingCount > 3 && (
                <p className="text-xs text-muted-foreground text-center mt-3">
                  +{pendingCount - 3} more videos
                </p>
              )}
            </>
          ) : (
            <div className="text-center py-6">
              <div className="w-12 h-12 rounded-full bg-emerald-500/15 mx-auto flex items-center justify-center mb-3">
                <CheckCircle2 className="h-6 w-6 text-emerald-500" />
              </div>
              <p className="text-sm text-emerald-500 font-medium">
                All videos are labeled!
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Recent Videos */}
      <div className="animate-slide-in-up" style={{ animationDelay: '0.6s', animationFillMode: 'backwards' }}>
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold">Recent Videos</h2>
          <span className="text-sm text-muted-foreground">
            Showing {Math.min(10, videos.length)} of {videos.length}
          </span>
        </div>
        
        {videos.length === 0 ? (
          <div className="premium-card text-center py-16">
            <div className="relative inline-flex mb-4">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center">
                <Video className="h-8 w-8 text-primary" />
              </div>
              <div className="absolute -inset-2 bg-primary/10 rounded-3xl blur-xl" />
            </div>
            <h3 className="text-lg font-semibold mb-2">No videos yet</h3>
            <p className="text-muted-foreground mb-6 max-w-sm mx-auto">
              Upload cow walking videos to start your lameness analysis journey
            </p>
            <Link to="/upload" className="btn-premium inline-flex items-center gap-2">
              <Upload className="h-4 w-4" />
              Upload Videos
            </Link>
          </div>
        ) : (
          <div className="premium-card overflow-hidden p-0">
            <div className="overflow-x-auto">
              <table className="premium-table">
                <thead>
                  <tr>
                    <th>Video</th>
                    <th>Status</th>
                    <th>Label</th>
                    <th>Size</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {videos.slice(0, 10).map((video, i) => (
                    <tr 
                      key={video.video_id}
                      className="animate-fade-in"
                      style={{ animationDelay: `${i * 0.03}s`, animationFillMode: 'backwards' }}
                    >
                      <td>
                        <div className="flex items-center gap-3">
                          <div className="w-10 h-10 rounded-lg bg-muted flex items-center justify-center flex-shrink-0">
                            <PlayCircle className="h-5 w-5 text-muted-foreground" />
                          </div>
                          <div>
                            <div className="font-medium truncate max-w-[200px]">{video.original_filename || video.filename}</div>
                            <div className="text-xs text-muted-foreground font-mono">
                              {video.video_id.slice(0, 8)}...
                            </div>
                          </div>
                        </div>
                      </td>
                      <td>
                        <div className="flex gap-1.5 flex-wrap">
                          {video.has_analysis && (
                            <span className="badge badge-primary">Analyzed</span>
                          )}
                          {video.has_annotated && (
                            <span className="badge bg-violet-500/15 text-violet-500">Annotated</span>
                          )}
                          {!video.has_analysis && !video.has_annotated && (
                            <span className="badge badge-muted">Pending</span>
                          )}
                        </div>
                      </td>
                      <td>
                        {video.has_label ? (
                          <span className={cn(
                            "badge",
                            video.label === 0 ? 'badge-success' : 'badge-destructive'
                          )}>
                            {video.label === 0 ? 'Sound' : 'Lame'}
                          </span>
                        ) : (
                          <span className="text-muted-foreground text-xs">Unlabeled</span>
                        )}
                      </td>
                      <td className="text-muted-foreground">
                        {(video.file_size / 1024 / 1024).toFixed(1)} MB
                      </td>
                      <td className="text-right">
                        <div className="flex gap-2 justify-end">
                          <Link
                            to={`/video/${video.video_id}`}
                            className="px-3 py-1.5 text-xs font-medium rounded-lg bg-muted hover:bg-muted/80 transition-colors"
                          >
                            View
                          </Link>
                          {video.has_analysis && (
                            <Link
                              to={`/results/${video.video_id}`}
                              className="px-3 py-1.5 text-xs font-medium rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
                            >
                              Results
                            </Link>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
