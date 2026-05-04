/**
 * Dashboard Page
 * Premium overview with modern metric cards and data visualization
 */
import { useEffect, useState, Fragment } from 'react'
import { Link } from 'react-router-dom'
import { cn } from '@/lib/utils'
import { videosApi, trainingApi } from '@/api/client'
import { useAuth } from '@/contexts/AuthContext'
import { getCowRankings, getConsensusData, type CowRanking } from '@/utils/pairwiseConsensus'
import { useLanguage } from '@/contexts/LanguageContext'
import {
  TrendingUp,
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
  Upload,
  PlayCircle,
  AlertCircle,
  ChevronRight,
  Trophy,
  ChevronDown,
  ChevronUp,
} from 'lucide-react'

export default function Dashboard() {
  const { user } = useAuth()
  const { t } = useLanguage()
  const isGuest = user?.id === 'guest'
  const useDemo = isGuest || user?.role === 'rater'

  const [videos, setVideos] = useState<any[]>([])
  const [stats, setStats] = useState<any>(null)
  const [trainingStatus, setTrainingStatus] = useState<any>(null)
  const [pairwiseStats, setPairwiseStats] = useState<any>(null)
  const [topLameCows, setTopLameCows] = useState<CowRanking[]>([])
  const [loading, setLoading] = useState(true)
  const [expandedCowId, setExpandedCowId] = useState<string | null>(null)

  useEffect(() => {
    if (useDemo) {
      loadDemoData()
    } else {
      loadData()
    }
  }, [useDemo])

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

  const loadDemoData = () => {
    // Derive everything from the real pairwise comparison CSV
    const rankings = getCowRankings()
    const consensus = getConsensusData()

    const totalJudgments = Array.from(consensus.values())
      .reduce((s, p) => s + p.count, 0)
    const pairsCompared = consensus.size
    // 30 cows in our dataset → 30*29/2 = 435 possible pairs
    const totalPossible = (rankings.length * (rankings.length - 1)) / 2

    // Build synthetic "video" rows so the table isn't empty
    const demoVideos = rankings.map((r) => ({
      video_id: r.cowId,
      filename: `cow_${r.cowId}.mp4`,
      file_size: 350 * 1024,
      has_analysis: true,
      has_annotated: true,
      has_label: true,
      label: r.severity === 'severe' || r.severity === 'moderate' ? 1 : 0,
      storage_backend: 'local',
      rank: r.rank,
      videoUrl: r.videoUrl || '',
      severity: r.severity,
      normalizedScore: r.normalizedScore,
      rawScore: r.rawScore,
    }))

    setVideos(demoVideos)
    setTopLameCows(rankings.slice(0, 5))
    setPairwiseStats({
      pairs_compared: pairsCompared,
      total_possible_pairs: totalPossible,
      completion_rate: pairsCompared / totalPossible,
      total_comparisons: totalJudgments,
    })
    setTrainingStatus({
      status: 'completed',
      last_trained: new Date().toISOString(),
      samples_used: totalJudgments,
    })
    setLoading(false)
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
  const pendingCount = videos.filter(v => !v.has_label).length

  const allMetrics = [
    {
      label: 'Total Videos',
      value: videos.length,
      icon: Video,
      trend: null,
      color: 'from-blue-500 to-blue-600',
      bgColor: 'bg-blue-500/10',
      textColor: 'text-blue-500',
      raterOnly: false,
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
      raterOnly: true,
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
      raterOnly: true,
    },
    {
      label: 'Pending',
      value: pendingCount,
      icon: Clock,
      trend: null,
      color: 'from-slate-500 to-slate-600',
      bgColor: 'bg-slate-500/10',
      textColor: 'text-slate-500',
      raterOnly: false,
    },
  ]
  // Raters see a simplified view without Analyzed / Labeled stats
  const metrics = useDemo ? allMetrics.filter(m => !m.raterOnly) : allMetrics

  return (
    <>
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start gap-4">
        <div className="animate-slide-in-up">
          <h1 className="text-3xl font-bold flex items-center gap-3">
            {t('nav.dashboard')}
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-lg bg-primary/10 text-primary text-sm font-medium">
              <Sparkles className="h-3.5 w-3.5" />
              {useDemo ? 'Demo' : 'Live'}
            </span>
          </h1>
          <p className="text-muted-foreground mt-1">
            {t('dashboard.overview')}
          </p>
        </div>
        {!useDemo && (
          <div className="flex gap-2">
            <Link
              to="/upload"
              className="btn-premium inline-flex items-center gap-2 animate-slide-in-up"
              style={{ animationDelay: '0.1s' }}
            >
              <Upload className="h-4 w-4" />
              Upload Videos
            </Link>
          </div>
        )}
      </div>

      {/* Metrics Grid */}
      <div className={cn("grid gap-4", useDemo ? "grid-cols-2 md:grid-cols-4" : "grid-cols-2 md:grid-cols-3 lg:grid-cols-6")}>
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
      <div className={cn("grid gap-6", useDemo ? "lg:grid-cols-2" : "lg:grid-cols-3")}>
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
              <Link
                to="/pairwise"
                className="flex items-center justify-between p-3 rounded-xl bg-muted/50 hover:bg-muted transition-colors group"
              >
                <span className="text-sm font-medium">{t('dashboard.continueComparing')}</span>
                <ArrowRight className="h-4 w-4 text-muted-foreground group-hover:translate-x-1 transition-transform" />
              </Link>
            </>
          ) : (
          <div className="text-center py-6">
            <div className="w-12 h-12 rounded-full bg-muted mx-auto flex items-center justify-center mb-3">
              <Target className="h-6 w-6 text-muted-foreground" />
            </div>
            <p className="text-sm text-muted-foreground">
              {t('dashboard.uploadToStart')}
            </p>
          </div>
          )}
        </div>

        {/* Training Status — hidden for rater/demo view */}
        {!useDemo && (
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
        )}

        {/* Top Lame Cows (demo) / Needs Attention (live) */}
        <div className="premium-card animate-slide-in-up" style={{ animationDelay: '0.5s', animationFillMode: 'backwards' }}>
          {useDemo && topLameCows.length > 0 ? (
            <>
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-rose-500 to-rose-600 flex items-center justify-center shadow-lg shadow-rose-500/20">
                  <Trophy className="h-5 w-5 text-white" />
                </div>
                <div>
                  <h3 className="font-semibold">{t('dashboard.mostLame')}</h3>
                  <p className="text-xs text-muted-foreground">{t('dashboard.top5')}</p>
                </div>
              </div>
              <div className="space-y-2">
                {topLameCows.map((cow, i) => (
                  <div key={cow.cowId} className="flex items-center gap-3 p-2.5 rounded-xl bg-muted/50">
                    <span className="text-sm font-bold w-6 text-center text-muted-foreground">
                      {i === 0 ? '🥇' : i === 1 ? '🥈' : i === 2 ? '🥉' : `#${i + 1}`}
                    </span>
                    <span className="font-mono text-sm font-medium flex-1">{cow.cowId}</span>
                    <span className={cn(
                      'px-2 py-0.5 rounded-md text-xs font-medium capitalize',
                      cow.severity === 'severe'   ? 'bg-red-500/15 text-red-500' :
                      cow.severity === 'moderate' ? 'bg-orange-500/15 text-orange-500' :
                      cow.severity === 'mild'     ? 'bg-amber-500/15 text-amber-500' :
                                                    'bg-emerald-500/15 text-emerald-500'
                    )}>
                      {(cow.rawScore || 0).toFixed(1)}
                    </span>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <>
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
                        <span className="text-sm truncate flex-1">{video.filename}</span>
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
            </>
          )}
        </div>
      </div>

      {/* Recent Videos */}
      <div className="animate-slide-in-up" style={{ animationDelay: '0.6s', animationFillMode: 'backwards' }}>
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold">{useDemo ? t('dashboard.cowRanking') : t('dashboard.recentVideos')}</h2>
          <span className="text-sm text-muted-foreground">
            {t('dashboard.showing')} {Math.min(10, videos.length)} {t('dashboard.of')} {videos.length} {useDemo ? t('dashboard.cows') : t('dashboard.videos')}
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
            <h3 className="text-lg font-semibold mb-2">{t('dashboard.noVideos')}</h3>
            <p className="text-muted-foreground mb-6 max-w-sm mx-auto">
              {t('dashboard.uploadToStartJourney')}
            </p>
            <Link to="/upload" className="btn-premium inline-flex items-center gap-2">
              <Upload className="h-4 w-4" />
              {t('dashboard.uploadVideos')}
            </Link>
          </div>
        ) : (
          <div className="premium-card overflow-hidden p-0">
            <div className="overflow-x-auto">
              <table className="premium-table">
                <thead>
                  <tr>
                    {useDemo && <th className="w-12 text-center">{t('dashboard.rank')}</th>}
                    <th>{t('dashboard.cow')}</th>
                    <th>{t('dashboard.status')}</th>
                    <th>{useDemo ? t('dashboard.eloScore') : t('dashboard.label')}</th>
                    {!useDemo && <th>{t('dashboard.size')}</th>}
                    {!useDemo && <th></th>}
                  </tr>
                </thead>
                <tbody>
                  {videos.slice(0, 10).map((video, i) => {
                    const isExpanded = expandedCowId === video.video_id
                    return (
                      <Fragment key={video.video_id}>
                        <tr
                          className="animate-fade-in"
                          style={{ animationDelay: `${i * 0.03}s`, animationFillMode: 'backwards' }}
                        >
                          {useDemo && (
                            <td className="text-center">
                              <span className="text-sm font-bold text-muted-foreground">
                                {video.rank === 1 ? '🥇' : video.rank === 2 ? '🥈' : video.rank === 3 ? '🥉' : `#${video.rank}`}
                              </span>
                            </td>
                          )}
                          <td>
                            <div className="flex items-center gap-3">
                              <button
                                onClick={() => setExpandedCowId(isExpanded ? null : video.video_id)}
                                className="relative w-12 h-9 rounded-lg overflow-hidden flex-shrink-0 border border-border/50 hover:border-primary/50 transition-colors group"
                                title={isExpanded ? 'Collapse' : 'Preview video'}
                              >
                                {video.videoUrl ? (
                                  <video
                                    src={`${video.videoUrl}#t=4`}
                                    muted
                                    playsInline
                                    preload="metadata"
                                    className="w-full h-full object-cover"
                                  />
                                ) : (
                                  <div className="w-full h-full bg-muted flex items-center justify-center">
                                    <PlayCircle className="h-4 w-4 text-muted-foreground" />
                                  </div>
                                )}
                                <div className="absolute inset-0 bg-black/30 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                                  {isExpanded
                                    ? <ChevronUp className="h-3 w-3 text-white" />
                                    : <PlayCircle className="h-3 w-3 text-white" />
                                  }
                                </div>
                              </button>
                              <div>
                                <div className="font-medium font-mono">{video.video_id}</div>
                                <div className="text-xs text-muted-foreground">
                                  {!useDemo && video.filename}
                                </div>
                              </div>
                            </div>
                          </td>
                          <td>
                            <div className="flex gap-1.5 flex-wrap">
                              {useDemo ? (
                                video.has_annotated && (
                                  <span className="badge bg-violet-500/15 text-violet-500">{t('dashboard.annotated')}</span>
                                )
                              ) : (
                                <>
                                  {video.has_analysis && (
                                    <span className="badge badge-primary">{t('dashboard.analyzed')}</span>
                                  )}
                                  {video.has_annotated && (
                                    <span className="badge bg-violet-500/15 text-violet-500">{t('dashboard.annotated')}</span>
                                  )}
                                  {!video.has_analysis && !video.has_annotated && (
                                    <span className="badge badge-muted">{t('dashboard.pending')}</span>
                                  )}
                                </>
                              )}
                            </div>
                          </td>
                          <td>
                            {useDemo ? (
                              <div className="flex items-center gap-2 min-w-[80px]">
                                <span className={cn(
                                  "text-sm font-mono font-medium tabular-nums",
                                  (video.normalizedScore ?? 0) >= 0.75 ? 'text-red-500' :
                                  (video.normalizedScore ?? 0) >= 0.50 ? 'text-orange-500' :
                                  (video.normalizedScore ?? 0) >= 0.25 ? 'text-amber-500' : 'text-emerald-500'
                                )}>
                                  {video.rawScore?.toFixed(1) || 0}
                                </span>
                              </div>
                            ) : (
                              video.has_label ? (
                                <span className={cn(
                                  "badge",
                                  video.label === 0 ? 'badge-success' : 'badge-destructive'
                                )}>
                                  {video.label === 0 ? t('dashboard.sound') : t('dashboard.lame')}
                                </span>
                              ) : (
                                <span className="text-muted-foreground text-xs">{t('dashboard.unlabeled')}</span>
                              )
                            )}
                          </td>
                          {!useDemo && (
                            <td className="text-muted-foreground">
                              {(video.file_size / 1024 / 1024).toFixed(1)} MB
                            </td>
                          )}
                          {!useDemo && (
                            <td className="text-right">
                              <div className="flex gap-2 justify-end">
                                <Link
                                  to={`/video/${video.video_id}`}
                                  className="px-3 py-1.5 text-xs font-medium rounded-lg bg-muted hover:bg-muted/80 transition-colors"
                                >
                                  {t('dashboard.view')}
                                </Link>
                                {video.has_analysis && (
                                  <Link
                                    to={`/results/${video.video_id}`}
                                    className="px-3 py-1.5 text-xs font-medium rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
                                  >
                                    {t('dashboard.results')}
                                  </Link>
                                )}
                              </div>
                            </td>
                          )}
                        </tr>
                        {isExpanded && (
                          <tr className="bg-muted/30">
                            <td colSpan={useDemo ? 5 : 5} className="p-4">
                              <div className="flex items-start gap-4">
                                {video.videoUrl ? (
                                  <video
                                    src={video.videoUrl}
                                    controls
                                    autoPlay
                                    muted
                                    playsInline
                                    className="rounded-xl border border-border/50 max-h-52 max-w-xs object-contain bg-black"
                                  />
                                ) : (
                                  <div className="w-48 h-32 rounded-xl bg-muted flex items-center justify-center">
                                    <span className="text-xs text-muted-foreground">{t('dashboard.noVideoAvailable')}</span>
                                  </div>
                                )}
                                <div className="flex-1 min-w-0">
                                  <div className="font-semibold font-mono text-base mb-1">Cow {video.video_id}</div>
                                  <div className="flex gap-2 flex-wrap mb-3">
                                    {!useDemo && (
                                      <span className={cn("badge", video.label === 0 ? 'badge-success' : 'badge-destructive')}>
                                        {video.label === 0 ? t('dashboard.sound') : t('dashboard.lame')}
                                      </span>
                                    )}
                                    {useDemo && (
                                      <span className="badge bg-muted text-muted-foreground">{t('dashboard.rank')} #{video.rank}</span>
                                    )}
                                    {useDemo && (
                                      <span className={cn(
                                        "badge",
                                        (video.normalizedScore ?? 0) >= 0.75 ? 'bg-red-500/15 text-red-500' :
                                        (video.normalizedScore ?? 0) >= 0.50 ? 'bg-orange-500/15 text-orange-500' :
                                        (video.normalizedScore ?? 0) >= 0.25 ? 'bg-amber-500/15 text-amber-500' : 'bg-emerald-500/15 text-emerald-500'
                                      )}>
                                        {video.rawScore?.toFixed(1) || 0} {t('dashboard.eloScore')}
                                      </span>
                                    )}
                                  </div>
                                  {!useDemo && (
                                    <p className="text-xs text-muted-foreground">
                                      {t('dashboard.clickResults')}
                                    </p>
                                  )}
                                </div>
                              </div>
                            </td>
                          </tr>
                        )}
                      </Fragment>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>

    </>
  )
}
