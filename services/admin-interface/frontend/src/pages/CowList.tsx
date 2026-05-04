import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { cowsApi, CowIdentity } from '@/api/client'
import { cn } from '@/lib/utils'
import { Search, Loader2, ChevronLeft, ChevronRight, Activity, Trophy } from 'lucide-react'
import { useAuth } from '@/contexts/AuthContext'
import { useLanguage } from '@/contexts/LanguageContext'
import { getCowRankings } from '@/utils/pairwiseConsensus'

interface RankedCow extends CowIdentity {
  rank?: number
  wins?: number
  losses?: number
  comparisons?: number
  rawScore?: number
}

interface SeverityStats {
  healthy: number
  mild: number
  moderate: number
  severe: number
  unknown: number
}

interface CowStats {
  total_cows: number
  active_cows: number
  total_videos_tracked: number
  total_lameness_records: number
  severity_distribution: SeverityStats
}

export default function CowList() {
  const { user } = useAuth()
  const { t } = useLanguage()
  const isGuest = user?.id === 'guest'
  const useDemo = isGuest || user?.role === 'rater'

  const [cows, setCows] = useState<RankedCow[]>([])
  const [stats, setStats] = useState<CowStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  
  // Search
  const [searchQuery, setSearchQuery] = useState('')
  
  // Sorting
  const [sortBy, setSortBy] = useState<'severity' | 'score' | 'videos' | 'lastSeen' | null>(null)
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc')
  
  // Pagination
  const [skip, setSkip] = useState(0)
  const [total, setTotal] = useState(0)
  const limit = 20

  useEffect(() => {
    if (useDemo) {
      loadDemoData()
    } else {
      loadData()
    }
  }, [skip, useDemo])

  const loadDemoData = () => {
    setError(null)

    // Use real lameness rankings derived from pairwise comparison CSV
    const rankings = getCowRankings()

    const rankedCows: RankedCow[] = rankings.map((r) => ({
      id: r.cowId,
      cow_id: r.cowId,
      tag_number: `#${r.cowId}`,
      total_sightings: r.comparisons,
      first_seen: null,
      last_seen: null,
      is_active: true,
      notes: `Rank #${r.rank} — ${r.wins}W / ${r.losses}L / ${r.ties}T from ${r.comparisons} judgments`,
      current_score: parseFloat(r.normalizedScore.toFixed(3)),
      severity_level: r.severity,
      num_videos: r.comparisons,
      rank: r.rank,
      wins: r.wins,
      losses: r.losses,
      comparisons: r.comparisons,
    }))

    const distribution = {
      healthy:  rankedCows.filter(c => c.severity_level === 'healthy').length,
      mild:     rankedCows.filter(c => c.severity_level === 'mild').length,
      moderate: rankedCows.filter(c => c.severity_level === 'moderate').length,
      severe:   rankedCows.filter(c => c.severity_level === 'severe').length,
      unknown:  0,
    }

    setCows(rankedCows)
    setTotal(rankedCows.length)
    setStats({
      total_cows: rankedCows.length,
      active_cows: rankedCows.length,
      total_videos_tracked: rankedCows.reduce((s, c) => s + (c.num_videos ?? 0), 0),
      total_lameness_records: rankedCows.length,
      severity_distribution: distribution,
    })
    setLoading(false)
  }
  
  const loadData = async () => {
    
    try {
      setLoading(true)
      
      const [cowsData, statsData] = await Promise.all([
        cowsApi.list({
          skip,
          limit
        }),
        cowsApi.getStats()
      ])
      
      setCows(cowsData.cows)
      setTotal(cowsData.total)
      setStats(statsData)
      setError(null)
    } catch (err: any) {
      console.error('Failed to load cows:', err)
      setError(err.response?.data?.detail || 'Failed to load cow data')
    } finally {
      setLoading(false)
    }
  }



  const formatDate = (dateStr: string | null | undefined): string => {
    if (!dateStr) return 'Never'
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    })
  }

  const handleSort = (column: 'severity' | 'score' | 'videos' | 'lastSeen') => {
    if (sortBy === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')
    } else {
      setSortBy(column)
      setSortDirection('desc')
    }
  }

  const filteredCows = cows.filter(cow => {
    if (!searchQuery) return true
    const query = searchQuery.toLowerCase()
    return cow.cow_id.toLowerCase().includes(query)
  })

  const sortedCows = [...filteredCows].sort((a, b) => {
    if (!sortBy) return 0
    
    let compareResult = 0
    
    if (sortBy === 'severity') {
      const severityOrder = { 'severe': 4, 'moderate': 3, 'mild': 2, 'healthy': 1, 'unknown': 0 }
      const aValue = severityOrder[a.severity_level as keyof typeof severityOrder] || 0
      const bValue = severityOrder[b.severity_level as keyof typeof severityOrder] || 0
      compareResult = aValue - bValue
    } else if (sortBy === 'score') {
      compareResult = (a.current_score || 0) - (b.current_score || 0)
    } else if (sortBy === 'videos') {
      compareResult = (a.num_videos || a.total_sightings || 0) - (b.num_videos || b.total_sightings || 0)
    } else if (sortBy === 'lastSeen') {
      const aDate = a.last_seen ? new Date(a.last_seen).getTime() : 0
      const bDate = b.last_seen ? new Date(b.last_seen).getTime() : 0
      compareResult = aDate - bDate
    }
    
    return sortDirection === 'asc' ? compareResult : -compareResult
  })

  if (loading && cows.length === 0) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="text-center animate-fade-in">
          <p className="text-muted-foreground">Loading cow registry...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between animate-slide-in-up">
        <div>
          <h1 className="text-2xl font-bold">{t('nav.cowRegistry')}</h1>
          <p className="text-muted-foreground">Track and monitor individual cows across video analyses</p>
          {useDemo && (
            <div className="mt-1">
              <span className="px-2 py-0.5 bg-warning/20 text-warning rounded-full text-xs font-medium">
                🎯 Demo Mode — Rankings derived from pairwise comparison CSV
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Stats Cards - Removed Active and Status blocks */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
          {[
            { label: 'Total Cows', value: stats.total_cows, icon: '🐮', color: 'text-foreground' },
            { label: 'Healthy', value: stats.severity_distribution.healthy, color: 'text-emerald-500' },
            { label: 'Moderate', value: stats.severity_distribution.moderate + stats.severity_distribution.mild, color: 'text-amber-500' },
            { label: 'Severe', value: stats.severity_distribution.severe, color: 'text-red-500' },
          ].map((stat, i) => (
            <div
              key={stat.label}
              className="premium-card animate-slide-in-up"
              style={{ animationDelay: `${i * 0.05}s`, animationFillMode: 'backwards' }}
            >
              <div className="flex items-center gap-2 mb-1">
                {stat.icon && <span>{stat.icon}</span>}
                <p className="text-sm text-muted-foreground">{stat.label}</p>
              </div>
              <p className={cn("text-2xl font-bold", stat.color)}>{stat.value}</p>
            </div>
          ))}
        </div>
      )}

      {/* Severity Distribution Chart */}
      {stats && (
        <div className="premium-card animate-slide-in-up" style={{ animationDelay: '0.3s', animationFillMode: 'backwards' }}>
          <h3 className="text-lg font-semibold mb-4">Severity Distribution</h3>
          <div className="flex items-center gap-1 h-3 rounded-full overflow-hidden bg-muted">
            {Object.entries(stats.severity_distribution).map(([severity, count]) => {
              const total = Object.values(stats.severity_distribution).reduce((a, b) => a + b, 0)
              const percentage = total > 0 ? (count / total) * 100 : 0
              if (percentage === 0) return null
              return (
                <div
                  key={severity}
                  className={cn(
                    "h-full transition-all",
                    severity === 'healthy' ? 'bg-emerald-500' :
                    severity === 'mild' ? 'bg-amber-400' :
                    severity === 'moderate' ? 'bg-orange-500' :
                    severity === 'severe' ? 'bg-red-500' : 'bg-muted-foreground'
                  )}
                  style={{ width: `${percentage}%` }}
                  title={`${severity}: ${count} (${percentage.toFixed(1)}%)`}
                />
              )
            })}
          </div>
          <div className="flex gap-4 mt-3 text-sm flex-wrap">
            {Object.entries(stats.severity_distribution).map(([severity, count]) => (
              <div key={severity} className="flex items-center gap-2">
                <span className={cn(
                  "w-3 h-3 rounded-full",
                  severity === 'healthy' ? 'bg-emerald-500' :
                  severity === 'mild' ? 'bg-amber-400' :
                  severity === 'moderate' ? 'bg-orange-500' :
                  severity === 'severe' ? 'bg-red-500' : 'bg-muted-foreground'
                )} />
                <span className="capitalize text-muted-foreground">{severity}: <span className="text-foreground font-medium">{count}</span></span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="premium-card animate-slide-in-up" style={{ animationDelay: '0.4s', animationFillMode: 'backwards' }}>
        <div className="flex flex-wrap gap-4 items-center">
          <div className="relative flex-1 min-w-[200px] max-w-md">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search by cow ID..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="input-premium pl-10 w-full"
            />
          </div>
        </div>
      </div>

      {/* Error State */}
      {error && (
        <div className="p-4 rounded-xl bg-destructive/10 border border-destructive/30 text-destructive animate-scale-in">
          {error}
        </div>
      )}

      {/* Cow Table */}
      {sortedCows.length === 0 ? (
        <div className="premium-card text-center py-16 animate-fade-in">
          <h4 className="text-lg font-semibold mb-2">No cows found</h4>
          <p className="text-muted-foreground">
            {cows.length === 0 
              ? 'Process videos with tracking enabled to identify cows'
              : 'No cows match your current filters'}
          </p>
        </div>
      ) : (
        <div className="premium-card p-0 overflow-hidden animate-slide-in-up" style={{ animationDelay: '0.5s', animationFillMode: 'backwards' }}>
          <div className="overflow-x-auto">
              <table className="premium-table">
              <thead>
                <tr>
                  {useDemo && <th className="w-12 text-center">Rank</th>}
                  <th>Cow ID</th>
                  <th 
                    className="cursor-pointer hover:bg-accent/50 transition-colors"
                    onClick={() => handleSort('score')}
                  >
                    <div className="flex items-center gap-1">
                      Elo Score
                      {sortBy === 'score' && (
                        <span className="text-xs">{sortDirection === 'asc' ? '↑' : '↓'}</span>
                      )}
                    </div>
                  </th>
                  {!useDemo && (
                    <>
                      <th 
                        className="cursor-pointer hover:bg-accent/50 transition-colors"
                        onClick={() => handleSort('videos')}
                      >
                        <div className="flex items-center gap-1">
                          Videos
                          {sortBy === 'videos' && (
                            <span className="text-xs">{sortDirection === 'asc' ? '↑' : '↓'}</span>
                          )}
                        </div>
                      </th>
                      <th 
                        className="cursor-pointer hover:bg-accent/50 transition-colors"
                        onClick={() => handleSort('lastSeen')}
                      >
                        <div className="flex items-center gap-1">
                          Last Seen
                          {sortBy === 'lastSeen' && (
                            <span className="text-xs">{sortDirection === 'asc' ? '↑' : '↓'}</span>
                          )}
                        </div>
                      </th>
                    </>
                  )}
                  {!useDemo && <th className="text-right">Actions</th>}
                </tr>
              </thead>
              <tbody>
                {sortedCows.map((cow, i) => (
                  <tr
                    key={cow.id}
                    className="animate-fade-in"
                    style={{ animationDelay: `${i * 0.03}s`, animationFillMode: 'backwards' }}
                  >
                    {useDemo && (
                      <td className="text-center">
                        <span className={cn(
                          'inline-flex items-center justify-center w-8 h-8 rounded-full text-xs font-bold',
                          (cow as RankedCow).rank === 1 ? 'bg-yellow-500/20 text-yellow-500' :
                          (cow as RankedCow).rank === 2 ? 'bg-gray-400/20 text-gray-400' :
                          (cow as RankedCow).rank === 3 ? 'bg-orange-500/20 text-orange-500' :
                          'bg-muted text-muted-foreground'
                        )}>
                          {(cow as RankedCow).rank === 1 ? '🥇' :
                           (cow as RankedCow).rank === 2 ? '🥈' :
                           (cow as RankedCow).rank === 3 ? '🥉' :
                           `#${(cow as RankedCow).rank}`}
                        </span>
                      </td>
                    )}
                    <td>
                      {useDemo ? (
                        <span className="font-medium font-mono">{cow.cow_id}</span>
                      ) : (
                        <Link
                          to={`/cows/${cow.cow_id}`}
                          className="font-medium text-primary hover:underline font-mono"
                        >
                          {cow.cow_id}
                        </Link>
                      )}
                    </td>
                    <td>
                      {cow.current_score !== null && cow.current_score !== undefined ? (
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-mono font-medium">
                            {useDemo ? ((cow as RankedCow).rawScore || 0).toFixed(1) : cow.current_score.toFixed(1)}
                          </span>
                        </div>
                      ) : (
                        <span className="text-muted-foreground text-sm">—</span>
                      )}
                    </td>
                    {!useDemo && (
                      <>
                        <td>
                          <span className="text-sm">{cow.num_videos ?? cow.total_sightings ?? 0}</span>
                        </td>
                        <td className="text-muted-foreground">
                          {formatDate(cow.last_seen)}
                        </td>
                      </>
                    )}
                    {!useDemo && (
                      <td className="text-right">
                        <Link
                          to={`/cows/${cow.cow_id}`}
                          className="px-3 py-1.5 text-xs font-medium rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
                        >
                          View Details →
                        </Link>
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Pagination */}
      {total > limit && (
        <div className="flex justify-between items-center animate-fade-in">
          <span className="text-sm text-muted-foreground">
            Showing {skip + 1} - {Math.min(skip + limit, total)} of {total} cows
          </span>
          <div className="flex gap-2">
            <button
              onClick={() => setSkip(Math.max(0, skip - limit))}
              disabled={skip === 0}
              className="px-4 py-2 rounded-xl border border-border hover:bg-accent/50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
            >
              <ChevronLeft className="h-4 w-4" />
              Previous
            </button>
            <button
              onClick={() => setSkip(skip + limit)}
              disabled={skip + limit >= total}
              className="px-4 py-2 rounded-xl border border-border hover:bg-accent/50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
            >
              Next
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
