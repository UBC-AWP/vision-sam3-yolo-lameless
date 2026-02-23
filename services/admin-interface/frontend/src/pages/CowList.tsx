import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { cowsApi, CowIdentity } from '@/api/client'
import { cn } from '@/lib/utils'
import { Beef, Search, RefreshCw, Loader2, ChevronLeft, ChevronRight, Activity } from 'lucide-react'

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
  const [cows, setCows] = useState<CowIdentity[]>([])
  const [stats, setStats] = useState<CowStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  
  // Filters
  const [severityFilter, setSeverityFilter] = useState<string>('')
  const [activeFilter, setActiveFilter] = useState<boolean | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  
  // Pagination
  const [skip, setSkip] = useState(0)
  const [total, setTotal] = useState(0)
  const limit = 20

  useEffect(() => {
    loadData()
  }, [severityFilter, activeFilter, skip])

  const loadData = async () => {
    try {
      setLoading(true)
      
      const [cowsData, statsData] = await Promise.all([
        cowsApi.list({
          skip,
          limit,
          is_active: activeFilter ?? undefined,
          severity_filter: severityFilter || undefined
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

  const getSeverityColor = (severity: string | null | undefined): string => {
    switch (severity) {
      case 'healthy': return 'bg-emerald-500/15 text-emerald-500 border-emerald-500/30'
      case 'mild': return 'bg-amber-500/15 text-amber-500 border-amber-500/30'
      case 'moderate': return 'bg-orange-500/15 text-orange-500 border-orange-500/30'
      case 'severe': return 'bg-red-500/15 text-red-500 border-red-500/30'
      default: return 'bg-muted text-muted-foreground border-border'
    }
  }

  const getSeverityIcon = (severity: string | null | undefined): string => {
    switch (severity) {
      case 'healthy': return '🐄'
      case 'mild': return '🟡'
      case 'moderate': return '🟠'
      case 'severe': return '🔴'
      default: return '❓'
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

  const filteredCows = cows.filter(cow => {
    if (!searchQuery) return true
    const query = searchQuery.toLowerCase()
    return cow.cow_id.toLowerCase().includes(query) ||
           (cow.tag_number?.toLowerCase().includes(query) ?? false)
  })

  if (loading && cows.length === 0) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="text-center animate-fade-in">
          <div className="relative inline-flex">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary to-primary/60 flex items-center justify-center animate-pulse-soft">
              <Beef className="h-8 w-8 text-primary-foreground" />
            </div>
            <div className="absolute -inset-2 bg-primary/20 rounded-3xl blur-xl animate-pulse-soft" />
          </div>
          <p className="mt-4 text-muted-foreground">Loading cow registry...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4 animate-slide-in-up">
        <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-primary to-primary/60 flex items-center justify-center shadow-lg shadow-primary/20">
          <Beef className="h-6 w-6 text-primary-foreground" />
        </div>
        <div>
          <h1 className="text-2xl font-bold">Cow Registry</h1>
          <p className="text-muted-foreground">Track and monitor individual cows across video analyses</p>
        </div>
      </div>

      {/* Stats Cards */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
          {[
            { label: 'Total Cows', value: stats.total_cows, icon: '🐮', color: 'text-foreground' },
            { label: 'Active', value: stats.active_cows, icon: '✅', color: 'text-emerald-500' },
            { label: 'Videos Tracked', value: stats.total_videos_tracked, icon: '📹', color: 'text-blue-500' },
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
              placeholder="Search by cow ID or tag..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="input-premium pl-10 w-full"
            />
          </div>
          
          <select
            value={severityFilter}
            onChange={(e) => setSeverityFilter(e.target.value)}
            className="input-premium w-auto"
          >
            <option value="">All Severities</option>
            <option value="healthy">Healthy</option>
            <option value="mild">Mild</option>
            <option value="moderate">Moderate</option>
            <option value="severe">Severe</option>
          </select>
          
          <select
            value={activeFilter === null ? '' : activeFilter ? 'active' : 'inactive'}
            onChange={(e) => {
              if (e.target.value === '') setActiveFilter(null)
              else setActiveFilter(e.target.value === 'active')
            }}
            className="input-premium w-auto"
          >
            <option value="">All Status</option>
            <option value="active">Active Only</option>
            <option value="inactive">Inactive Only</option>
          </select>
          
          <button
            onClick={loadData}
            className="p-2.5 rounded-xl hover:bg-accent/50 text-muted-foreground hover:text-foreground transition-colors"
          >
            <RefreshCw className={cn("h-5 w-5", loading && 'animate-spin')} />
          </button>
        </div>
      </div>

      {/* Error State */}
      {error && (
        <div className="p-4 rounded-xl bg-destructive/10 border border-destructive/30 text-destructive animate-scale-in">
          {error}
        </div>
      )}

      {/* Cow Table */}
      {filteredCows.length === 0 ? (
        <div className="premium-card text-center py-16 animate-fade-in">
          <div className="w-16 h-16 rounded-2xl bg-muted flex items-center justify-center mx-auto mb-4">
            <Beef className="h-8 w-8 text-muted-foreground" />
          </div>
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
                  <th>Cow ID</th>
                  <th>Tag</th>
                  <th>Severity</th>
                  <th>Score</th>
                  <th>Videos</th>
                  <th>Last Seen</th>
                  <th>Status</th>
                  <th className="text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredCows.map((cow, i) => (
                  <tr
                    key={cow.id}
                    className="animate-fade-in"
                    style={{ animationDelay: `${i * 0.03}s`, animationFillMode: 'backwards' }}
                  >
                    <td>
                      <Link 
                        to={`/cows/${cow.cow_id}`}
                        className="font-medium text-primary hover:underline"
                      >
                        {cow.cow_id.slice(0, 8)}...
                      </Link>
                    </td>
                    <td>
                      {cow.tag_number ? (
                        <span className="px-2 py-1 bg-muted rounded-lg text-sm font-mono">
                          {cow.tag_number}
                        </span>
                      ) : (
                        <span className="text-muted-foreground text-sm">—</span>
                      )}
                    </td>
                    <td>
                      <span className={cn(
                        "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium border",
                        getSeverityColor(cow.severity_level)
                      )}>
                        {getSeverityIcon(cow.severity_level)}
                        <span className="capitalize">{cow.severity_level || 'Unknown'}</span>
                      </span>
                    </td>
                    <td>
                      {cow.current_score !== null && cow.current_score !== undefined ? (
                        <div className="flex items-center gap-2">
                          <div className="w-16 bg-muted rounded-full h-2 overflow-hidden">
                            <div
                              className={cn(
                                "h-full rounded-full transition-all",
                                cow.current_score < 0.3 ? 'bg-emerald-500' :
                                cow.current_score < 0.5 ? 'bg-amber-500' :
                                cow.current_score < 0.7 ? 'bg-orange-500' : 'bg-red-500'
                              )}
                              style={{ width: `${cow.current_score * 100}%` }}
                            />
                          </div>
                          <span className="text-sm font-mono">
                            {(cow.current_score * 100).toFixed(0)}%
                          </span>
                        </div>
                      ) : (
                        <span className="text-muted-foreground text-sm">—</span>
                      )}
                    </td>
                    <td>
                      <span className="text-sm">
                        {cow.num_videos ?? cow.total_sightings ?? 0}
                      </span>
                    </td>
                    <td className="text-muted-foreground">
                      {formatDate(cow.last_seen)}
                    </td>
                    <td>
                      {cow.is_active ? (
                        <span className="badge badge-success">Active</span>
                      ) : (
                        <span className="badge badge-muted">Inactive</span>
                      )}
                    </td>
                    <td className="text-right">
                      <Link
                        to={`/cows/${cow.cow_id}`}
                        className="px-3 py-1.5 text-xs font-medium rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
                      >
                        View Details →
                      </Link>
                    </td>
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
