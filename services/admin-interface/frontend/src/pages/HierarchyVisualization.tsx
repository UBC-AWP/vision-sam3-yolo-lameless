import { useEffect, useState, useRef, useCallback } from 'react'
import { eloRankingApi, videosApi } from '@/api/client'

interface RankingItem {
  video_id: string
  rank: number
  elo_rating: number
  elo_uncertainty: number
  davids_score: number
  wins: number
  losses: number
  ties: number
  total_comparisons: number
  win_rate: number
  category: 'lame' | 'intermediate' | 'healthy'
  confidence: number
}

interface HierarchyMetrics {
  steepness: number
  steepness_se: number
  inter_rater_agreement: number
  hierarchy_linearity: string
}

interface VideoPreview {
  video_id: string
  x: number
  y: number
}

interface Snapshot {
  id: string
  name: string
  description?: string
  total_videos: number
  total_comparisons: number
  steepness: number
  inter_rater_reliability: number
  created_at: string
}

export default function HierarchyVisualization() {
  const [ranking, setRanking] = useState<RankingItem[]>([])
  const [metrics, setMetrics] = useState<HierarchyMetrics | null>(null)
  const [stats, setStats] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [selectedVideo, setSelectedVideo] = useState<string | null>(null)
  const [hoveredVideo, setHoveredVideo] = useState<VideoPreview | null>(null)
  const [viewMode, setViewMode] = useState<'list' | 'bar' | 'distribution' | 'davids'>('bar')
  const [categoryFilter, setCategoryFilter] = useState<string>('all')
  const [snapshots, setSnapshots] = useState<Snapshot[]>([])
  const [showSnapshotModal, setShowSnapshotModal] = useState(false)
  const [snapshotName, setSnapshotName] = useState('')
  const [snapshotDescription, setSnapshotDescription] = useState('')
  const [videoHistory, setVideoHistory] = useState<any>(null)
  const chartRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    setLoading(true)
    try {
      const [hierarchyData, statsData, snapshotsData] = await Promise.all([
        eloRankingApi.getHierarchy(),
        eloRankingApi.getStats(),
        eloRankingApi.getSnapshots()
      ])

      // Categorize based on Elo rating
      const categorizedRanking = hierarchyData.ranking.map((item: any) => ({
        ...item,
        category: item.elo_rating > 1550 ? 'lame' :
                  item.elo_rating < 1450 ? 'healthy' : 'intermediate',
        confidence: calculateConfidence(item.total_comparisons, item.elo_uncertainty)
      }))

      setRanking(categorizedRanking)
      setMetrics(hierarchyData.metrics)
      setStats(statsData)
      setSnapshots(snapshotsData.snapshots || [])
    } catch (error) {
      console.error('Failed to load hierarchy:', error)
    } finally {
      setLoading(false)
    }
  }

  const calculateConfidence = (comparisons: number, uncertainty: number): number => {
    // Combine comparison count and Elo uncertainty for confidence
    const compConfidence = Math.min(1, comparisons / 20)
    const uncertaintyConfidence = 1 - (uncertainty / 350) // 350 is initial uncertainty
    return (compConfidence + uncertaintyConfidence) / 2
  }

  const handleCreateSnapshot = async () => {
    if (!snapshotName.trim()) return

    try {
      await eloRankingApi.createSnapshot(snapshotName, snapshotDescription)
      setShowSnapshotModal(false)
      setSnapshotName('')
      setSnapshotDescription('')
      loadData() // Refresh snapshots
    } catch (error) {
      console.error('Failed to create snapshot:', error)
      alert('Failed to create snapshot')
    }
  }

  const handleRecalculate = async () => {
    if (!confirm('This will recalculate all Elo ratings from scratch. Continue?')) return

    try {
      setLoading(true)
      await eloRankingApi.recalculateRatings()
      await loadData()
    } catch (error) {
      console.error('Failed to recalculate:', error)
      alert('Failed to recalculate ratings')
    }
  }

  const loadVideoHistory = async (videoId: string) => {
    try {
      const history = await eloRankingApi.getVideoHistory(videoId)
      setVideoHistory(history)
    } catch (error) {
      console.error('Failed to load video history:', error)
    }
  }

  const filteredRanking = categoryFilter === 'all'
    ? ranking
    : ranking.filter(item => item.category === categoryFilter)

  const handleVideoHover = useCallback((e: React.MouseEvent, videoId: string) => {
    const rect = (e.target as HTMLElement).getBoundingClientRect()
    setHoveredVideo({
      video_id: videoId,
      x: rect.left + rect.width / 2,
      y: rect.top
    })
  }, [])

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'lame': return 'text-red-600'
      case 'intermediate': return 'text-yellow-600'
      case 'healthy': return 'text-green-600'
      default: return 'text-gray-600'
    }
  }

  const getCategoryBg = (category: string) => {
    switch (category) {
      case 'lame': return 'bg-red-100'
      case 'intermediate': return 'bg-yellow-100'
      case 'healthy': return 'bg-green-100'
      default: return 'bg-gray-100'
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <div className="text-muted-foreground">Loading hierarchy...</div>
        </div>
      </div>
    )
  }

  const minElo = Math.min(...ranking.map(r => r.elo_rating), 1400)
  const maxElo = Math.max(...ranking.map(r => r.elo_rating), 1600)
  const eloRange = maxElo - minElo || 1

  // Calculate distribution data
  const distributionBins = [
    { label: 'Healthy (< 1450)', count: ranking.filter(r => r.elo_rating < 1450).length, color: 'bg-green-500' },
    { label: 'Intermediate (1450-1550)', count: ranking.filter(r => r.elo_rating >= 1450 && r.elo_rating <= 1550).length, color: 'bg-yellow-500' },
    { label: 'Lame (> 1550)', count: ranking.filter(r => r.elo_rating > 1550).length, color: 'bg-red-500' },
  ]
  const maxBinCount = Math.max(...distributionBins.map(b => b.count), 1)

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold">Lameness Hierarchy</h2>
          <p className="text-muted-foreground mt-1">
            EloSteepness-based ranking with David's Scores from pairwise comparisons
          </p>
        </div>
        <div className="flex items-center gap-4">
          <button
            onClick={() => setShowSnapshotModal(true)}
            className="px-4 py-2 border rounded-lg hover:bg-accent"
          >
            Save Snapshot
          </button>
          <button
            onClick={handleRecalculate}
            className="px-4 py-2 border rounded-lg hover:bg-accent text-orange-600"
          >
            Recalculate
          </button>
        </div>
      </div>

      {/* Hierarchy Metrics - Based on Research Paper */}
      {metrics && (
        <div className="bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Hierarchy Quality Metrics</h3>
          <div className="grid grid-cols-4 gap-6">
            <div className="text-center">
              <div className="text-3xl font-bold text-blue-600">
                {metrics.steepness.toFixed(3)}
              </div>
              <div className="text-sm text-muted-foreground">Steepness</div>
              <div className="text-xs text-gray-500 mt-1">
                (SE: {metrics.steepness_se.toFixed(3)})
              </div>
            </div>
            <div className="text-center">
              <div className={`text-3xl font-bold ${
                metrics.steepness > 0.7 ? 'text-green-600' :
                metrics.steepness > 0.4 ? 'text-yellow-600' : 'text-red-600'
              }`}>
                {metrics.hierarchy_linearity}
              </div>
              <div className="text-sm text-muted-foreground">Hierarchy Linearity</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-purple-600">
                {(metrics.inter_rater_agreement * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-muted-foreground">Inter-Rater Agreement</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-gray-700">
                {stats?.total_comparisons || 0}
              </div>
              <div className="text-sm text-muted-foreground">Total Comparisons</div>
            </div>
          </div>

          {/* Steepness interpretation */}
          <div className="mt-4 p-3 bg-white/50 rounded-lg">
            <p className="text-sm text-gray-600">
              <strong>Steepness</strong> measures hierarchy linearity (0-1). Values &gt;0.7 indicate a clear,
              linear hierarchy. Based on the EloSteepness methodology from the research paper.
            </p>
          </div>
        </div>
      )}

      {/* Stats and Filters Row */}
      <div className="flex justify-between items-center">
        {/* Stats Summary */}
        <div className="flex gap-4">
          <div className="border rounded-lg px-4 py-2 text-center">
            <div className="text-xl font-bold">{ranking.length}</div>
            <div className="text-xs text-muted-foreground">Videos</div>
          </div>
          <div className="border rounded-lg px-4 py-2 text-center bg-red-50">
            <div className="text-xl font-bold text-red-600">
              {ranking.filter(r => r.category === 'lame').length}
            </div>
            <div className="text-xs text-red-600">Lame</div>
          </div>
          <div className="border rounded-lg px-4 py-2 text-center bg-yellow-50">
            <div className="text-xl font-bold text-yellow-600">
              {ranking.filter(r => r.category === 'intermediate').length}
            </div>
            <div className="text-xs text-yellow-600">Intermediate</div>
          </div>
          <div className="border rounded-lg px-4 py-2 text-center bg-green-50">
            <div className="text-xl font-bold text-green-600">
              {ranking.filter(r => r.category === 'healthy').length}
            </div>
            <div className="text-xs text-green-600">Healthy</div>
          </div>
        </div>

        {/* Filters */}
        <div className="flex items-center gap-4">
          <select
            value={categoryFilter}
            onChange={(e) => setCategoryFilter(e.target.value)}
            className="px-3 py-2 border rounded-lg"
          >
            <option value="all">All Categories</option>
            <option value="lame">Lame Only</option>
            <option value="intermediate">Intermediate Only</option>
            <option value="healthy">Healthy Only</option>
          </select>

          <div className="flex border rounded-lg overflow-hidden">
            {['list', 'bar', 'davids', 'distribution'].map((mode) => (
              <button
                key={mode}
                onClick={() => setViewMode(mode as any)}
                className={`px-4 py-2 text-sm capitalize ${
                  viewMode === mode
                    ? 'bg-primary text-primary-foreground'
                    : 'hover:bg-accent'
                }`}
              >
                {mode === 'davids' ? "David's" : mode}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Bar Chart View - Elo Ratings */}
      {viewMode === 'bar' && (
        <div className="border rounded-lg p-6" ref={chartRef}>
          <h3 className="text-lg font-semibold mb-4">Elo Rating Distribution</h3>
          <div className="space-y-2">
            {filteredRanking.map((item) => {
              const barWidth = ((item.elo_rating - minElo) / eloRange) * 100
              const barColor = item.category === 'lame' ? 'bg-red-500' :
                               item.category === 'healthy' ? 'bg-green-500' : 'bg-yellow-500'

              return (
                <div
                  key={item.video_id}
                  className="flex items-center gap-3 group cursor-pointer"
                  onMouseEnter={(e) => handleVideoHover(e, item.video_id)}
                  onMouseLeave={() => setHoveredVideo(null)}
                  onClick={() => {
                    setSelectedVideo(item.video_id)
                    loadVideoHistory(item.video_id)
                  }}
                >
                  <div className="w-8 text-right text-sm text-muted-foreground">
                    #{item.rank}
                  </div>
                  <div className="flex-1 bg-gray-100 rounded-full h-6 overflow-hidden relative">
                    <div
                      className={`h-full ${barColor} transition-all group-hover:opacity-80`}
                      style={{
                        width: `${Math.max(5, barWidth)}%`,
                        opacity: 0.5 + item.confidence * 0.5
                      }}
                    />
                    {/* Uncertainty indicator */}
                    <div
                      className="absolute top-0 h-full bg-black/10"
                      style={{
                        left: `${Math.max(0, barWidth - (item.elo_uncertainty / 10))}%`,
                        width: `${Math.min(100 - barWidth, item.elo_uncertainty / 5)}%`
                      }}
                    />
                  </div>
                  <div className="w-24 text-right">
                    <span className={`font-medium ${getCategoryColor(item.category)}`}>
                      {item.elo_rating.toFixed(0)}
                    </span>
                    <span className="text-xs text-gray-400 ml-1">
                      ±{item.elo_uncertainty.toFixed(0)}
                    </span>
                  </div>
                  <div className={`w-24 px-2 py-1 rounded text-xs text-center ${getCategoryBg(item.category)}`}>
                    {item.category}
                  </div>
                </div>
              )
            })}
          </div>

          {/* Legend */}
          <div className="mt-6 flex justify-center gap-6">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-green-500 rounded"></div>
              <span className="text-sm">Healthy (&lt;1450)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-yellow-500 rounded"></div>
              <span className="text-sm">Intermediate</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-red-500 rounded"></div>
              <span className="text-sm">Lame (&gt;1550)</span>
            </div>
          </div>
        </div>
      )}

      {/* David's Score View */}
      {viewMode === 'davids' && (
        <div className="border rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-2">David's Score Distribution</h3>
          <p className="text-sm text-muted-foreground mb-4">
            David's Score accounts for win quality - wins against strong opponents count more.
            Range: 0 (most healthy) to 1 (most lame).
          </p>
          <div className="space-y-2">
            {filteredRanking
              .sort((a, b) => b.davids_score - a.davids_score)
              .map((item, idx) => {
                const barWidth = item.davids_score * 100
                const barColor = item.davids_score > 0.6 ? 'bg-red-500' :
                                 item.davids_score < 0.4 ? 'bg-green-500' : 'bg-yellow-500'

                return (
                  <div
                    key={item.video_id}
                    className="flex items-center gap-3 group cursor-pointer"
                    onMouseEnter={(e) => handleVideoHover(e, item.video_id)}
                    onMouseLeave={() => setHoveredVideo(null)}
                    onClick={() => {
                      setSelectedVideo(item.video_id)
                      loadVideoHistory(item.video_id)
                    }}
                  >
                    <div className="w-8 text-right text-sm text-muted-foreground">
                      #{idx + 1}
                    </div>
                    <div className="flex-1 bg-gray-100 rounded-full h-6 overflow-hidden">
                      <div
                        className={`h-full ${barColor} transition-all group-hover:opacity-80`}
                        style={{ width: `${Math.max(5, barWidth)}%` }}
                      />
                    </div>
                    <div className="w-20 text-right font-mono">
                      {item.davids_score.toFixed(3)}
                    </div>
                    <div className="w-20 text-center text-sm text-muted-foreground">
                      {item.wins}W/{item.losses}L/{item.ties}T
                    </div>
                  </div>
                )
              })}
          </div>
        </div>
      )}

      {/* Distribution View */}
      {viewMode === 'distribution' && (
        <div className="border rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Category Distribution</h3>
          <div className="flex items-end justify-center gap-8 h-64">
            {distributionBins.map((bin) => (
              <div key={bin.label} className="flex flex-col items-center">
                <div
                  className={`w-24 ${bin.color} rounded-t-lg transition-all`}
                  style={{ height: `${(bin.count / maxBinCount) * 200}px` }}
                />
                <div className="mt-2 text-center">
                  <div className="font-bold text-2xl">{bin.count}</div>
                  <div className="text-xs text-muted-foreground max-w-[100px]">{bin.label}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* List View */}
      {viewMode === 'list' && (
        <div className="border rounded-lg overflow-hidden">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left text-sm font-medium">Rank</th>
                <th className="px-4 py-3 text-left text-sm font-medium">Video ID</th>
                <th className="px-4 py-3 text-left text-sm font-medium">Elo Rating</th>
                <th className="px-4 py-3 text-left text-sm font-medium">David's Score</th>
                <th className="px-4 py-3 text-left text-sm font-medium">Category</th>
                <th className="px-4 py-3 text-left text-sm font-medium">W/L/T</th>
                <th className="px-4 py-3 text-left text-sm font-medium">Confidence</th>
                <th className="px-4 py-3 text-left text-sm font-medium">Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredRanking.map((item) => (
                <tr
                  key={item.video_id}
                  className="border-t hover:bg-gray-50 cursor-pointer"
                  onClick={() => {
                    setSelectedVideo(item.video_id)
                    loadVideoHistory(item.video_id)
                  }}
                  onMouseEnter={(e) => handleVideoHover(e, item.video_id)}
                  onMouseLeave={() => setHoveredVideo(null)}
                >
                  <td className="px-4 py-3">
                    <span className="w-8 h-8 flex items-center justify-center bg-primary text-primary-foreground rounded-full text-sm font-bold">
                      {item.rank}
                    </span>
                  </td>
                  <td className="px-4 py-3 font-mono text-sm">
                    {item.video_id.slice(0, 12)}...
                  </td>
                  <td className={`px-4 py-3 font-medium ${getCategoryColor(item.category)}`}>
                    {item.elo_rating.toFixed(0)}
                    <span className="text-xs text-gray-400 ml-1">±{item.elo_uncertainty.toFixed(0)}</span>
                  </td>
                  <td className="px-4 py-3 font-mono text-sm">
                    {item.davids_score.toFixed(3)}
                  </td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-1 rounded text-xs ${getCategoryBg(item.category)} ${getCategoryColor(item.category)}`}>
                      {item.category}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-sm">
                    <span className="text-green-600">{item.wins}</span>/
                    <span className="text-red-600">{item.losses}</span>/
                    <span className="text-gray-600">{item.ties}</span>
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <div className="w-16 bg-gray-200 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full ${
                            item.confidence > 0.7 ? 'bg-green-500' :
                            item.confidence > 0.4 ? 'bg-yellow-500' : 'bg-red-500'
                          }`}
                          style={{ width: `${item.confidence * 100}%` }}
                        />
                      </div>
                      <span className="text-xs text-muted-foreground">
                        {(item.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <button
                      className="text-sm text-blue-600 hover:underline"
                      onClick={(e) => {
                        e.stopPropagation()
                        window.open(`/analysis/${item.video_id}`, '_blank')
                      }}
                    >
                      View Analysis
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Snapshots Section */}
      {snapshots.length > 0 && (
        <div className="border rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Saved Snapshots</h3>
          <div className="grid grid-cols-3 gap-4">
            {snapshots.map((snapshot) => (
              <div key={snapshot.id} className="border rounded-lg p-4 hover:bg-gray-50">
                <div className="font-medium">{snapshot.name}</div>
                {snapshot.description && (
                  <div className="text-sm text-muted-foreground mt-1">{snapshot.description}</div>
                )}
                <div className="mt-2 grid grid-cols-2 gap-2 text-sm">
                  <div>Videos: {snapshot.total_videos}</div>
                  <div>Comparisons: {snapshot.total_comparisons}</div>
                  <div>Steepness: {snapshot.steepness?.toFixed(3)}</div>
                  <div>IRR: {(snapshot.inter_rater_reliability * 100)?.toFixed(1)}%</div>
                </div>
                <div className="mt-2 text-xs text-muted-foreground">
                  {new Date(snapshot.created_at).toLocaleString()}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Video Preview Popup */}
      {hoveredVideo && (
        <div
          className="fixed z-50 bg-white rounded-lg shadow-xl border overflow-hidden"
          style={{
            left: hoveredVideo.x - 150,
            top: hoveredVideo.y - 180,
            width: 300
          }}
        >
          <video
            src={videosApi.getStreamUrl(hoveredVideo.video_id)}
            className="w-full aspect-video bg-black"
            autoPlay
            muted
            loop
          />
          <div className="p-2 text-center text-sm text-muted-foreground">
            {hoveredVideo.video_id.slice(0, 16)}...
          </div>
        </div>
      )}

      {/* Selected Video Modal */}
      {selectedVideo && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg max-w-3xl w-full mx-4 overflow-hidden max-h-[90vh] overflow-y-auto">
            <div className="p-4 border-b flex justify-between items-center sticky top-0 bg-white">
              <h3 className="font-semibold">Video Details</h3>
              <button
                onClick={() => {
                  setSelectedVideo(null)
                  setVideoHistory(null)
                }}
                className="text-gray-500 hover:text-gray-700"
              >
                X
              </button>
            </div>
            <div className="p-4">
              <video
                src={videosApi.getStreamUrl(selectedVideo)}
                className="w-full aspect-video bg-black rounded-lg"
                controls
                autoPlay
              />

              {/* Video Stats */}
              <div className="mt-4 grid grid-cols-2 gap-4">
                {ranking.filter(r => r.video_id === selectedVideo).map(item => (
                  <div key={item.video_id} className="col-span-2">
                    <div className="grid grid-cols-5 gap-4">
                      <div className="text-center p-3 bg-gray-50 rounded">
                        <div className="text-2xl font-bold">#{item.rank}</div>
                        <div className="text-xs text-muted-foreground">Rank</div>
                      </div>
                      <div className={`text-center p-3 rounded ${getCategoryBg(item.category)}`}>
                        <div className={`text-2xl font-bold ${getCategoryColor(item.category)}`}>
                          {item.elo_rating.toFixed(0)}
                        </div>
                        <div className="text-xs text-muted-foreground">Elo Rating</div>
                      </div>
                      <div className="text-center p-3 bg-blue-50 rounded">
                        <div className="text-2xl font-bold text-blue-600">
                          {item.davids_score.toFixed(3)}
                        </div>
                        <div className="text-xs text-muted-foreground">David's Score</div>
                      </div>
                      <div className="text-center p-3 bg-gray-50 rounded">
                        <div className="text-xl font-bold">
                          <span className="text-green-600">{item.wins}</span>/
                          <span className="text-red-600">{item.losses}</span>/
                          <span className="text-gray-600">{item.ties}</span>
                        </div>
                        <div className="text-xs text-muted-foreground">W/L/T</div>
                      </div>
                      <div className="text-center p-3 bg-gray-50 rounded">
                        <div className="text-2xl font-bold">{(item.confidence * 100).toFixed(0)}%</div>
                        <div className="text-xs text-muted-foreground">Confidence</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Elo History Chart */}
              {videoHistory && videoHistory.history?.length > 0 && (
                <div className="mt-4 border rounded-lg p-4">
                  <h4 className="font-medium mb-3">Elo Rating History</h4>
                  <div className="h-32 flex items-end gap-1">
                    {videoHistory.history.slice(-30).map((h: any, idx: number) => {
                      const minH = Math.min(...videoHistory.history.map((x: any) => x.elo_rating))
                      const maxH = Math.max(...videoHistory.history.map((x: any) => x.elo_rating))
                      const range = maxH - minH || 1
                      const height = ((h.elo_rating - minH) / range) * 100

                      return (
                        <div
                          key={idx}
                          className="flex-1 bg-blue-500 rounded-t transition-all hover:bg-blue-600"
                          style={{ height: `${Math.max(5, height)}%` }}
                          title={`${h.elo_rating.toFixed(0)} - ${new Date(h.recorded_at).toLocaleDateString()}`}
                        />
                      )
                    })}
                  </div>
                  <div className="mt-2 text-xs text-muted-foreground text-center">
                    Last {Math.min(30, videoHistory.history.length)} comparisons
                  </div>
                </div>
              )}

              <div className="mt-4 flex gap-2">
                <button
                  onClick={() => window.open(`/analysis/${selectedVideo}`, '_blank')}
                  className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg"
                >
                  Full Analysis
                </button>
                <button
                  onClick={() => {
                    setSelectedVideo(null)
                    setVideoHistory(null)
                  }}
                  className="flex-1 px-4 py-2 border rounded-lg"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Create Snapshot Modal */}
      {showSnapshotModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg max-w-md w-full mx-4 p-6">
            <h3 className="text-lg font-semibold mb-4">Create Hierarchy Snapshot</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1">Name</label>
                <input
                  type="text"
                  value={snapshotName}
                  onChange={(e) => setSnapshotName(e.target.value)}
                  className="w-full px-3 py-2 border rounded-lg"
                  placeholder="e.g., Week 1 Assessment"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Description (optional)</label>
                <textarea
                  value={snapshotDescription}
                  onChange={(e) => setSnapshotDescription(e.target.value)}
                  className="w-full px-3 py-2 border rounded-lg"
                  rows={3}
                  placeholder="Notes about this snapshot..."
                />
              </div>
              <div className="flex gap-2">
                <button
                  onClick={handleCreateSnapshot}
                  disabled={!snapshotName.trim()}
                  className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg disabled:opacity-50"
                >
                  Create Snapshot
                </button>
                <button
                  onClick={() => setShowSnapshotModal(false)}
                  className="flex-1 px-4 py-2 border rounded-lg"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
