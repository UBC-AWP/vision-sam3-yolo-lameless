import { useEffect, useState, useRef, useCallback } from 'react'
import { trainingApi, videosApi } from '@/api/client'

interface RankingItem {
  video_id: string
  rank: number
  elo_rating: number
  comparisons: number
  confidence: number
  category: 'lame' | 'intermediate' | 'healthy'
}

interface VideoPreview {
  video_id: string
  x: number
  y: number
}

export default function HierarchyVisualization() {
  const [ranking, setRanking] = useState<RankingItem[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedVideo, setSelectedVideo] = useState<string | null>(null)
  const [hoveredVideo, setHoveredVideo] = useState<VideoPreview | null>(null)
  const [viewMode, setViewMode] = useState<'list' | 'bar' | 'distribution'>('bar')
  const [categoryFilter, setCategoryFilter] = useState<string>('all')
  const chartRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    loadRanking()
  }, [])

  const loadRanking = async () => {
    setLoading(true)
    try {
      const data = await trainingApi.getPairwiseRanking()
      
      // Categorize based on Elo rating
      const categorizedRanking = data.ranking.map((item: any) => ({
        ...item,
        category: item.elo_rating > 1550 ? 'lame' :
                  item.elo_rating < 1450 ? 'healthy' : 'intermediate',
        confidence: calculateConfidence(item.comparisons)
      }))
      
      setRanking(categorizedRanking)
    } catch (error) {
      console.error('Failed to load ranking:', error)
    } finally {
      setLoading(false)
    }
  }

  const calculateConfidence = (comparisons: number): number => {
    // More comparisons = higher confidence
    return Math.min(1, comparisons / 20)
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
            EloSteepness-based ranking from pairwise comparisons
          </p>
        </div>
        <div className="flex items-center gap-4">
          {/* Category Filter */}
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
          
          {/* View Mode */}
          <div className="flex border rounded-lg overflow-hidden">
            {['list', 'bar', 'distribution'].map((mode) => (
              <button
                key={mode}
                onClick={() => setViewMode(mode as any)}
                className={`px-4 py-2 text-sm capitalize ${
                  viewMode === mode 
                    ? 'bg-primary text-primary-foreground' 
                    : 'hover:bg-accent'
                }`}
              >
                {mode}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Stats Summary */}
      <div className="grid grid-cols-4 gap-4">
        <div className="border rounded-lg p-4 text-center">
          <div className="text-3xl font-bold">{ranking.length}</div>
          <div className="text-sm text-muted-foreground">Total Videos</div>
        </div>
        <div className="border rounded-lg p-4 text-center bg-red-50">
          <div className="text-3xl font-bold text-red-600">
            {ranking.filter(r => r.category === 'lame').length}
          </div>
          <div className="text-sm text-red-600">Lame</div>
        </div>
        <div className="border rounded-lg p-4 text-center bg-yellow-50">
          <div className="text-3xl font-bold text-yellow-600">
            {ranking.filter(r => r.category === 'intermediate').length}
          </div>
          <div className="text-sm text-yellow-600">Intermediate</div>
        </div>
        <div className="border rounded-lg p-4 text-center bg-green-50">
          <div className="text-3xl font-bold text-green-600">
            {ranking.filter(r => r.category === 'healthy').length}
          </div>
          <div className="text-sm text-green-600">Healthy</div>
        </div>
      </div>

      {/* Insufficient Data Warning */}
      {ranking.some(r => r.confidence < 0.5) && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 flex items-start gap-3">
          <span className="text-yellow-600 text-xl">⚠️</span>
          <div>
            <h4 className="font-semibold text-yellow-800">Insufficient Data</h4>
            <p className="text-sm text-yellow-700">
              Some videos have low confidence scores due to limited comparisons.
              Continue labeling to improve ranking accuracy.
            </p>
          </div>
        </div>
      )}

      {/* Bar Chart View */}
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
                  onClick={() => setSelectedVideo(item.video_id)}
                >
                  <div className="w-8 text-right text-sm text-muted-foreground">
                    #{item.rank}
                  </div>
                  <div className="flex-1 bg-gray-100 rounded-full h-6 overflow-hidden">
                    <div
                      className={`h-full ${barColor} transition-all group-hover:opacity-80`}
                      style={{ 
                        width: `${Math.max(5, barWidth)}%`,
                        opacity: 0.5 + item.confidence * 0.5
                      }}
                    />
                  </div>
                  <div className="w-20 text-right">
                    <span className={`font-medium ${getCategoryColor(item.category)}`}>
                      {item.elo_rating}
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
                <th className="px-4 py-3 text-left text-sm font-medium">Category</th>
                <th className="px-4 py-3 text-left text-sm font-medium">Comparisons</th>
                <th className="px-4 py-3 text-left text-sm font-medium">Confidence</th>
                <th className="px-4 py-3 text-left text-sm font-medium">Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredRanking.map((item) => (
                <tr 
                  key={item.video_id}
                  className="border-t hover:bg-gray-50 cursor-pointer"
                  onClick={() => setSelectedVideo(item.video_id)}
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
                    {item.elo_rating}
                  </td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-1 rounded text-xs ${getCategoryBg(item.category)} ${getCategoryColor(item.category)}`}>
                      {item.category}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-sm text-muted-foreground">
                    {item.comparisons}
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
          <div className="bg-white rounded-lg max-w-2xl w-full mx-4 overflow-hidden">
            <div className="p-4 border-b flex justify-between items-center">
              <h3 className="font-semibold">Video Details</h3>
              <button 
                onClick={() => setSelectedVideo(null)}
                className="text-gray-500 hover:text-gray-700"
              >
                ✕
              </button>
            </div>
            <div className="p-4">
              <video
                src={videosApi.getStreamUrl(selectedVideo)}
                className="w-full aspect-video bg-black rounded-lg"
                controls
                autoPlay
              />
              <div className="mt-4 grid grid-cols-2 gap-4">
                {ranking.filter(r => r.video_id === selectedVideo).map(item => (
                  <div key={item.video_id} className="col-span-2 grid grid-cols-4 gap-4">
                    <div className="text-center p-3 bg-gray-50 rounded">
                      <div className="text-2xl font-bold">#{item.rank}</div>
                      <div className="text-xs text-muted-foreground">Rank</div>
                    </div>
                    <div className={`text-center p-3 rounded ${getCategoryBg(item.category)}`}>
                      <div className={`text-2xl font-bold ${getCategoryColor(item.category)}`}>
                        {item.elo_rating}
                      </div>
                      <div className="text-xs text-muted-foreground">Elo Rating</div>
                    </div>
                    <div className="text-center p-3 bg-gray-50 rounded">
                      <div className="text-2xl font-bold">{item.comparisons}</div>
                      <div className="text-xs text-muted-foreground">Comparisons</div>
                    </div>
                    <div className="text-center p-3 bg-gray-50 rounded">
                      <div className="text-2xl font-bold">{(item.confidence * 100).toFixed(0)}%</div>
                      <div className="text-xs text-muted-foreground">Confidence</div>
                    </div>
                  </div>
                ))}
              </div>
              <div className="mt-4 flex gap-2">
                <button
                  onClick={() => window.open(`/analysis/${selectedVideo}`, '_blank')}
                  className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg"
                >
                  Full Analysis
                </button>
                <button
                  onClick={() => setSelectedVideo(null)}
                  className="flex-1 px-4 py-2 border rounded-lg"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

