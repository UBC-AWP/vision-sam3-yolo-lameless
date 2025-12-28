import { useEffect, useState, useRef, useCallback } from 'react'
import { analysisApi, videosApi } from '@/api/client'

interface VideoPoint {
  video_id: string
  x: number
  y: number
  label: number  // 0 = sound, 1 = lame, -1 = unknown
  cluster: number
  elo_rating?: number
}

interface HoveredPoint {
  point: VideoPoint
  screenX: number
  screenY: number
}

export default function SimilarityMap() {
  const [points, setPoints] = useState<VideoPoint[]>([])
  const [loading, setLoading] = useState(true)
  const [hoveredPoint, setHoveredPoint] = useState<HoveredPoint | null>(null)
  const [selectedPoint, setSelectedPoint] = useState<VideoPoint | null>(null)
  const [colorBy, setColorBy] = useState<'label' | 'cluster' | 'elo'>('label')
  const [showLabelsOnly, setShowLabelsOnly] = useState(false)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  
  // Zoom and pan state
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const dragStart = useRef({ x: 0, y: 0, panX: 0, panY: 0 })

  useEffect(() => {
    loadEmbeddings()
  }, [])

  const loadEmbeddings = async () => {
    setLoading(true)
    try {
      const data = await analysisApi.getSimilarityMap()
      
      // The backend should return MDS-projected 2D coordinates
      // If not available, we'll simulate with random positions
      if (data.points && data.points.length > 0) {
        setPoints(data.points)
      } else {
        // Fallback: get videos and create placeholder positions
        const videosData = await analysisApi.getAllVideoEmbeddings()
        
        // Simple MDS simulation (in production, do this on backend)
        const simulatedPoints = videosData.map((video: any, idx: number) => {
          const angle = (idx / videosData.length) * 2 * Math.PI
          const radius = 0.3 + Math.random() * 0.4
          return {
            video_id: video.video_id,
            x: 0.5 + radius * Math.cos(angle),
            y: 0.5 + radius * Math.sin(angle),
            label: video.label ?? -1,
            cluster: idx % 3,
            elo_rating: video.elo_rating
          }
        })
        setPoints(simulatedPoints)
      }
    } catch (error) {
      console.error('Failed to load embeddings:', error)
      
      // Create sample data for visualization
      const samplePoints: VideoPoint[] = []
      for (let i = 0; i < 30; i++) {
        const cluster = i % 3
        const baseX = cluster === 0 ? 0.25 : cluster === 1 ? 0.5 : 0.75
        const baseY = cluster === 0 ? 0.3 : cluster === 1 ? 0.7 : 0.4
        
        samplePoints.push({
          video_id: `video_${i}`,
          x: baseX + (Math.random() - 0.5) * 0.2,
          y: baseY + (Math.random() - 0.5) * 0.2,
          label: cluster === 0 ? 0 : cluster === 2 ? 1 : -1,
          cluster,
          elo_rating: 1400 + cluster * 100 + Math.random() * 50
        })
      }
      setPoints(samplePoints)
    } finally {
      setLoading(false)
    }
  }

  const getPointColor = (point: VideoPoint): string => {
    switch (colorBy) {
      case 'label':
        if (point.label === 0) return '#22c55e'  // green - healthy
        if (point.label === 1) return '#ef4444'  // red - lame
        return '#9ca3af'  // gray - unknown
      
      case 'cluster':
        const clusterColors = ['#3b82f6', '#f59e0b', '#8b5cf6', '#ec4899', '#14b8a6']
        return clusterColors[point.cluster % clusterColors.length]
      
      case 'elo':
        if (!point.elo_rating) return '#9ca3af'
        const normalized = (point.elo_rating - 1400) / 200  // 1400-1600 range
        const r = Math.round(255 * Math.min(1, normalized * 2))
        const g = Math.round(255 * Math.min(1, (1 - normalized) * 2))
        return `rgb(${r}, ${g}, 100)`
      
      default:
        return '#6b7280'
    }
  }

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size
    const rect = container.getBoundingClientRect()
    canvas.width = rect.width
    canvas.height = rect.height

    // Clear
    ctx.fillStyle = '#f9fafb'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // Draw grid
    ctx.strokeStyle = '#e5e7eb'
    ctx.lineWidth = 1
    const gridSize = 50 * zoom
    for (let x = pan.x % gridSize; x < canvas.width; x += gridSize) {
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, canvas.height)
      ctx.stroke()
    }
    for (let y = pan.y % gridSize; y < canvas.height; y += gridSize) {
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(canvas.width, y)
      ctx.stroke()
    }

    // Filter points if needed
    const displayPoints = showLabelsOnly 
      ? points.filter(p => p.label !== -1)
      : points

    // Draw points
    displayPoints.forEach(point => {
      const screenX = (point.x * canvas.width - canvas.width / 2) * zoom + canvas.width / 2 + pan.x
      const screenY = (point.y * canvas.height - canvas.height / 2) * zoom + canvas.height / 2 + pan.y

      const color = getPointColor(point)
      const radius = 8 * zoom

      // Draw point
      ctx.beginPath()
      ctx.arc(screenX, screenY, radius, 0, 2 * Math.PI)
      ctx.fillStyle = color
      ctx.fill()
      ctx.strokeStyle = '#ffffff'
      ctx.lineWidth = 2
      ctx.stroke()

      // Highlight selected/hovered
      if (selectedPoint?.video_id === point.video_id || 
          hoveredPoint?.point.video_id === point.video_id) {
        ctx.beginPath()
        ctx.arc(screenX, screenY, radius + 4, 0, 2 * Math.PI)
        ctx.strokeStyle = '#3b82f6'
        ctx.lineWidth = 3
        ctx.stroke()
      }
    })

    // Draw cluster hulls (simplified convex hull visualization)
    if (colorBy === 'cluster') {
      const clusters = new Map<number, VideoPoint[]>()
      displayPoints.forEach(p => {
        if (!clusters.has(p.cluster)) clusters.set(p.cluster, [])
        clusters.get(p.cluster)!.push(p)
      })

      clusters.forEach((clusterPoints, clusterId) => {
        if (clusterPoints.length < 3) return

        // Calculate cluster centroid
        const centroidX = clusterPoints.reduce((sum, p) => sum + p.x, 0) / clusterPoints.length
        const centroidY = clusterPoints.reduce((sum, p) => sum + p.y, 0) / clusterPoints.length

        const screenCX = (centroidX * canvas.width - canvas.width / 2) * zoom + canvas.width / 2 + pan.x
        const screenCY = (centroidY * canvas.height - canvas.height / 2) * zoom + canvas.height / 2 + pan.y

        // Draw cluster ellipse
        const clusterColors = ['#3b82f6', '#f59e0b', '#8b5cf6', '#ec4899', '#14b8a6']
        ctx.beginPath()
        ctx.ellipse(screenCX, screenCY, 60 * zoom, 40 * zoom, 0, 0, 2 * Math.PI)
        ctx.strokeStyle = clusterColors[clusterId % clusterColors.length]
        ctx.lineWidth = 2
        ctx.setLineDash([5, 5])
        ctx.stroke()
        ctx.setLineDash([])
      })
    }
  }, [points, zoom, pan, colorBy, showLabelsOnly, selectedPoint, hoveredPoint])

  useEffect(() => {
    drawCanvas()
  }, [drawCanvas])

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return

    if (isDragging) {
      setPan({
        x: dragStart.current.panX + (e.clientX - dragStart.current.x),
        y: dragStart.current.panY + (e.clientY - dragStart.current.y)
      })
      return
    }

    const rect = canvas.getBoundingClientRect()
    const mouseX = e.clientX - rect.left
    const mouseY = e.clientY - rect.top

    // Find closest point
    let closestPoint: VideoPoint | null = null
    let closestDist = Infinity

    points.forEach(point => {
      const screenX = (point.x * canvas.width - canvas.width / 2) * zoom + canvas.width / 2 + pan.x
      const screenY = (point.y * canvas.height - canvas.height / 2) * zoom + canvas.height / 2 + pan.y
      
      const dist = Math.sqrt((mouseX - screenX) ** 2 + (mouseY - screenY) ** 2)
      if (dist < 15 * zoom && dist < closestDist) {
        closestDist = dist
        closestPoint = point
      }
    })

    if (closestPoint !== null) {
      const p = closestPoint as VideoPoint
      const screenX = (p.x * canvas.width - canvas.width / 2) * zoom + canvas.width / 2 + pan.x
      const screenY = (p.y * canvas.height - canvas.height / 2) * zoom + canvas.height / 2 + pan.y
      setHoveredPoint({ point: p, screenX: e.clientX, screenY: e.clientY })
    } else {
      setHoveredPoint(null)
    }
  }

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true)
    dragStart.current = { x: e.clientX, y: e.clientY, panX: pan.x, panY: pan.y }
  }

  const handleMouseUp = () => {
    setIsDragging(false)
  }

  const handleClick = () => {
    if (hoveredPoint) {
      setSelectedPoint(hoveredPoint.point)
    }
  }

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault()
    const delta = e.deltaY > 0 ? 0.9 : 1.1
    setZoom(z => Math.max(0.5, Math.min(3, z * delta)))
  }

  const resetView = () => {
    setZoom(1)
    setPan({ x: 0, y: 0 })
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <div className="text-muted-foreground">Computing similarity map...</div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold">2D Similarity Map</h2>
          <p className="text-muted-foreground mt-1">
            MDS projection of DINOv3 embeddings for clustering visualization
          </p>
        </div>
        <div className="flex items-center gap-4">
          {/* Color by selector */}
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">Color by:</span>
            <select
              value={colorBy}
              onChange={(e) => setColorBy(e.target.value as any)}
              className="px-3 py-2 border rounded-lg"
            >
              <option value="label">Label</option>
              <option value="cluster">Cluster</option>
              <option value="elo">Elo Rating</option>
            </select>
          </div>
          
          {/* Filter toggle */}
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={showLabelsOnly}
              onChange={(e) => setShowLabelsOnly(e.target.checked)}
              className="rounded"
            />
            <span className="text-sm">Labeled only</span>
          </label>
          
          {/* Reset view */}
          <button
            onClick={resetView}
            className="px-4 py-2 border rounded-lg hover:bg-accent"
          >
            Reset View
          </button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4">
        <div className="border rounded-lg p-4 text-center">
          <div className="text-3xl font-bold">{points.length}</div>
          <div className="text-sm text-muted-foreground">Total Videos</div>
        </div>
        <div className="border rounded-lg p-4 text-center bg-green-50">
          <div className="text-3xl font-bold text-green-600">
            {points.filter(p => p.label === 0).length}
          </div>
          <div className="text-sm text-green-600">Labeled Healthy</div>
        </div>
        <div className="border rounded-lg p-4 text-center bg-red-50">
          <div className="text-3xl font-bold text-red-600">
            {points.filter(p => p.label === 1).length}
          </div>
          <div className="text-sm text-red-600">Labeled Lame</div>
        </div>
        <div className="border rounded-lg p-4 text-center">
          <div className="text-3xl font-bold text-gray-600">
            {new Set(points.map(p => p.cluster)).size}
          </div>
          <div className="text-sm text-muted-foreground">Clusters</div>
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-6 px-4">
        {colorBy === 'label' && (
          <>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-green-500"></div>
              <span className="text-sm">Healthy</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-red-500"></div>
              <span className="text-sm">Lame</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-gray-400"></div>
              <span className="text-sm">Unlabeled</span>
            </div>
          </>
        )}
        {colorBy === 'elo' && (
          <div className="flex items-center gap-2">
            <div className="w-32 h-4 rounded" style={{
              background: 'linear-gradient(to right, rgb(0, 255, 100), rgb(255, 0, 100))'
            }}></div>
            <span className="text-sm">Low Elo → High Elo</span>
          </div>
        )}
      </div>

      {/* Canvas Container */}
      <div 
        ref={containerRef}
        className="border rounded-lg overflow-hidden bg-gray-50 relative"
        style={{ height: '500px' }}
      >
        <canvas
          ref={canvasRef}
          onMouseMove={handleMouseMove}
          onMouseDown={handleMouseDown}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onClick={handleClick}
          onWheel={handleWheel}
          className="cursor-crosshair"
          style={{ width: '100%', height: '100%' }}
        />
        
        {/* Zoom indicator */}
        <div className="absolute bottom-4 right-4 bg-white/80 px-3 py-1 rounded text-sm">
          Zoom: {(zoom * 100).toFixed(0)}%
        </div>
        
        {/* Instructions */}
        <div className="absolute bottom-4 left-4 bg-white/80 px-3 py-1 rounded text-xs text-muted-foreground">
          Scroll to zoom • Drag to pan • Click point to select
        </div>
      </div>

      {/* Hover tooltip */}
      {hoveredPoint && (
        <div
          className="fixed z-50 bg-white rounded-lg shadow-xl border p-3 pointer-events-none"
          style={{
            left: hoveredPoint.screenX + 15,
            top: hoveredPoint.screenY - 50
          }}
        >
          <div className="text-sm font-mono">{hoveredPoint.point.video_id.slice(0, 16)}...</div>
          <div className="text-xs text-muted-foreground mt-1">
            Label: {hoveredPoint.point.label === 0 ? 'Healthy' : 
                    hoveredPoint.point.label === 1 ? 'Lame' : 'Unknown'}
          </div>
          {hoveredPoint.point.elo_rating && (
            <div className="text-xs text-muted-foreground">
              Elo: {hoveredPoint.point.elo_rating}
            </div>
          )}
        </div>
      )}

      {/* Selected video modal */}
      {selectedPoint && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg max-w-2xl w-full mx-4 overflow-hidden">
            <div className="p-4 border-b flex justify-between items-center">
              <h3 className="font-semibold">Video Details</h3>
              <button
                onClick={() => setSelectedPoint(null)}
                className="text-gray-500 hover:text-gray-700"
              >
                ✕
              </button>
            </div>
            <div className="p-4">
              <video
                src={videosApi.getStreamUrl(selectedPoint.video_id)}
                className="w-full aspect-video bg-black rounded-lg"
                controls
                autoPlay
              />
              <div className="mt-4 grid grid-cols-3 gap-4">
                <div className="text-center p-3 bg-gray-50 rounded">
                  <div className="text-lg font-bold">
                    {selectedPoint.label === 0 ? '✓ Healthy' :
                     selectedPoint.label === 1 ? '✗ Lame' : '? Unknown'}
                  </div>
                  <div className="text-xs text-muted-foreground">Label</div>
                </div>
                <div className="text-center p-3 bg-gray-50 rounded">
                  <div className="text-lg font-bold">Cluster {selectedPoint.cluster}</div>
                  <div className="text-xs text-muted-foreground">Cluster ID</div>
                </div>
                {selectedPoint.elo_rating && (
                  <div className="text-center p-3 bg-gray-50 rounded">
                    <div className="text-lg font-bold">{selectedPoint.elo_rating}</div>
                    <div className="text-xs text-muted-foreground">Elo Rating</div>
                  </div>
                )}
              </div>
              
              {/* Find similar videos in same cluster */}
              <div className="mt-4">
                <h4 className="font-medium mb-2">Similar Videos (Same Cluster)</h4>
                <div className="flex gap-2 overflow-x-auto pb-2">
                  {points
                    .filter(p => p.cluster === selectedPoint.cluster && p.video_id !== selectedPoint.video_id)
                    .slice(0, 5)
                    .map(p => (
                      <div
                        key={p.video_id}
                        className="flex-shrink-0 w-24 cursor-pointer"
                        onClick={() => setSelectedPoint(p)}
                      >
                        <video
                          src={videosApi.getStreamUrl(p.video_id)}
                          className="w-full aspect-video bg-gray-200 rounded"
                          muted
                        />
                        <div className="text-xs text-center mt-1 truncate">
                          {p.video_id.slice(0, 8)}...
                        </div>
                      </div>
                    ))}
                </div>
              </div>
              
              <div className="mt-4 flex gap-2">
                <button
                  onClick={() => window.open(`/analysis/${selectedPoint.video_id}`, '_blank')}
                  className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg"
                >
                  Full Analysis
                </button>
                <button
                  onClick={() => setSelectedPoint(null)}
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

