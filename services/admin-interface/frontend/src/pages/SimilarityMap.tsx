import { useEffect, useState, useRef, useCallback, Suspense } from 'react'
import { analysisApi, videosApi } from '@/api/client'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Html, Text } from '@react-three/drei'
import * as THREE from 'three'
import { Maximize2, Minimize2, X } from 'lucide-react'

interface VideoPoint {
  video_id: string
  x: number
  y: number
  z?: number  // Optional z coordinate for 3D
  label: number  // 0 = sound, 1 = lame, -1 = unknown
  cluster: number
  elo_rating?: number
}

interface HoveredPoint {
  point: VideoPoint
  screenX: number
  screenY: number
}

// 3D Point component
interface Point3DProps {
  point: VideoPoint
  color: string
  isSelected: boolean
  isHovered: boolean
  onHover: (point: VideoPoint | null) => void
  onClick: (point: VideoPoint) => void
}

function Point3D({ point, color, isSelected, isHovered, onHover, onClick }: Point3DProps) {
  const meshRef = useRef<THREE.Mesh>(null)

  useFrame(() => {
    if (meshRef.current) {
      // Gentle floating animation
      meshRef.current.position.y = (point.y - 0.5) * 4 + Math.sin(Date.now() * 0.001 + point.x * 10) * 0.02
    }
  })

  const scale = isSelected ? 1.5 : isHovered ? 1.2 : 1

  return (
    <mesh
      ref={meshRef}
      position={[(point.x - 0.5) * 8, (point.y - 0.5) * 4, ((point.z ?? point.x * point.y) - 0.25) * 6]}
      scale={scale}
      onPointerOver={(e) => {
        e.stopPropagation()
        onHover(point)
      }}
      onPointerOut={() => onHover(null)}
      onClick={(e) => {
        e.stopPropagation()
        onClick(point)
      }}
    >
      <sphereGeometry args={[0.12, 16, 16]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={isHovered || isSelected ? 0.5 : 0.2}
        metalness={0.3}
        roughness={0.7}
      />
      {(isHovered || isSelected) && (
        <mesh>
          <sphereGeometry args={[0.18, 16, 16]} />
          <meshBasicMaterial color={color} transparent opacity={0.3} wireframe />
        </mesh>
      )}
    </mesh>
  )
}

// Cluster visual (ellipsoid)
function ClusterEllipsoid({ points, clusterId, color }: { points: VideoPoint[], clusterId: number, color: string }) {
  if (points.length < 3) return null

  const centroid = {
    x: points.reduce((sum, p) => sum + p.x, 0) / points.length,
    y: points.reduce((sum, p) => sum + p.y, 0) / points.length,
    z: points.reduce((sum, p) => sum + (p.z ?? p.x * p.y), 0) / points.length
  }

  return (
    <mesh position={[(centroid.x - 0.5) * 8, (centroid.y - 0.5) * 4, (centroid.z - 0.25) * 6]}>
      <sphereGeometry args={[1.2, 16, 16]} />
      <meshBasicMaterial color={color} transparent opacity={0.08} />
      <lineSegments>
        <edgesGeometry args={[new THREE.SphereGeometry(1.2, 8, 8)]} />
        <lineBasicMaterial color={color} opacity={0.3} transparent />
      </lineSegments>
    </mesh>
  )
}

// 3D Scene component
interface Scene3DProps {
  points: VideoPoint[]
  colorBy: 'label' | 'cluster' | 'elo'
  showLabelsOnly: boolean
  selectedPoint: VideoPoint | null
  hoveredPoint3D: VideoPoint | null
  onHover: (point: VideoPoint | null) => void
  onClick: (point: VideoPoint) => void
  getPointColor: (point: VideoPoint) => string
}

function Scene3D({ points, colorBy, showLabelsOnly, selectedPoint, hoveredPoint3D, onHover, onClick, getPointColor }: Scene3DProps) {
  const displayPoints = showLabelsOnly ? points.filter(p => p.label !== -1) : points

  // Group by cluster for cluster visualization
  const clusters = new Map<number, VideoPoint[]>()
  if (colorBy === 'cluster') {
    displayPoints.forEach(p => {
      if (!clusters.has(p.cluster)) clusters.set(p.cluster, [])
      clusters.get(p.cluster)!.push(p)
    })
  }

  const clusterColors = ['#3b82f6', '#f59e0b', '#8b5cf6', '#ec4899', '#14b8a6']

  return (
    <>
      {/* Ambient and directional lights */}
      <ambientLight intensity={0.6} />
      <directionalLight position={[10, 10, 5]} intensity={0.8} />
      <directionalLight position={[-10, -10, -5]} intensity={0.4} />
      <pointLight position={[0, 5, 0]} intensity={0.5} color="#ffffff" />

      {/* Grid helper */}
      <gridHelper args={[10, 20, '#e5e7eb', '#e5e7eb']} position={[0, -2.5, 0]} />

      {/* Axis helper */}
      <axesHelper args={[5]} />

      {/* Cluster ellipsoids */}
      {colorBy === 'cluster' && Array.from(clusters.entries()).map(([clusterId, clusterPoints]) => (
        <ClusterEllipsoid
          key={clusterId}
          points={clusterPoints}
          clusterId={clusterId}
          color={clusterColors[clusterId % clusterColors.length]}
        />
      ))}

      {/* Points */}
      {displayPoints.map(point => (
        <Point3D
          key={point.video_id}
          point={point}
          color={getPointColor(point)}
          isSelected={selectedPoint?.video_id === point.video_id}
          isHovered={hoveredPoint3D?.video_id === point.video_id}
          onHover={onHover}
          onClick={onClick}
        />
      ))}

      {/* Hover tooltip in 3D space */}
      {hoveredPoint3D && (
        <Html
          position={[
            (hoveredPoint3D.x - 0.5) * 8 + 0.3,
            (hoveredPoint3D.y - 0.5) * 4 + 0.3,
            ((hoveredPoint3D.z ?? hoveredPoint3D.x * hoveredPoint3D.y) - 0.25) * 6
          ]}
          distanceFactor={8}
        >
          <div className="bg-white/95 backdrop-blur rounded-lg shadow-xl border p-2 pointer-events-none text-xs min-w-32">
            <div className="font-mono font-medium">{hoveredPoint3D.video_id.slice(0, 16)}...</div>
            <div className="text-muted-foreground mt-1">
              {hoveredPoint3D.label === 0 ? 'Healthy' : hoveredPoint3D.label === 1 ? 'Lame' : 'Unknown'}
            </div>
            {hoveredPoint3D.elo_rating && (
              <div className="text-muted-foreground">Elo: {hoveredPoint3D.elo_rating}</div>
            )}
          </div>
        </Html>
      )}

      {/* Camera controls */}
      <OrbitControls
        enableDamping
        dampingFactor={0.05}
        minDistance={3}
        maxDistance={20}
        maxPolarAngle={Math.PI / 1.5}
      />
    </>
  )
}

export default function SimilarityMap() {
  const [points, setPoints] = useState<VideoPoint[]>([])
  const [loading, setLoading] = useState(true)
  const [hoveredPoint, setHoveredPoint] = useState<HoveredPoint | null>(null)
  const [hoveredPoint3D, setHoveredPoint3D] = useState<VideoPoint | null>(null)
  const [selectedPoint, setSelectedPoint] = useState<VideoPoint | null>(null)
  const [colorBy, setColorBy] = useState<'label' | 'cluster' | 'elo'>('label')
  const [showLabelsOnly, setShowLabelsOnly] = useState(false)
  const [viewMode, setViewMode] = useState<'2d' | '3d'>('2d')
  const [isFullscreen, setIsFullscreen] = useState(false)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const fullscreenContainerRef = useRef<HTMLDivElement>(null)

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

      // Fallback: Load actual videos from the video list API
      try {
        const videosResponse = await videosApi.list(0, 1000)
        const videos = videosResponse.videos || videosResponse || []

        if (videos.length > 0) {
          // Create points with real video IDs
          const realPoints: VideoPoint[] = videos.map((video: any, idx: number) => {
            const cluster = idx % 3
            const baseX = cluster === 0 ? 0.25 : cluster === 1 ? 0.5 : 0.75
            const baseY = cluster === 0 ? 0.3 : cluster === 1 ? 0.7 : 0.4

            return {
              video_id: video.id || video.video_id,
              x: baseX + (Math.random() - 0.5) * 0.2,
              y: baseY + (Math.random() - 0.5) * 0.2,
              z: 0.25 + Math.random() * 0.5,  // Add z for 3D
              label: video.label ?? -1,
              cluster,
              elo_rating: video.elo_rating || (1400 + cluster * 100 + Math.random() * 50)
            }
          })
          setPoints(realPoints)
        } else {
          // Final fallback: empty state
          setPoints([])
        }
      } catch (videoError) {
        console.error('Failed to load videos:', videoError)
        setPoints([])
      }
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

  const toggleFullscreen = useCallback(() => {
    setIsFullscreen(prev => !prev)
  }, [])

  // Handle ESC key to exit fullscreen
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isFullscreen) {
        setIsFullscreen(false)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isFullscreen])

  // Redraw canvas when fullscreen changes
  useEffect(() => {
    if (viewMode === '2d') {
      // Small delay to let the container resize
      const timer = setTimeout(() => drawCanvas(), 100)
      return () => clearTimeout(timer)
    }
  }, [isFullscreen, viewMode, drawCanvas])

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
          <h2 className="text-3xl font-bold">{viewMode === '2d' ? '2D' : '3D'} Similarity Map</h2>
          <p className="text-muted-foreground mt-1">
            MDS projection of DINOv3 embeddings for clustering visualization
          </p>
        </div>
        <div className="flex items-center gap-4">
          {/* 2D/3D Toggle */}
          <div className="flex items-center bg-muted rounded-lg p-1">
            <button
              onClick={() => setViewMode('2d')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                viewMode === '2d'
                  ? 'bg-card shadow-sm text-primary'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              2D
            </button>
            <button
              onClick={() => setViewMode('3d')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                viewMode === '3d'
                  ? 'bg-card shadow-sm text-primary'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              3D
            </button>
          </div>

          {/* Color by selector */}
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">Color by:</span>
            <select
              value={colorBy}
              onChange={(e) => setColorBy(e.target.value as any)}
              className="px-3 py-2 border border-border rounded-lg bg-background text-foreground"
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

          {/* Reset view - only for 2D */}
          {viewMode === '2d' && (
            <button
              onClick={resetView}
              className="px-4 py-2 border rounded-lg hover:bg-accent"
            >
              Reset View
            </button>
          )}
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4">
        <div className="border border-border rounded-lg p-4 text-center bg-card">
          <div className="text-3xl font-bold text-foreground">{points.length}</div>
          <div className="text-sm text-muted-foreground">Total Videos</div>
        </div>
        <div className="border border-success/30 rounded-lg p-4 text-center bg-success/10">
          <div className="text-3xl font-bold text-success">
            {points.filter(p => p.label === 0).length}
          </div>
          <div className="text-sm text-success">Labeled Healthy</div>
        </div>
        <div className="border border-destructive/30 rounded-lg p-4 text-center bg-destructive/10">
          <div className="text-3xl font-bold text-destructive">
            {points.filter(p => p.label === 1).length}
          </div>
          <div className="text-sm text-destructive">Labeled Lame</div>
        </div>
        <div className="border border-border rounded-lg p-4 text-center bg-card">
          <div className="text-3xl font-bold text-muted-foreground">
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
        ref={isFullscreen ? fullscreenContainerRef : containerRef}
        className={`border border-border rounded-lg overflow-hidden bg-muted/30 relative ${
          isFullscreen ? 'fixed inset-0 z-50 rounded-none border-none' : ''
        }`}
        style={{ height: isFullscreen ? '100vh' : '500px' }}
      >
        {/* Fullscreen header bar */}
        {isFullscreen && (
          <div className="absolute top-0 left-0 right-0 z-10 bg-card/95 backdrop-blur border-b border-border px-4 py-3 flex items-center justify-between">
            <div className="flex items-center gap-4">
              <h3 className="font-semibold text-foreground">{viewMode === '2d' ? '2D' : '3D'} Similarity Map</h3>
              <div className="flex items-center bg-muted rounded-lg p-1">
                <button
                  onClick={() => setViewMode('2d')}
                  className={`px-3 py-1 rounded text-sm font-medium transition-all ${
                    viewMode === '2d'
                      ? 'bg-card shadow-sm text-primary'
                      : 'text-muted-foreground hover:text-foreground'
                  }`}
                >
                  2D
                </button>
                <button
                  onClick={() => setViewMode('3d')}
                  className={`px-3 py-1 rounded text-sm font-medium transition-all ${
                    viewMode === '3d'
                      ? 'bg-card shadow-sm text-primary'
                      : 'text-muted-foreground hover:text-foreground'
                  }`}
                >
                  3D
                </button>
              </div>
              <select
                value={colorBy}
                onChange={(e) => setColorBy(e.target.value as any)}
                className="px-3 py-1.5 border border-border rounded-lg text-sm bg-background text-foreground"
              >
                <option value="label">Color: Label</option>
                <option value="cluster">Color: Cluster</option>
                <option value="elo">Color: Elo Rating</option>
              </select>
              {viewMode === '2d' && (
                <button
                  onClick={resetView}
                  className="px-3 py-1.5 border border-border rounded-lg text-sm hover:bg-accent text-foreground"
                >
                  Reset View
                </button>
              )}
            </div>
            <button
              onClick={toggleFullscreen}
              className="p-2 hover:bg-accent rounded-lg text-foreground"
              title="Exit Fullscreen (ESC)"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        )}

        {viewMode === '2d' ? (
          <>
            <canvas
              ref={canvasRef}
              onMouseMove={handleMouseMove}
              onMouseDown={handleMouseDown}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
              onClick={handleClick}
              onWheel={handleWheel}
              className="cursor-crosshair"
              style={{
                width: '100%',
                height: '100%',
                marginTop: isFullscreen ? '56px' : 0
              }}
            />

            {/* Fullscreen toggle button */}
            <button
              onClick={toggleFullscreen}
              className="absolute top-4 right-4 p-2 bg-card/90 hover:bg-card rounded-lg shadow-md transition-all text-foreground"
              title={isFullscreen ? 'Exit Fullscreen (ESC)' : 'Enter Fullscreen'}
            >
              {isFullscreen ? (
                <Minimize2 className="w-5 h-5" />
              ) : (
                <Maximize2 className="w-5 h-5" />
              )}
            </button>

            {/* Zoom indicator */}
            <div className="absolute bottom-4 right-4 bg-card/80 px-3 py-1 rounded text-sm text-foreground">
              Zoom: {(zoom * 100).toFixed(0)}%
            </div>

            {/* Instructions */}
            <div className="absolute bottom-4 left-4 bg-card/80 px-3 py-1 rounded text-xs text-muted-foreground">
              Scroll to zoom • Drag to pan • Click point to select {isFullscreen && '• ESC to exit'}
            </div>
          </>
        ) : (
          <>
            <Canvas
              camera={{ position: [6, 4, 8], fov: 50 }}
              style={{
                background: 'linear-gradient(to bottom, #f0f9ff, #e0f2fe, #f0f9ff)',
                marginTop: isFullscreen ? '56px' : 0,
                height: isFullscreen ? 'calc(100% - 56px)' : '100%'
              }}
            >
              <Suspense fallback={null}>
                <Scene3D
                  points={points}
                  colorBy={colorBy}
                  showLabelsOnly={showLabelsOnly}
                  selectedPoint={selectedPoint}
                  hoveredPoint3D={hoveredPoint3D}
                  onHover={setHoveredPoint3D}
                  onClick={setSelectedPoint}
                  getPointColor={getPointColor}
                />
              </Suspense>
            </Canvas>

            {/* Fullscreen toggle button */}
            {!isFullscreen && (
              <button
                onClick={toggleFullscreen}
                className="absolute top-4 right-4 p-2 bg-card/90 hover:bg-card rounded-lg shadow-md transition-all text-foreground"
                title="Enter Fullscreen"
              >
                <Maximize2 className="w-5 h-5" />
              </button>
            )}

            {/* 3D Instructions */}
            <div className="absolute bottom-4 left-4 bg-card/80 px-3 py-1 rounded text-xs text-muted-foreground">
              Drag to rotate • Scroll to zoom • Right-click to pan • Click point to select {isFullscreen && '• ESC to exit'}
            </div>

            {/* 3D Legend */}
            <div className={`absolute ${isFullscreen ? 'top-20' : 'top-4'} left-4 bg-card/90 backdrop-blur px-3 py-2 rounded-lg text-xs text-foreground`}>
              <div className="flex items-center gap-2 mb-1">
                <div className="w-3 h-0.5 bg-red-500"></div>
                <span>X axis</span>
              </div>
              <div className="flex items-center gap-2 mb-1">
                <div className="w-3 h-0.5 bg-green-500"></div>
                <span>Y axis</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-0.5 bg-blue-500"></div>
                <span>Z axis</span>
              </div>
            </div>
          </>
        )}
      </div>

      {/* Hover tooltip - only for 2D mode */}
      {viewMode === '2d' && hoveredPoint && (
        <div
          className="fixed z-50 bg-card border border-border rounded-lg shadow-xl p-3 pointer-events-none"
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
          <div className="bg-card border border-border rounded-lg max-w-2xl w-full mx-4 overflow-hidden shadow-xl">
            <div className="p-4 border-b border-border flex justify-between items-center">
              <h3 className="font-semibold text-foreground">Video Details</h3>
              <button
                onClick={() => setSelectedPoint(null)}
                className="text-muted-foreground hover:text-foreground"
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
                <div className="text-center p-3 bg-muted rounded">
                  <div className="text-lg font-bold text-foreground">
                    {selectedPoint.label === 0 ? '✓ Healthy' :
                     selectedPoint.label === 1 ? '✗ Lame' : '? Unknown'}
                  </div>
                  <div className="text-xs text-muted-foreground">Label</div>
                </div>
                <div className="text-center p-3 bg-muted rounded">
                  <div className="text-lg font-bold text-foreground">Cluster {selectedPoint.cluster}</div>
                  <div className="text-xs text-muted-foreground">Cluster ID</div>
                </div>
                {selectedPoint.elo_rating && (
                  <div className="text-center p-3 bg-muted rounded">
                    <div className="text-lg font-bold text-foreground">{selectedPoint.elo_rating}</div>
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
                  className="flex-1 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90"
                >
                  Full Analysis
                </button>
                <button
                  onClick={() => setSelectedPoint(null)}
                  className="flex-1 px-4 py-2 border border-border rounded-lg text-foreground hover:bg-accent"
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

