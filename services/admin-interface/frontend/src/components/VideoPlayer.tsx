import { useRef, useState, useEffect, useCallback } from 'react'
import { videosApi } from '@/api/client'

interface VideoPlayerProps {
  videoId: string
  showAnnotations?: boolean
  onFrameChange?: (frame: number) => void
  onTimeUpdate?: (time: number) => void
}

export default function VideoPlayer({
  videoId,
  showAnnotations = false,
  onFrameChange,
  onTimeUpdate,
}: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [playbackSpeed, setPlaybackSpeed] = useState(1)
  const [showAnnotated, setShowAnnotated] = useState(showAnnotations)
  const [hasAnnotated, setHasAnnotated] = useState(false)
  const [isAnnotating, setIsAnnotating] = useState(false)
  const [annotationProgress, setAnnotationProgress] = useState(0)
  const [annotationMessage, setAnnotationMessage] = useState('')
  const [videoMetadata, setVideoMetadata] = useState<any>(null)
  const [streamUrl, setStreamUrl] = useState<string | null>(null)

  // Load video metadata, stream URL, and check for annotations
  useEffect(() => {
    const loadVideoInfo = async () => {
      try {
        const info = await videosApi.get(videoId)
        setVideoMetadata(info.metadata)
        setHasAnnotated(info.has_annotated)
        // Use stream_url from API if available (S3 pre-signed URL)
        if (info.stream_url) {
          setStreamUrl(info.stream_url)
        }
      } catch (error) {
        console.error('Failed to load video info:', error)
      }
    }
    loadVideoInfo()
  }, [videoId])

  // Check annotation status periodically when annotating
  useEffect(() => {
    if (!isAnnotating) return

    let tleapStartTime = Date.now()

    const checkStatus = async () => {
      try {
        const status = await videosApi.getAnnotationStatus(videoId)
        console.log('Annotation status:', status)
        
        // Handle both "complete" and "completed" status
        if (status.status === 'complete' || status.status === 'completed') {
          setIsAnnotating(false)
          setHasAnnotated(true)
          setAnnotationProgress(100)
          setAnnotationMessage('Complete!')
          setShowAnnotated(true) // Auto-show annotations when complete
        } else if (status.status === 'rendering' || status.status === 'processing') {
          // Rendering phase: show 50-100% (annotation rendering after T-LEAP)
          const renderProgress = Math.round(status.progress || 0)
          const adjustedProgress = 50 + (renderProgress * 0.5) // 50-100%
          setAnnotationProgress(adjustedProgress)
          setAnnotationMessage(`Rendering video... ${renderProgress}%`)
        } else if (status.status === 'starting') {
          setAnnotationProgress(45) // Starting render after T-LEAP
          setAnnotationMessage('Starting video render...')
        } else if (status.status === 'not_found') {
          // T-LEAP is processing - show progress 0-50% based on time elapsed
          // Typical T-LEAP takes 10-30 seconds for short videos
          const elapsed = (Date.now() - tleapStartTime) / 1000
          const estimatedProgress = Math.min(45, Math.round(elapsed * 1.5)) // ~30 sec to reach 45%
          setAnnotationProgress(estimatedProgress)
          setAnnotationMessage('Analyzing pose with AI... 🦴')
        } else if (status.status === 'error' || status.status === 'failed') {
          setIsAnnotating(false)
          alert('Annotation failed: ' + (status.error || status.message || 'Unknown error'))
        }
      } catch (error) {
        console.error('Failed to check annotation status:', error)
      }
    }

    // Check immediately then every 500ms for smoother updates
    checkStatus()
    const interval = setInterval(checkStatus, 500)
    return () => clearInterval(interval)
  }, [isAnnotating, videoId])

  // Use S3 stream URL if available, otherwise fall back to local streaming endpoint
  const videoSrc = showAnnotated && hasAnnotated
    ? videosApi.getAnnotatedUrl(videoId)
    : (streamUrl || videosApi.getStreamUrl(videoId))

  // Reload video when source changes
  useEffect(() => {
    if (videoRef.current) {
      const currentTime = videoRef.current.currentTime
      videoRef.current.load()
      // Restore playback position after load
      videoRef.current.onloadeddata = () => {
        if (videoRef.current) {
          videoRef.current.currentTime = currentTime
        }
      }
    }
  }, [videoSrc])

  const handleTimeUpdate = useCallback(() => {
    if (videoRef.current) {
      const time = videoRef.current.currentTime
      setCurrentTime(time)
      onTimeUpdate?.(time)

      if (videoMetadata?.fps) {
        const frame = Math.floor(time * videoMetadata.fps)
        onFrameChange?.(frame)
      }
    }
  }, [videoMetadata, onTimeUpdate, onFrameChange])

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration)
    }
  }

  const togglePlay = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause()
      } else {
        videoRef.current.play()
      }
      setIsPlaying(!isPlaying)
    }
  }

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value)
    if (videoRef.current) {
      videoRef.current.currentTime = time
      setCurrentTime(time)
    }
  }

  const stepFrame = (direction: number) => {
    if (videoRef.current && videoMetadata?.fps) {
      const frameTime = 1 / videoMetadata.fps
      videoRef.current.currentTime = Math.max(
        0,
        Math.min(duration, videoRef.current.currentTime + direction * frameTime)
      )
    }
  }

  const changeSpeed = (speed: number) => {
    if (videoRef.current) {
      videoRef.current.playbackRate = speed
      setPlaybackSpeed(speed)
    }
  }

  const handleTriggerAnnotation = async () => {
    setIsAnnotating(true)
    setAnnotationProgress(0)
    setAnnotationMessage('Starting...')
    try {
      const result = await videosApi.triggerAnnotation(videoId, {
        include_yolo: true,
        include_pose: true,
        show_confidence: true,
        show_labels: true,
      })
      console.log('Annotation triggered:', result)
    } catch (error) {
      console.error('Failed to trigger annotation:', error)
      setIsAnnotating(false)
      alert('Failed to start annotation: ' + (error as Error).message)
    }
  }

  const handleDeleteAnnotation = async () => {
    if (!confirm('Delete annotation and pose data? You can regenerate them later.')) {
      return
    }
    try {
      await videosApi.deleteAnnotation(videoId)
      setHasAnnotated(false)
      setShowAnnotated(false)
      setAnnotationProgress(0)
    } catch (error) {
      console.error('Failed to delete annotation:', error)
      alert('Failed to delete annotation: ' + (error as Error).message)
    }
  }

  const handleRegenerateAnnotation = async () => {
    // First delete, then regenerate
    try {
      await videosApi.deleteAnnotation(videoId)
      setHasAnnotated(false)
      setShowAnnotated(false)
      // Then trigger new annotation
      handleTriggerAnnotation()
    } catch (error) {
      console.error('Failed to regenerate annotation:', error)
      alert('Failed to regenerate annotation: ' + (error as Error).message)
    }
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const currentFrame = videoMetadata?.fps 
    ? Math.floor(currentTime * videoMetadata.fps)
    : 0

  return (
    <div className="bg-black rounded-lg overflow-hidden">
      {/* Video element */}
      <div className="relative aspect-video">
        <video
          ref={videoRef}
          src={videoSrc}
          className="w-full h-full"
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={handleLoadedMetadata}
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
          onEnded={() => setIsPlaying(false)}
        />
        
        {/* Annotation progress overlay */}
        {isAnnotating && (
          <div className="absolute inset-0 bg-black/70 flex items-center justify-center">
            <div className="text-center text-white">
              <div className="mb-3 text-lg font-medium">
                {annotationMessage || 'Generating annotated video...'}
              </div>
              <div className="w-64 bg-gray-700 rounded-full h-3">
                <div
                  className="bg-blue-500 h-3 rounded-full transition-all duration-300"
                  style={{ width: `${annotationProgress}%` }}
                />
              </div>
              <div className="mt-2 text-sm text-gray-300">{annotationProgress}%</div>
            </div>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="p-4 bg-gray-900 space-y-3">
        {/* Progress bar */}
        <div className="flex items-center gap-3">
          <span className="text-white text-sm w-12">{formatTime(currentTime)}</span>
          <input
            type="range"
            min={0}
            max={duration || 100}
            step={0.01}
            value={currentTime}
            onChange={handleSeek}
            className="flex-1 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
          />
          <span className="text-white text-sm w-12">{formatTime(duration)}</span>
        </div>

        {/* Main controls */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {/* Frame controls */}
            <button
              onClick={() => stepFrame(-1)}
              className="px-3 py-1 bg-gray-700 text-white rounded hover:bg-gray-600 text-sm"
              title="Previous frame"
            >
              ⏮
            </button>
            
            <button
              onClick={togglePlay}
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 font-medium"
            >
              {isPlaying ? '⏸ Pause' : '▶ Play'}
            </button>
            
            <button
              onClick={() => stepFrame(1)}
              className="px-3 py-1 bg-gray-700 text-white rounded hover:bg-gray-600 text-sm"
              title="Next frame"
            >
              ⏭
            </button>
          </div>

          {/* Speed controls */}
          <div className="flex items-center gap-1">
            <span className="text-gray-400 text-sm mr-2">Speed:</span>
            {[0.25, 0.5, 1, 2].map((speed) => (
              <button
                key={speed}
                onClick={() => changeSpeed(speed)}
                className={`px-2 py-1 text-sm rounded ${
                  playbackSpeed === speed
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                {speed}x
              </button>
            ))}
          </div>

          {/* Annotation controls */}
          <div className="flex items-center gap-2">
            {hasAnnotated ? (
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setShowAnnotated(!showAnnotated)}
                  className={`px-3 py-1 rounded text-sm ${
                    showAnnotated
                      ? 'bg-green-600 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  {showAnnotated ? '✓ On' : 'Show'}
                </button>
                <button
                  onClick={handleRegenerateAnnotation}
                  disabled={isAnnotating}
                  className="px-2 py-1 bg-yellow-600 text-white rounded text-sm hover:bg-yellow-700 disabled:opacity-50"
                  title="Regenerate annotations"
                >
                  🔄
                </button>
                <button
                  onClick={handleDeleteAnnotation}
                  disabled={isAnnotating}
                  className="px-2 py-1 bg-red-600 text-white rounded text-sm hover:bg-red-700 disabled:opacity-50"
                  title="Delete annotations"
                >
                  🗑️
                </button>
              </div>
            ) : (
              <button
                onClick={handleTriggerAnnotation}
                disabled={isAnnotating}
                className="px-3 py-1 bg-orange-600 text-white rounded text-sm hover:bg-orange-700 disabled:opacity-50"
              >
                {isAnnotating ? `Annotating... ${annotationProgress}%` : 'Annotate'}
              </button>
            )}
          </div>
        </div>

        {/* Frame info */}
        {videoMetadata && (
          <div className="flex justify-between text-gray-400 text-xs">
            <span>Frame: {currentFrame} / {videoMetadata.frame_count}</span>
            <span>{videoMetadata.width}x{videoMetadata.height} @ {videoMetadata.fps?.toFixed(1)} fps</span>
          </div>
        )}
      </div>
    </div>
  )
}

