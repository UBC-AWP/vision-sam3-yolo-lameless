import { useEffect, useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { trainingApi, videosApi } from '@/api/client'

interface TripletTask {
  reference_id: string
  comparison_a_id: string
  comparison_b_id: string
  task_type: 'similarity' | 'dissimilarity'
  pending_tasks: number
  total_tasks: number
}

export default function TripletComparison() {
  const navigate = useNavigate()
  const [task, setTask] = useState<TripletTask | null>(null)
  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [selectedAnswer, setSelectedAnswer] = useState<'A' | 'B' | null>(null)
  const [confidence, setConfidence] = useState<'high' | 'medium' | 'low'>('medium')
  const [stats, setStats] = useState<any>(null)
  
  const refVideoRef = useRef<HTMLVideoElement>(null)
  const compAVideoRef = useRef<HTMLVideoElement>(null)
  const compBVideoRef = useRef<HTMLVideoElement>(null)
  const [isPlaying, setIsPlaying] = useState(false)

  useEffect(() => {
    loadNextTask()
    loadStats()
  }, [])

  const loadNextTask = async () => {
    setLoading(true)
    setSelectedAnswer(null)
    try {
      const data = await trainingApi.getNextTriplet()
      setTask(data)
    } catch (error) {
      console.error('Failed to load triplet task:', error)
    } finally {
      setLoading(false)
    }
  }

  const loadStats = async () => {
    try {
      const statsData = await trainingApi.getTripletStats()
      setStats(statsData)
    } catch (error) {
      console.error('Failed to load stats:', error)
    }
  }

  const handleSubmit = async () => {
    if (!task || selectedAnswer === null) return

    setSubmitting(true)
    try {
      await trainingApi.submitTriplet(
        task.reference_id,
        task.comparison_a_id,
        task.comparison_b_id,
        selectedAnswer,
        confidence,
        task.task_type
      )
      await loadStats()
      await loadNextTask()
    } catch (error) {
      console.error('Failed to submit:', error)
      alert('Failed to submit comparison')
    } finally {
      setSubmitting(false)
    }
  }

  const togglePlayback = () => {
    const videos = [refVideoRef.current, compAVideoRef.current, compBVideoRef.current]
    if (videos.every(v => v)) {
      if (isPlaying) {
        videos.forEach(v => v?.pause())
      } else {
        videos.forEach(v => v?.play())
      }
      setIsPlaying(!isPlaying)
    }
  }

  const restartVideos = () => {
    const videos = [refVideoRef.current, compAVideoRef.current, compBVideoRef.current]
    videos.forEach(v => {
      if (v) {
        v.currentTime = 0
        v.play()
      }
    })
    setIsPlaying(true)
  }

  // Sync video playback
  useEffect(() => {
    const refVideo = refVideoRef.current
    const videos = [compAVideoRef.current, compBVideoRef.current]
    
    if (!refVideo) return

    const syncPlayback = () => {
      videos.forEach(v => {
        if (v && Math.abs(refVideo.currentTime - v.currentTime) > 0.1) {
          v.currentTime = refVideo.currentTime
        }
      })
    }

    refVideo.addEventListener('timeupdate', syncPlayback)
    return () => refVideo.removeEventListener('timeupdate', syncPlayback)
  }, [task])

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return
      
      switch (e.key.toLowerCase()) {
        case 'a':
        case '1':
          setSelectedAnswer('A')
          break
        case 'b':
        case '2':
          setSelectedAnswer('B')
          break
        case ' ':
          e.preventDefault()
          togglePlayback()
          break
        case 'enter':
          if (selectedAnswer) handleSubmit()
          break
      }
    }

    window.addEventListener('keydown', handleKeyPress)
    return () => window.removeEventListener('keydown', handleKeyPress)
  }, [selectedAnswer])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <div className="text-muted-foreground">Loading triplet task...</div>
        </div>
      </div>
    )
  }

  if (!task || task.pending_tasks === 0) {
    return (
      <div className="text-center py-12">
        <div className="text-6xl mb-4">🎯</div>
        <h2 className="text-3xl font-bold mb-4">All Triplet Tasks Complete!</h2>
        <p className="text-muted-foreground mb-8">
          You've completed all triplet comparisons. Great work!
        </p>
        <button
          onClick={() => navigate('/pairwise')}
          className="px-6 py-3 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90"
        >
          Go to Pairwise Comparison
        </button>
      </div>
    )
  }

  const questionText = task.task_type === 'similarity'
    ? 'Which cow walks MORE SIMILARLY to the reference?'
    : 'Which cow walks MORE DIFFERENTLY from the reference?'

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold">Triplet Comparison</h2>
          <p className="text-muted-foreground mt-1">
            {task.task_type === 'similarity' 
              ? 'Select which cow walks most similarly to the reference'
              : 'Select which cow walks most differently from the reference'}
          </p>
        </div>
        {stats && (
          <div className="text-sm text-muted-foreground">
            Progress: {stats.completed_tasks} / {stats.total_tasks} tasks
            ({((stats.completed_tasks / stats.total_tasks) * 100).toFixed(1)}%)
          </div>
        )}
      </div>

      {/* Progress Bar */}
      {stats && (
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className="bg-primary h-2 rounded-full transition-all"
            style={{ width: `${(stats.completed_tasks / stats.total_tasks) * 100}%` }}
          />
        </div>
      )}

      {/* Task Type Badge */}
      <div className="flex justify-center">
        <span className={`px-4 py-2 rounded-full text-sm font-medium ${
          task.task_type === 'similarity'
            ? 'bg-blue-100 text-blue-800'
            : 'bg-purple-100 text-purple-800'
        }`}>
          {task.task_type === 'similarity' ? '🔗 Similarity Task' : '↔️ Dissimilarity Task'}
        </span>
      </div>

      {/* Question */}
      <div className="text-center">
        <h3 className="text-xl font-semibold">{questionText}</h3>
      </div>

      {/* Video Grid */}
      <div className="grid grid-cols-3 gap-4">
        {/* Reference Video (Center/Top) */}
        <div className="col-span-3 md:col-span-1 md:col-start-2">
          <div className="text-center mb-2">
            <span className="inline-block px-3 py-1 bg-yellow-100 text-yellow-800 rounded-full text-sm font-medium">
              Reference Cow
            </span>
          </div>
          <div className="border-4 border-yellow-400 rounded-lg overflow-hidden">
            <video
              ref={refVideoRef}
              src={videosApi.getStreamUrl(task.reference_id)}
              className="w-full aspect-video bg-black"
              loop
              muted
            />
          </div>
        </div>

        {/* Comparison A */}
        <div className="col-span-3 md:col-span-1 md:col-start-1 md:row-start-2">
          <div className="text-center mb-2">
            <span className="text-sm font-medium text-gray-600">Comparison A</span>
          </div>
          <div
            className={`border-4 rounded-lg overflow-hidden cursor-pointer transition-all ${
              selectedAnswer === 'A'
                ? 'border-green-500 ring-4 ring-green-200'
                : 'border-transparent hover:border-gray-300'
            }`}
            onClick={() => setSelectedAnswer('A')}
          >
            <video
              ref={compAVideoRef}
              src={videosApi.getStreamUrl(task.comparison_a_id)}
              className="w-full aspect-video bg-black"
              loop
              muted
            />
          </div>
          <div className="text-center mt-2">
            <button
              onClick={() => setSelectedAnswer('A')}
              className={`px-6 py-2 rounded-lg font-medium transition-all ${
                selectedAnswer === 'A'
                  ? 'bg-green-600 text-white'
                  : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
              }`}
            >
              Select A (Press 1)
            </button>
          </div>
        </div>

        {/* VS Indicator */}
        <div className="col-span-3 md:col-span-1 md:row-start-2 flex items-center justify-center">
          <div className="text-4xl font-bold text-gray-300">VS</div>
        </div>

        {/* Comparison B */}
        <div className="col-span-3 md:col-span-1 md:col-start-3 md:row-start-2">
          <div className="text-center mb-2">
            <span className="text-sm font-medium text-gray-600">Comparison B</span>
          </div>
          <div
            className={`border-4 rounded-lg overflow-hidden cursor-pointer transition-all ${
              selectedAnswer === 'B'
                ? 'border-green-500 ring-4 ring-green-200'
                : 'border-transparent hover:border-gray-300'
            }`}
            onClick={() => setSelectedAnswer('B')}
          >
            <video
              ref={compBVideoRef}
              src={videosApi.getStreamUrl(task.comparison_b_id)}
              className="w-full aspect-video bg-black"
              loop
              muted
            />
          </div>
          <div className="text-center mt-2">
            <button
              onClick={() => setSelectedAnswer('B')}
              className={`px-6 py-2 rounded-lg font-medium transition-all ${
                selectedAnswer === 'B'
                  ? 'bg-green-600 text-white'
                  : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
              }`}
            >
              Select B (Press 2)
            </button>
          </div>
        </div>
      </div>

      {/* Playback Controls */}
      <div className="flex justify-center gap-4">
        <button
          onClick={restartVideos}
          className="px-6 py-2 border rounded-lg hover:bg-accent"
        >
          ↺ Restart All
        </button>
        <button
          onClick={togglePlayback}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          {isPlaying ? '⏸ Pause' : '▶ Play All'}
        </button>
      </div>

      {/* Confidence Selection */}
      {selectedAnswer && (
        <div className="flex justify-center items-center gap-4">
          <span className="text-sm text-muted-foreground">Confidence:</span>
          {(['high', 'medium', 'low'] as const).map((conf) => (
            <button
              key={conf}
              onClick={() => setConfidence(conf)}
              className={`px-4 py-2 rounded-lg text-sm capitalize transition-colors ${
                confidence === conf
                  ? 'bg-primary text-primary-foreground'
                  : 'border hover:bg-accent'
              }`}
            >
              {conf}
            </button>
          ))}
        </div>
      )}

      {/* Submit Button */}
      <div className="flex justify-center">
        <button
          onClick={handleSubmit}
          disabled={selectedAnswer === null || submitting}
          className="px-8 py-3 bg-green-600 text-white rounded-lg font-medium hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {submitting ? 'Submitting...' : 'Submit & Next (Enter)'}
        </button>
      </div>

      {/* Instructions */}
      <div className="bg-gray-50 rounded-lg p-4 text-sm text-muted-foreground">
        <h4 className="font-semibold mb-2">How to Compare</h4>
        <ul className="list-disc list-inside space-y-1">
          <li>Watch the <strong>Reference cow</strong> carefully first</li>
          <li>Then compare both A and B to the reference</li>
          <li>
            {task.task_type === 'similarity'
              ? 'Select which cow walks most SIMILARLY to the reference'
              : 'Select which cow walks most DIFFERENTLY from the reference'}
          </li>
          <li>Consider: gait pattern, speed, posture, and lameness indicators</li>
        </ul>
      </div>

      {/* Keyboard shortcuts */}
      <div className="text-center text-xs text-muted-foreground">
        Shortcuts: <kbd className="px-1 bg-gray-100 rounded">1/A</kbd> select A,{' '}
        <kbd className="px-1 bg-gray-100 rounded">2/B</kbd> select B,{' '}
        <kbd className="px-1 bg-gray-100 rounded">Space</kbd> play/pause,{' '}
        <kbd className="px-1 bg-gray-100 rounded">Enter</kbd> submit
      </div>
    </div>
  )
}

