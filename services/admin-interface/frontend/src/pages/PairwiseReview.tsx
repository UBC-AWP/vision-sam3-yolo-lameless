import { useEffect, useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { videosApi, eloRankingApi, tutorialApi, TutorialExample } from '@/api/client'

interface VideoPair {
  video_id_1: string
  video_id_2: string
  pending_pairs: number
  total_pairs: number
  completed_pairs: number
  status?: string
}

// 7-point comparison scale per DSI specification
const COMPARISON_SCALE = [
  { value: -3, label: 'A Much More Lame', color: 'bg-red-700' },
  { value: -2, label: 'A More Lame', color: 'bg-red-500' },
  { value: -1, label: 'A Slightly More Lame', color: 'bg-red-300' },
  { value: 0, label: 'Equal / Cannot Decide', color: 'bg-gray-400' },
  { value: 1, label: 'B Slightly More Lame', color: 'bg-orange-300' },
  { value: 2, label: 'B More Lame', color: 'bg-orange-500' },
  { value: 3, label: 'B Much More Lame', color: 'bg-orange-700' },
]

export default function PairwiseReview() {
  const navigate = useNavigate()
  const [pair, setPair] = useState<VideoPair | null>(null)
  const [stats, setStats] = useState<any>(null)
  const [ranking, setRanking] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [selectedValue, setSelectedValue] = useState<number | null>(null)
  const [showRanking, setShowRanking] = useState(false)

  // Tutorial state
  const [inTutorial, setInTutorial] = useState(true)
  const [tutorialStep, setTutorialStep] = useState(0)
  const [showTutorialFeedback, setShowTutorialFeedback] = useState(false)
  const [tutorialScore, setTutorialScore] = useState(0)
  const [tutorialExamples, setTutorialExamples] = useState<TutorialExample[]>([])
  const [tutorialLoading, setTutorialLoading] = useState(true)
  
  // Share functionality
  const [showShareModal, setShowShareModal] = useState(false)
  const [shareUrl, setShareUrl] = useState('')
  
  const video1Ref = useRef<HTMLVideoElement>(null)
  const video2Ref = useRef<HTMLVideoElement>(null)
  const [isPlaying, setIsPlaying] = useState(false)

  useEffect(() => {
    // Check if user has completed tutorial
    const tutorialComplete = localStorage.getItem('pairwise_tutorial_complete')
    if (tutorialComplete === 'true') {
      setInTutorial(false)
      setTutorialLoading(false)
      loadNextPair()
    } else {
      // Load tutorial examples from API
      loadTutorialExamples()
    }
    loadStats()
  }, [])

  const loadTutorialExamples = async () => {
    setTutorialLoading(true)
    try {
      const data = await tutorialApi.getExamples()
      if (data.examples && data.examples.length > 0) {
        setTutorialExamples(data.examples)
      } else {
        // No tutorials configured, skip to real comparisons
        console.log('No tutorial examples found, skipping tutorial')
        setInTutorial(false)
        loadNextPair()
      }
    } catch (error) {
      console.error('Failed to load tutorial examples:', error)
      // On error, skip tutorial and go to real comparisons
      setInTutorial(false)
      loadNextPair()
    } finally {
      setTutorialLoading(false)
    }
  }

  const loadNextPair = async () => {
    setLoading(true)
    setSelectedValue(null)
    setIsPlaying(false)
    try {
      // Use Elo API for intelligent pair selection
      const data = await eloRankingApi.getNextPair()
      setPair(data)
    } catch (error) {
      console.error('Failed to load pair:', error)
    } finally {
      setLoading(false)
    }
  }

  const loadStats = async () => {
    try {
      // Use Elo API for stats and hierarchy
      const [statsData, hierarchyData] = await Promise.all([
        eloRankingApi.getStats(),
        eloRankingApi.getHierarchy()
      ])
      setStats(statsData)
      setRanking(hierarchyData)
    } catch (error) {
      console.error('Failed to load stats:', error)
    }
  }

  const handleTutorialAnswer = () => {
    const currentExample = tutorialExamples[tutorialStep]
    if (!currentExample) return

    const isCorrect = selectedValue === currentExample.correct_answer

    if (isCorrect) {
      setTutorialScore(prev => prev + 1)
    }

    setShowTutorialFeedback(true)
  }

  const handleTutorialNext = () => {
    setShowTutorialFeedback(false)
    setSelectedValue(null)

    if (tutorialStep < tutorialExamples.length - 1) {
      setTutorialStep(prev => prev + 1)
    } else {
      // Tutorial complete
      localStorage.setItem('pairwise_tutorial_complete', 'true')
      setInTutorial(false)
      loadNextPair()
    }
  }

  const handleSubmit = async () => {
    if (!pair || selectedValue === null) return

    setSubmitting(true)
    try {
      // Convert 7-point scale to winner format and degree for Elo API
      let winner: number
      let degree: number
      let confidence: string

      if (selectedValue < 0) {
        winner = 1 // Video A is more lame (wins in lameness hierarchy)
        degree = Math.abs(selectedValue) // 1, 2, or 3
        confidence = Math.abs(selectedValue) === 3 ? 'very_confident' :
                     Math.abs(selectedValue) === 2 ? 'confident' : 'uncertain'
      } else if (selectedValue > 0) {
        winner = 2 // Video B is more lame
        degree = selectedValue // 1, 2, or 3
        confidence = selectedValue === 3 ? 'very_confident' :
                     selectedValue === 2 ? 'confident' : 'uncertain'
      } else {
        winner = 0 // Tie - cannot decide
        degree = 0
        confidence = 'uncertain'
      }

      // Submit to Elo ranking system with degree of preference
      await eloRankingApi.submitComparison(
        pair.video_id_1,
        pair.video_id_2,
        winner,
        degree,
        confidence,
        selectedValue // Raw score for additional analysis
      )
      await loadStats()
      await loadNextPair()
    } catch (error) {
      console.error('Failed to submit:', error)
      alert('Failed to submit comparison')
    } finally {
      setSubmitting(false)
    }
  }

  const togglePlayback = () => {
    if (video1Ref.current && video2Ref.current) {
      if (isPlaying) {
        video1Ref.current.pause()
        video2Ref.current.pause()
      } else {
        video1Ref.current.play()
        video2Ref.current.play()
      }
      setIsPlaying(!isPlaying)
    }
  }

  const restartVideos = () => {
    if (video1Ref.current && video2Ref.current) {
      video1Ref.current.currentTime = 0
      video2Ref.current.currentTime = 0
      video1Ref.current.play()
      video2Ref.current.play()
      setIsPlaying(true)
    }
  }

  // Sync video playback
  useEffect(() => {
    const video1 = video1Ref.current
    const video2 = video2Ref.current
    
    if (!video1 || !video2) return

    const syncPlayback = () => {
      if (Math.abs(video1.currentTime - video2.currentTime) > 0.1) {
        video2.currentTime = video1.currentTime
      }
    }

    video1.addEventListener('timeupdate', syncPlayback)
    return () => video1.removeEventListener('timeupdate', syncPlayback)
  }, [pair])

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return
      
      switch (e.key) {
        case '1': setSelectedValue(-3); break
        case '2': setSelectedValue(-2); break
        case '3': setSelectedValue(-1); break
        case '4': setSelectedValue(0); break
        case '5': setSelectedValue(1); break
        case '6': setSelectedValue(2); break
        case '7': setSelectedValue(3); break
        case ' ':
          e.preventDefault()
          togglePlayback()
          break
        case 'Enter':
          if (selectedValue !== null) {
            if (inTutorial) {
              if (showTutorialFeedback) {
                handleTutorialNext()
              } else {
                handleTutorialAnswer()
              }
            } else {
              handleSubmit()
            }
          }
          break
      }
    }

    window.addEventListener('keydown', handleKeyPress)
    return () => window.removeEventListener('keydown', handleKeyPress)
  }, [selectedValue, inTutorial, showTutorialFeedback])

  const generateShareUrl = () => {
    if (!pair) return
    const url = `${window.location.origin}/compare/${pair.video_id_1}/${pair.video_id_2}`
    setShareUrl(url)
    setShowShareModal(true)
  }

  // Tutorial UI
  if (inTutorial) {
    // Show loading state while fetching tutorial examples
    if (tutorialLoading) {
      return (
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
            <div className="text-muted-foreground">Loading tutorial...</div>
          </div>
        </div>
      )
    }

    // If no tutorial examples, this shouldn't render (handled in useEffect)
    if (tutorialExamples.length === 0) {
      return null
    }

    const currentExample = tutorialExamples[tutorialStep]

    return (
      <div className="space-y-6 max-w-4xl mx-auto">
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
          <h2 className="text-2xl font-bold text-blue-800 mb-2">
            Tutorial: Learn to Assess Lameness
          </h2>
          <p className="text-blue-700">
            Step {tutorialStep + 1} of {tutorialExamples.length}
          </p>
          <div className="mt-4">
            <div className="w-full bg-blue-200 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full transition-all"
                style={{ width: `${((tutorialStep + 1) / tutorialExamples.length) * 100}%` }}
              />
            </div>
          </div>
        </div>

        <div className="border rounded-lg p-6 bg-white">
          <h3 className="text-lg font-semibold mb-2">{currentExample.description}</h3>

          {/* Tutorial videos - actual videos from API */}
          <div className="grid grid-cols-2 gap-4 my-6">
            <div className="space-y-2">
              <div className="text-center font-medium text-gray-700">Video A</div>
              <div className="aspect-video bg-black rounded-lg overflow-hidden">
                <video
                  src={videosApi.getStreamUrl(currentExample.video_id_1)}
                  className="w-full h-full object-contain"
                  controls
                  loop
                  muted
                  autoPlay
                />
              </div>
            </div>
            <div className="space-y-2">
              <div className="text-center font-medium text-gray-700">Video B</div>
              <div className="aspect-video bg-black rounded-lg overflow-hidden">
                <video
                  src={videosApi.getStreamUrl(currentExample.video_id_2)}
                  className="w-full h-full object-contain"
                  controls
                  loop
                  muted
                  autoPlay
                />
              </div>
            </div>
          </div>

          {/* 7-Point Scale */}
          <div className="space-y-3">
            <label className="block text-sm font-medium text-gray-700">
              Select your comparison (1-7 keys work too):
            </label>
            <div className="flex gap-2 flex-wrap justify-center">
              {COMPARISON_SCALE.map((option, idx) => (
                <button
                  key={option.value}
                  onClick={() => setSelectedValue(option.value)}
                  className={`px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                    selectedValue === option.value
                      ? `${option.color} text-white ring-2 ring-offset-2 ring-blue-500`
                      : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
                  }`}
                >
                  <span className="text-xs opacity-60">{idx + 1}</span> {option.label}
                </button>
              ))}
            </div>
          </div>

          {showTutorialFeedback && (
            <div className={`mt-6 p-4 rounded-lg ${
              selectedValue === currentExample.correct_answer
                ? 'bg-green-100 border border-green-300'
                : 'bg-yellow-100 border border-yellow-300'
            }`}>
              <h4 className={`font-semibold ${
                selectedValue === currentExample.correct_answer ? 'text-green-800' : 'text-yellow-800'
              }`}>
                {selectedValue === currentExample.correct_answer ? '✓ Correct!' : '○ Not quite right'}
              </h4>
              <p className="text-sm mt-1">{currentExample.hint}</p>
            </div>
          )}

          <div className="mt-6 flex justify-center gap-4">
            {!showTutorialFeedback ? (
              <button
                onClick={handleTutorialAnswer}
                disabled={selectedValue === null}
                className="px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50"
              >
                Check Answer
              </button>
            ) : (
              <button
                onClick={handleTutorialNext}
                className="px-6 py-3 bg-green-600 text-white rounded-lg font-medium hover:bg-green-700"
              >
                {tutorialStep < tutorialExamples.length - 1 ? 'Next Example' : 'Start Real Comparisons'}
              </button>
            )}
          </div>
        </div>

        <div className="flex justify-between items-center text-sm text-gray-500">
          <div>Score: {tutorialScore}/{tutorialStep + (showTutorialFeedback ? 1 : 0)}</div>
          <button
            onClick={() => {
              localStorage.setItem('pairwise_tutorial_complete', 'true')
              setInTutorial(false)
              loadNextPair()
            }}
            className="text-blue-600 hover:text-blue-800 underline"
          >
            Skip Tutorial
          </button>
        </div>
      </div>
    )
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <div className="text-muted-foreground">Loading video pair...</div>
        </div>
      </div>
    )
  }

  if (pair?.status === 'all_completed') {
    return (
      <div className="text-center py-12">
        <div className="text-6xl mb-4">🎉</div>
        <h2 className="text-3xl font-bold mb-4">All Comparisons Complete!</h2>
        <p className="text-muted-foreground mb-8">
          You've completed all {pair.total_pairs} pairwise comparisons.
        </p>
        <button
          onClick={() => setShowRanking(true)}
          className="px-6 py-3 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90"
        >
          View Lameness Ranking
        </button>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold">Pairwise Comparison</h2>
          <p className="text-muted-foreground mt-1">
            Compare videos using a 7-point scale to build a lameness hierarchy
          </p>
        </div>
        <div className="flex items-center gap-4">
          {stats && (
            <div className="text-sm text-muted-foreground">
              Progress: {stats.unique_pairs_compared} / {stats.total_possible_pairs} pairs
              ({(stats.completion_rate * 100).toFixed(1)}%)
            </div>
          )}
          <button
            onClick={() => setShowRanking(!showRanking)}
            className="px-4 py-2 border rounded-lg hover:bg-accent"
          >
            {showRanking ? 'Hide Ranking' : 'Show Ranking'}
          </button>
          <button
            onClick={generateShareUrl}
            className="px-4 py-2 border rounded-lg hover:bg-accent"
          >
            Share
          </button>
          <button
            onClick={() => {
              localStorage.removeItem('pairwise_tutorial_complete')
              setInTutorial(true)
              setTutorialStep(0)
              setTutorialScore(0)
            }}
            className="px-4 py-2 text-sm text-muted-foreground hover:text-foreground"
          >
            Retake Tutorial
          </button>
        </div>
      </div>

      {/* Progress Bar */}
      {stats && (
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className="bg-primary h-2 rounded-full transition-all"
            style={{ width: `${stats.completion_rate * 100}%` }}
          />
        </div>
      )}

      {/* Ranking Panel */}
      {showRanking && ranking && (
        <div className="border rounded-lg p-6">
          <div className="flex justify-between items-start mb-4">
            <div>
              <h3 className="text-lg font-semibold">Lameness Hierarchy (EloSteepness)</h3>
              <p className="text-sm text-muted-foreground">
                Higher Elo = More Lame. Based on {ranking.total_comparisons} comparisons.
              </p>
            </div>
            {ranking.metrics && (
              <div className="text-right text-sm">
                <div className="font-medium">
                  Steepness: {(ranking.metrics.steepness * 100).toFixed(1)}%
                </div>
                <div className="text-muted-foreground">
                  {ranking.metrics.hierarchy_linearity} hierarchy
                </div>
              </div>
            )}
          </div>
          <div className="grid gap-2 max-h-64 overflow-y-auto">
            {ranking.ranking?.map((item: any) => (
              <div
                key={item.video_id}
                className="flex items-center justify-between p-2 bg-gray-50 rounded"
              >
                <div className="flex items-center gap-3">
                  <span className="w-8 h-8 flex items-center justify-center bg-primary text-primary-foreground rounded-full text-sm font-bold">
                    {item.rank}
                  </span>
                  <span className="font-mono text-sm">{item.video_id.slice(0, 8)}...</span>
                  <span className="text-xs text-muted-foreground">
                    W:{item.wins} L:{item.losses}
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  <div className="text-xs text-muted-foreground">
                    ±{item.elo_uncertainty?.toFixed(0) || '?'}
                  </div>
                  <div className={`font-medium ${
                    item.elo_rating > 1550 ? 'text-red-600' :
                    item.elo_rating < 1450 ? 'text-green-600' : 'text-gray-600'
                  }`}>
                    {item.elo_rating?.toFixed(0) || 1500} Elo
                  </div>
                </div>
              </div>
            ))}
          </div>
          {ranking.ranking?.length === 0 && (
            <p className="text-center text-muted-foreground py-4">
              No comparisons yet. Start comparing videos to build the hierarchy.
            </p>
          )}
        </div>
      )}

      {/* Main Comparison Area */}
      {pair && (
        <>
          {/* Videos Side by Side */}
          <div className="grid grid-cols-2 gap-6">
            {/* Video A */}
            <div className="space-y-2">
              <div className="text-center font-semibold text-lg">Video A</div>
              <div className={`border-4 rounded-lg overflow-hidden transition-colors ${
                selectedValue !== null && selectedValue < 0 ? 'border-red-500' : 'border-transparent'
              }`}>
                <video
                  ref={video1Ref}
                  src={videosApi.getStreamUrl(pair.video_id_1)}
                  className="w-full aspect-video bg-black"
                  loop
                  muted
                />
              </div>
            </div>

            {/* Video B */}
            <div className="space-y-2">
              <div className="text-center font-semibold text-lg">Video B</div>
              <div className={`border-4 rounded-lg overflow-hidden transition-colors ${
                selectedValue !== null && selectedValue > 0 ? 'border-orange-500' : 'border-transparent'
              }`}>
                <video
                  ref={video2Ref}
                  src={videosApi.getStreamUrl(pair.video_id_2)}
                  className="w-full aspect-video bg-black"
                  loop
                  muted
                />
              </div>
            </div>
          </div>

          {/* Playback Controls */}
          <div className="flex justify-center gap-4">
            <button
              onClick={restartVideos}
              className="px-6 py-2 border rounded-lg hover:bg-accent"
            >
              ↺ Restart
            </button>
            <button
              onClick={togglePlayback}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              {isPlaying ? '⏸ Pause' : '▶ Play'}
            </button>
          </div>

          {/* 7-Point Comparison Scale */}
          <div className="space-y-4">
            <label className="block text-center font-medium">
              Which cow appears more lame? (Keys 1-7)
            </label>
            <div className="flex gap-2 flex-wrap justify-center">
              {COMPARISON_SCALE.map((option, idx) => (
                <button
                  key={option.value}
                  onClick={() => setSelectedValue(option.value)}
                  className={`px-4 py-3 rounded-lg text-sm font-medium transition-all flex-1 min-w-[120px] max-w-[160px] ${
                    selectedValue === option.value
                      ? `${option.color} text-white ring-2 ring-offset-2 ring-blue-500 scale-105`
                      : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
                  }`}
                >
                  <div className="text-xs opacity-60 mb-1">Press {idx + 1}</div>
                  {option.label}
                </button>
              ))}
            </div>
          </div>

          {/* Submit Button */}
          <div className="flex justify-center">
            <button
              onClick={handleSubmit}
              disabled={selectedValue === null || submitting}
              className="px-8 py-3 bg-green-600 text-white rounded-lg font-medium hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {submitting ? 'Submitting...' : 'Submit & Next Pair (Enter)'}
            </button>
          </div>

          {/* Lameness Indicators Guide */}
          <div className="bg-gray-50 rounded-lg p-4 text-sm">
            <h4 className="font-semibold mb-2">What to Look For:</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="flex items-start gap-2">
                <span className="text-red-500">●</span>
                <div>
                  <div className="font-medium">Arched Back</div>
                  <div className="text-muted-foreground">Hunched posture while walking</div>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-orange-500">●</span>
                <div>
                  <div className="font-medium">Head Bobbing</div>
                  <div className="text-muted-foreground">Up/down head movement</div>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-yellow-500">●</span>
                <div>
                  <div className="font-medium">Uneven Stride</div>
                  <div className="text-muted-foreground">Favoring one leg</div>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-blue-500">●</span>
                <div>
                  <div className="font-medium">Slow Movement</div>
                  <div className="text-muted-foreground">Hesitant or cautious gait</div>
                </div>
              </div>
            </div>
          </div>

          {/* Keyboard shortcuts */}
          <div className="text-center text-xs text-muted-foreground">
            Shortcuts: <kbd className="px-1 bg-gray-100 rounded">1-7</kbd> select scale,{' '}
            <kbd className="px-1 bg-gray-100 rounded">Space</kbd> play/pause,{' '}
            <kbd className="px-1 bg-gray-100 rounded">Enter</kbd> submit
          </div>
        </>
      )}

      {/* Share Modal */}
      {showShareModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
            <h3 className="text-lg font-semibold mb-4">Share Comparison</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1">Share URL</label>
                <input
                  type="text"
                  value={shareUrl}
                  readOnly
                  className="w-full p-2 border rounded-lg text-sm"
                  onClick={(e) => (e.target as HTMLInputElement).select()}
                />
              </div>
              <div className="flex justify-center">
                <div className="p-4 bg-gray-100 rounded-lg">
                  {/* QR Code placeholder - would use a QR library in production */}
                  <div className="w-32 h-32 bg-white border-2 border-dashed border-gray-300 flex items-center justify-center text-xs text-gray-500">
                    QR Code
                  </div>
                </div>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => {
                    navigator.clipboard.writeText(shareUrl)
                    alert('URL copied!')
                  }}
                  className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg"
                >
                  Copy URL
                </button>
                <button
                  onClick={() => setShowShareModal(false)}
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
