import { useEffect, useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { videosApi, eloRankingApi, tutorialApi, TutorialExample } from '@/api/client'
import { useLanguage } from '@/contexts/LanguageContext'
import { useAuth } from '@/contexts/AuthContext'
import { cn } from '@/lib/utils'
import { getRandomDemoPair, parseDemoCSV, resetDemoCache, type DemoPair } from '@/utils/demoData'
import {
  getValidDemoPairs,
  type DemoConsensusPair,
} from '@/utils/pairwiseConsensus'

// ── Task definitions ─────────────────────────────────────────────────────────
interface PairwiseTask {
  id: number
  name: string
  description: string
  pairCount: number
  completedPeople: number
  requiredPeople: number
  color: string
  bgColor: string
  textColor: string
  borderColor: string
}

const PAIRWISE_TASKS: PairwiseTask[] = [
  {
    id: 1,
    name: 'UBC Farm 1',
    description: 'Morning session — Spring 2021',
    pairCount: 10,
    completedPeople: 13,
    requiredPeople: 20,
    color: 'from-blue-500 to-blue-600',
    bgColor: 'bg-blue-500/10',
    textColor: 'text-blue-500',
    borderColor: 'border-blue-500/30',
  },
  {
    id: 2,
    name: 'UBC Farm 2',
    description: 'Afternoon session — Spring 2021',
    pairCount: 10,
    completedPeople: 8,
    requiredPeople: 20,
    color: 'from-violet-500 to-violet-600',
    bgColor: 'bg-violet-500/10',
    textColor: 'text-violet-500',
    borderColor: 'border-violet-500/30',
  },
  {
    id: 3,
    name: 'UBC Farm 3',
    description: 'Evening rounds — Summer 2021',
    pairCount: 10,
    completedPeople: 5,
    requiredPeople: 20,
    color: 'from-amber-500 to-amber-600',
    bgColor: 'bg-amber-500/10',
    textColor: 'text-amber-500',
    borderColor: 'border-amber-500/30',
  },
  {
    id: 4,
    name: 'UBC Farm 4',
    description: 'Expert validation — Fall 2021',
    pairCount: 10,
    completedPeople: 2,
    requiredPeople: 20,
    color: 'from-rose-500 to-rose-600',
    bgColor: 'bg-rose-500/10',
    textColor: 'text-rose-500',
    borderColor: 'border-rose-500/30',
  },
]

// Evenly spread 4 tasks across all 435 pairs (435 / 4 ≈ 108 apart)
// Task k starts at k * floor(totalPairs / numTasks)
const TASK_OFFSETS = [0, 108, 217, 326]

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
  const { t } = useLanguage()
  const { user } = useAuth()
  const useDemo = user?.id === 'guest' || user?.role === 'rater'

  // Task selection (for rater/demo users)
  const [selectedTask, setSelectedTask] = useState<PairwiseTask | null>(null)

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
  
  // Demo mode
  const [demoMode, setDemoMode] = useState(false)
  const [demoPair, setDemoPair] = useState<DemoConsensusPair | null>(null)
  const [demoIndex, setDemoIndex] = useState(0)
  const [demoPairs, setDemoPairs] = useState<DemoConsensusPair[]>([])
  const [showDemoComplete, setShowDemoComplete] = useState(false)

  useEffect(() => {
    if (useDemo) {
      // Rater / demo users skip tutorial & live API calls — show task selection
      setInTutorial(false)
      setTutorialLoading(false)
      setLoading(false)
      return
    }
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
  }, [useDemo])

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
    setIsPlaying(demoMode) // Auto-playing in demo mode
    
    if (demoMode) {
      // Load next demo pair (max 3 pairs)
      const nextIndex = demoIndex + 1
      if (nextIndex < demoPairs.length) {
        setDemoPair(demoPairs[nextIndex])
        setDemoIndex(nextIndex)
      }
      setLoading(false)
      return
    }
    
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
  
  const enableDemoMode = () => {
    setDemoMode(true)
    setInTutorial(false)
    setTutorialLoading(false)
    localStorage.setItem('pairwise_tutorial_complete', 'true')

    // Use local video pairs + CSV consensus data
    const allPairs = getValidDemoPairs()

    // Shuffle for variety; fall back to legacy S3 pairs if no local videos found
    if (allPairs.length === 0) {
      resetDemoCache()
      const legacyAll = parseDemoCSV()
      // Legacy pairs don't have consensusData; wrap them minimally
      // (feedback won't fire for these but UI still works)
      setDemoPairs([])
      setDemoPair(null)
      setDemoIndex(0)
      setShowDemoComplete(false)
      setIsPlaying(true)
      setLoading(false)
      return
    }

    const shuffled = [...allPairs].sort(() => Math.random() - 0.5)

    setDemoPairs(shuffled)
    setDemoPair(shuffled[0])
    setDemoIndex(0)
    setShowDemoComplete(false)
    setIsPlaying(true)
    setLoading(false)
  }

  const handleTaskSelect = (task: PairwiseTask) => {
    const allPairs = getValidDemoPairs()
    if (allPairs.length === 0) return

    const offset = TASK_OFFSETS[task.id - 1]
    const taskPairs: DemoConsensusPair[] = []
    for (let i = 0; i < task.pairCount; i++) {
      const pair = allPairs[(offset + i) % allPairs.length]
      // Randomly assign which cow appears on the left (matching the original study)
      const flipped = Math.random() < 0.5
      taskPairs.push(flipped ? {
        ...pair,
        cow_L: pair.cow_R,
        cow_R: pair.cow_L,
        cow_L_URL: pair.cow_R_URL,
        cow_R_URL: pair.cow_L_URL,
      } : pair)
    }

    setSelectedTask(task)
    setDemoMode(true)
    setInTutorial(false)
    setTutorialLoading(false)
    localStorage.setItem('pairwise_tutorial_complete', 'true')
    setDemoPairs(taskPairs)
    setDemoPair(taskPairs[0])
    setDemoIndex(0)
    setShowDemoComplete(false)
    setIsPlaying(true)
    setLoading(false)
  }

  const handleBackToTasks = () => {
    setSelectedTask(null)
    setDemoMode(false)
    setDemoPairs([])
    setDemoPair(null)
    setDemoIndex(0)
    setShowDemoComplete(false)
    setIsPlaying(false)
    setLoading(false)
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
    if ((!pair && !demoMode) || selectedValue === null) return

    setSubmitting(true)
    
    if (demoMode && demoPair) {
      const isLastPair = demoIndex === demoPairs.length - 1
      if (isLastPair) {
        setShowDemoComplete(true)
      } else {
        const nextIndex = demoIndex + 1
        setDemoPair(demoPairs[nextIndex])
        setDemoIndex(nextIndex)
        setSelectedValue(null)
        setIsPlaying(true)
      }
      setSubmitting(false)
      return
    }
    
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
        pair!.video_id_1,
        pair!.video_id_2,
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
  }, [pair, demoPair])

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

  // ── Task Selection Screen (rater / demo users) ───────────────────────────
  if (useDemo && !selectedTask && !demoMode) {
    const completedTaskIds: number[] = JSON.parse(
      localStorage.getItem('pairwise_completed_tasks') ?? '[]'
    )
    return (
      <div className="max-w-4xl mx-auto space-y-8 animate-fade-in">
        {/* Header */}
        <div className="text-center">
          <h1 className="text-3xl font-bold mb-2">{t('pairwise.tasksTitle')}</h1>
          <p className="text-muted-foreground whitespace-nowrap">
            {t('pairwise.tasksSubtitle')}
          </p>
          <div className="mt-4 flex items-center justify-center gap-2 flex-wrap">
            <span className="text-sm text-muted-foreground">{t('pairwise.notSure')}</span>
            <button
              onClick={() => navigate('/pairwise-tutorial')}
              className="px-3 py-1.5 rounded-lg border border-primary text-primary text-sm font-medium hover:bg-primary/10 transition-colors"
            >
              {t('pairwise.clickTutorial')}
            </button>
          </div>
        </div>

        {/* Task Cards */}
        <div className="grid sm:grid-cols-2 gap-5">
          {PAIRWISE_TASKS.map((task, i) => {
            const isCompleted = completedTaskIds.includes(task.id)
            const progressPct = Math.round((task.completedPeople / task.requiredPeople) * 100)
            return (
              <div
                key={task.id}
                className={cn(
                  "relative rounded-2xl border bg-card p-6 flex flex-col gap-4 cursor-pointer",
                  "transition-all duration-200 hover:shadow-xl hover:-translate-y-0.5 hover:border-primary/40",
                  "animate-slide-in-up",
                  isCompleted ? "border-emerald-500/40" : "border-border"
                )}
                style={{ animationDelay: `${i * 0.08}s`, animationFillMode: 'backwards' }}
                onClick={() => handleTaskSelect(task)}
              >
                {/* Completed badge */}
                {isCompleted && (
                  <div className="absolute top-4 right-4 flex items-center gap-1 px-2 py-0.5 rounded-full bg-emerald-500/15 text-emerald-500 text-xs font-medium">
                    ✓ {t('pairwise.completed')}
                  </div>
                )}

                {/* Title row */}
                <div className="flex items-start gap-4">
                  <div className={cn(
                    "w-12 h-12 rounded-xl flex items-center justify-center shrink-0 bg-gradient-to-br",
                    task.color
                  )}>
                    <span className="text-xl text-white font-bold">{task.id}</span>
                  </div>
                  <div>
                    <h3 className="text-lg font-bold">{task.name}</h3>
                    <p className="text-sm text-muted-foreground">{task.description}</p>
                  </div>
                </div>

                {/* Stats row */}
                <div className="flex items-center gap-3">
                  <span className={cn(
                    "px-2.5 py-1 rounded-lg text-xs font-semibold",
                    task.bgColor, task.textColor
                  )}>
                    {task.pairCount} {t('pairwise.pairs')}
                  </span>
                  <span className="text-xs text-muted-foreground">·</span>
                  <span className="text-xs text-muted-foreground">
                    {task.completedPeople}/{task.requiredPeople} {t('pairwise.ratersCompleted')}
                  </span>
                </div>

                {/* Progress bar */}
                <div>
                  <div className="flex justify-between text-xs text-muted-foreground mb-1.5">
                    <span>{t('pairwise.completion')}</span>
                    <span className="font-medium text-foreground">{progressPct}%</span>
                  </div>
                  <div className="h-2 rounded-full bg-muted overflow-hidden">
                    <div
                      className={cn("h-full rounded-full bg-gradient-to-r transition-all duration-700", task.color)}
                      style={{ width: `${progressPct}%` }}
                    />
                  </div>
                  <p className="text-xs text-muted-foreground mt-1.5">
                    {task.completedPeople}/{task.requiredPeople} people have completed this task
                  </p>
                </div>

                {/* CTA */}
                <button
                  className={cn(
                    "mt-auto w-full py-2.5 rounded-xl font-medium text-sm transition-colors",
                    isCompleted
                      ? "bg-emerald-500/15 text-emerald-500 hover:bg-emerald-500/25"
                      : "bg-primary text-primary-foreground hover:bg-primary/90"
                  )}
                  onClick={(e) => { e.stopPropagation(); handleTaskSelect(task) }}
                >
                  {isCompleted ? t('pairwise.redoTask') : t('pairwise.startTaskArrow')}
                </button>
              </div>
            )
          })}
        </div>
      </div>
    )
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
        <div className="bg-primary/10 border border-primary/30 rounded-lg p-6">
          <h2 className="text-2xl font-bold text-primary mb-2">
            Tutorial: Learn to Assess Lameness
          </h2>
          <p className="text-primary/80">
            Step {tutorialStep + 1} of {tutorialExamples.length}
          </p>
          <div className="mt-4">
            <div className="w-full bg-primary/20 rounded-full h-2">
              <div
                className="bg-primary h-2 rounded-full transition-all"
                style={{ width: `${((tutorialStep + 1) / tutorialExamples.length) * 100}%` }}
              />
            </div>
          </div>
        </div>

        <div className="border border-border rounded-lg p-6 bg-card">
          <h3 className="text-lg font-semibold mb-2 text-foreground">{currentExample.description}</h3>

          {/* Tutorial videos - actual videos from API */}
          <div className="grid grid-cols-2 gap-4 my-6">
            <div className="space-y-2">
              <div className="text-center font-medium text-muted-foreground">Left Cow</div>
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
              <div className="text-center font-medium text-muted-foreground">Right Cow</div>
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

          {/* 7-Point Scale - Circular Buttons */}
          <div className="space-y-2">
            <label className="block text-center font-medium text-muted-foreground text-sm">
              Select the option that best describes the lameness difference
            </label>
            <div className="flex items-center justify-center gap-8 py-3">
              {COMPARISON_SCALE.map((option, idx) => {
                // Smaller sizes
                const sizeClass = 
                  idx === 0 || idx === 6 ? 'w-12 h-12' :  // Large outer
                  idx === 1 || idx === 5 ? 'w-10 h-10' :  // Medium-large
                  idx === 2 || idx === 4 ? 'w-9 h-9' :    // Medium
                  'w-7 h-7';                               // Small (center)
                
                // Softer, less bright colors
                const colorClass = 
                  selectedValue === option.value
                    ? option.value < 0 ? 'bg-blue-600 border-blue-700' :
                      option.value > 0 ? 'bg-orange-600 border-orange-700' :
                      'bg-gray-500 border-gray-600'
                    : option.value < 0 ? 'bg-blue-500/50 hover:bg-blue-500/70 border-blue-500/60' :
                      option.value > 0 ? 'bg-orange-500/50 hover:bg-orange-500/70 border-orange-500/60' :
                      'bg-gray-400/40 hover:bg-gray-400/60 border-gray-400/50';

                // Text label for outer buttons
                const showText = idx === 0 || idx === 6;
                const labelText = idx === 0 ? 'Left more lame' : idx === 6 ? 'Right more lame' : '';

                return (
                  <div key={option.value} className="flex flex-col items-center gap-1 flex-shrink-0">
                    <span className={`text-[10px] mb-0.5 h-3 ${showText ? 'text-muted-foreground' : 'invisible'}`}>
                      {showText ? labelText : 'placeholder'}
                    </span>
                    <button
                      onClick={() => setSelectedValue(option.value)}
                      className={`rounded-full ${sizeClass} ${colorClass} border-2 transition-all flex-shrink-0 ${
                        selectedValue === option.value ? 'ring-2 ring-offset-2 ring-primary scale-110' : ''
                      }`}
                      title={option.label}
                      style={{ aspectRatio: '1 / 1' }}
                    >
                      <span className="sr-only">{option.label}</span>
                    </button>
                  </div>
                )
              })}
            </div>
          </div>

          {showTutorialFeedback && (
            <div className={`mt-6 p-4 rounded-lg ${
              selectedValue === currentExample.correct_answer
                ? 'bg-success/10 border border-success/30'
                : 'bg-warning/10 border border-warning/30'
            }`}>
              <h4 className={`font-semibold ${
                selectedValue === currentExample.correct_answer ? 'text-success' : 'text-warning'
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
                className="px-6 py-3 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90 disabled:opacity-50"
              >
                Check Answer
              </button>
            ) : (
              <button
                onClick={handleTutorialNext}
                className="px-6 py-3 bg-success text-white rounded-lg font-medium hover:bg-success/90"
              >
                {tutorialStep < tutorialExamples.length - 1 ? 'Next Example' : 'Start Real Comparisons'}
              </button>
            )}
          </div>
        </div>

        <div className="flex justify-between items-center text-sm text-muted-foreground">
          <div>{t('tutorial.score')} {tutorialScore}/{tutorialStep + (showTutorialFeedback ? 1 : 0)}</div>
          <button
            onClick={() => {
              localStorage.setItem('pairwise_tutorial_complete', 'true')
              setInTutorial(false)
              loadNextPair()
            }}
            className="text-primary hover:text-primary/80 underline"
          >
            {t('pairwise.skipTutorial')}
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
          <div className="text-muted-foreground">{t('pairwise.loading')}</div>
        </div>
      </div>
    )
  }

  if (showDemoComplete) {
    // Mark this task as completed in localStorage
    if (selectedTask) {
      const completed: number[] = JSON.parse(
        localStorage.getItem('pairwise_completed_tasks') ?? '[]'
      )
      if (!completed.includes(selectedTask.id)) {
        localStorage.setItem('pairwise_completed_tasks', JSON.stringify([...completed, selectedTask.id]))
      }
    }

    return (
      <div className="text-center py-12 relative">
        {/* Confetti effect */}
        <div className="absolute inset-0 pointer-events-none overflow-hidden">
          {[...Array(50)].map((_, i) => (
            <div
              key={i}
              className="absolute animate-confetti"
              style={{
                left: `${Math.random() * 100}%`,
                top: `-20px`,
                animationDelay: `${Math.random() * 2}s`,
                animationDuration: `${2 + Math.random() * 2}s`
              }}
            >
              {['🎉', '🎊', '✨', '⭐', '🌟'][Math.floor(Math.random() * 5)]}
            </div>
          ))}
        </div>
        
        <div className="relative z-10">
          <div className="text-6xl mb-4 animate-bounce">🎉</div>
          <h2 className="text-3xl font-bold mb-4">{t('pairwise.taskComplete')}</h2>
          <p className="text-muted-foreground mb-2">
            {t('pairwise.finishedAll')} {selectedTask?.pairCount ?? demoPairs.length} {t('pairwise.comparisons')}
            {selectedTask ? ` ${t('pairwise.comparisonsFor')} ${selectedTask.name}` : ''}.
          </p>
          <p className="text-muted-foreground mb-8">{t('pairwise.allCompleteMsg')}</p>
          <div className="flex gap-4 justify-center flex-wrap">
            <button
              onClick={() => {
                setShowDemoComplete(false)
                setDemoMode(false)
                setDemoIndex(0)
                setDemoPairs([])
                setDemoPair(null)
                setSelectedTask(null)
              }}
              className="px-6 py-3 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90"
            >
              {t('pairwise.returnToMainPage')}
            </button>
          </div>
        </div>
      </div>
    )
  }

  if (pair?.status === 'all_completed' && !demoMode) {
    return (
      <div className="text-center py-12">
        <div className="text-6xl mb-4">🎉</div>
        <h2 className="text-3xl font-bold mb-4">{t('pairwise.allComplete')}</h2>
        <p className="text-muted-foreground mb-8">
          {t('pairwise.allCompleteMsg')}
        </p>
        <div className="flex gap-4 justify-center">
          <button
            onClick={() => setShowRanking(true)}
            className="px-6 py-3 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90"
          >
            {t('pairwise.showRanking')}
          </button>
          <button
            onClick={enableDemoMode}
            className="px-6 py-3 border border-primary text-primary rounded-lg hover:bg-primary/10"
          >
            Load Demo Data
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold">{t('pairwise.title')}</h2>
          <p className="text-muted-foreground text-sm mt-1">
            {t('pairwise.subtitle')}
          </p>
          {demoMode && (
            <div className="mt-2 flex items-center gap-2 flex-wrap">
              <span className="px-3 py-1 bg-warning/20 text-warning rounded-full text-xs font-medium">
                🎯 {selectedTask ? selectedTask.name : 'Demo'} — Pair {demoIndex + 1} / {demoPairs.length}
              </span>
              {selectedTask && (
                <button
                  onClick={handleBackToTasks}
                  className="px-2.5 py-1 text-xs text-muted-foreground hover:text-foreground border border-border rounded-full transition-colors"
                >
                  ← Tasks
                </button>
              )}
            </div>
          )}
        </div>
        <div className="flex items-center gap-3 text-sm">
          {!demoMode && (
            <button
              onClick={enableDemoMode}
              className="px-3 py-1.5 border border-primary text-primary rounded-lg hover:bg-primary/10"
            >
              Load Demo Data
            </button>
          )}
          {stats && (
            <div className="text-muted-foreground">
              {t('common.progress')} {stats.unique_pairs_compared} / {stats.total_possible_pairs} {t('common.pairs')}
              ({(stats.completion_rate * 100).toFixed(1)}%)
            </div>
          )}
          <button
            onClick={() => setShowRanking(!showRanking)}
            className="px-3 py-1.5 border rounded-lg hover:bg-accent"
          >
            {t('pairwise.showRanking')}
          </button>
          <button
            onClick={generateShareUrl}
            className="px-3 py-1.5 border rounded-lg hover:bg-accent"
          >
            {t('pairwise.share')}
          </button>
          <button
            onClick={() => navigate('/pairwise-tutorial')}
            className="px-3 py-1.5 border border-primary text-primary rounded-lg hover:bg-primary/10 font-medium"
          >
            Tutorial
          </button>
        </div>
      </div>

      {/* Progress Bar */}
      {stats && (
        <div className="w-full bg-muted rounded-full h-2">
          <div
            className="bg-primary h-2 rounded-full transition-all"
            style={{ width: `${stats.completion_rate * 100}%` }}
          />
        </div>
      )}

      {/* Ranking Panel */}
      {showRanking && ranking && (
        <div className="border border-border rounded-lg p-6 bg-card">
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
                className="flex items-center justify-between p-2 bg-muted/50 rounded"
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
                    item.elo_rating > 1550 ? 'text-destructive' :
                    item.elo_rating < 1450 ? 'text-success' : 'text-muted-foreground'
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
      {(pair || demoPair) && (
        <>
          {/* Videos Side by Side */}
          <div className="grid grid-cols-2 gap-4">
            {/* Left Cow */}
            <div className="space-y-1">
              <div className="text-center font-semibold">{t('pairwise.leftCow')}</div>
              <div className={`border-4 rounded-lg overflow-hidden transition-colors ${
                selectedValue !== null && selectedValue < 0 ? 'border-red-500' : 'border-transparent'
              }`}>
                <video
                  ref={video1Ref}
                  src={demoMode && demoPair ? demoPair.cow_L_URL : pair ? videosApi.getStreamUrl(pair.video_id_1) : ''}
                  className="w-full aspect-video bg-black"
                  loop
                  muted
                  playsInline
                  controls
                  autoPlay={demoMode}
                />
              </div>
            </div>

            {/* Right Cow */}
            <div className="space-y-1">
              <div className="text-center font-semibold">{t('pairwise.rightCow')}</div>
              <div className={`border-4 rounded-lg overflow-hidden transition-colors ${
                selectedValue !== null && selectedValue > 0 ? 'border-orange-500' : 'border-transparent'
              }`}>
                <video
                  ref={video2Ref}
                  src={demoMode && demoPair ? demoPair.cow_R_URL : pair ? videosApi.getStreamUrl(pair.video_id_2) : ''}
                  className="w-full aspect-video bg-black"
                  loop
                  muted
                  playsInline
                  controls
                  autoPlay={demoMode}
                />
              </div>
            </div>
          </div>

          {/* Playback Controls */}
          <div className="flex justify-center gap-3">
            <button
              onClick={togglePlayback}
              className="px-5 py-1.5 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 text-sm"
            >
              {isPlaying ? `⏸ ${t('pairwise.pause')}` : `▶ ${t('pairwise.play')}`}
            </button>
            <button
              onClick={restartVideos}
              className="px-5 py-1.5 border border-border rounded-lg hover:bg-accent text-sm"
            >
              ↺ {t('pairwise.restart')}
            </button>
          </div>

          <>
              {/* 7-Point Comparison Scale - Circular Buttons */}
              <div className="space-y-2">
                <label className="block text-center font-medium text-foreground text-sm">
                  {t('pairwise.selectOption')}
                </label>
                <div className="flex items-center justify-center gap-10 py-3">
                  {COMPARISON_SCALE.map((option, idx) => {
                    const sizeClass =
                      idx === 0 || idx === 6 ? 'w-14 h-14' :
                      idx === 1 || idx === 5 ? 'w-12 h-12' :
                      idx === 2 || idx === 4 ? 'w-10 h-10' :
                      'w-8 h-8'

                    const colorClass =
                      selectedValue === option.value
                        ? option.value < 0 ? 'bg-blue-600 border-blue-700' :
                          option.value > 0 ? 'bg-orange-600 border-orange-700' :
                          'bg-gray-500 border-gray-600'
                        : option.value < 0 ? 'bg-blue-500/50 hover:bg-blue-500/70 border-blue-500/60' :
                          option.value > 0 ? 'bg-orange-500/50 hover:bg-orange-500/70 border-orange-500/60' :
                          'bg-gray-400/40 hover:bg-gray-400/60 border-gray-400/50'

                    const showText = idx === 0 || idx === 6
                    const labelText = idx === 0 ? 'Left cow much more lame' : idx === 6 ? 'Right cow much more lame' : ''

                    return (
                      <div key={option.value} className="flex flex-col items-center gap-1 flex-shrink-0">
                        <span className={`text-xs mb-1 h-4 ${showText ? 'text-muted-foreground' : 'invisible'}`}>
                          {showText ? labelText : 'placeholder'}
                        </span>
                        <button
                          onClick={() => setSelectedValue(option.value)}
                          className={`rounded-full ${sizeClass} ${colorClass} border-2 transition-all flex-shrink-0 ${
                            selectedValue === option.value ? 'ring-3 ring-offset-2 ring-primary scale-110' : ''
                          }`}
                          title={option.label}
                          style={{ aspectRatio: '1 / 1' }}
                        >
                          <span className="sr-only">{option.label}</span>
                        </button>
                      </div>
                    )
                  })}
                </div>
              </div>

              {/* Submit Button */}
              <div className="flex justify-center mt-3">
                <button
                  onClick={handleSubmit}
                  disabled={selectedValue === null || submitting}
                  className="px-8 py-2 bg-success text-white rounded-lg font-medium hover:bg-success/90 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {submitting ? 'Loading...' : t('pairwise.submit')}
                </button>
              </div>
            </>

          {/* Keyboard shortcuts */}
          <div className="text-center text-xs text-muted-foreground">
            Shortcuts: <kbd className="px-1 bg-muted rounded">1-7</kbd> select scale,{' '}
            <kbd className="px-1 bg-muted rounded">Space</kbd> play/pause,{' '}
            <kbd className="px-1 bg-muted rounded">Enter</kbd> submit
          </div>
        </>
      )}

      {/* Share Modal */}
      {showShareModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-card border border-border rounded-lg p-6 max-w-md w-full mx-4 shadow-xl">
            <h3 className="text-lg font-semibold mb-4 text-foreground">Share Comparison</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1 text-foreground">Share URL</label>
                <input
                  type="text"
                  value={shareUrl}
                  readOnly
                  className="w-full p-2 border border-border rounded-lg text-sm bg-background text-foreground"
                  onClick={(e) => (e.target as HTMLInputElement).select()}
                />
              </div>
              <div className="flex justify-center">
                <div className="p-4 bg-muted rounded-lg">
                  {/* QR Code placeholder - would use a QR library in production */}
                  <div className="w-32 h-32 bg-background border-2 border-dashed border-border flex items-center justify-center text-xs text-muted-foreground">
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
                  className="flex-1 px-4 py-2 bg-primary text-primary-foreground rounded-lg"
                >
                  Copy URL
                </button>
                <button
                  onClick={() => setShowShareModal(false)}
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
