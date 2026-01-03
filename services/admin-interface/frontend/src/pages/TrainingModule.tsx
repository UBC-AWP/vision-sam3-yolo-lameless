import { useEffect, useState, useRef, useCallback } from 'react'
import { videosApi, trainingApi, tutorialApi } from '@/api/client'
import { useAuth } from '@/contexts/AuthContext'

interface Video {
  video_id: string
  filename: string
  uploaded_at?: string
}

// Training levels and their configurations
const TRAINING_LEVELS = [
  { level: 1, name: 'Beginner', minScore: 0, requiredCorrect: 3, difficulty: 'easy' },
  { level: 2, name: 'Apprentice', minScore: 3, requiredCorrect: 5, difficulty: 'easy' },
  { level: 3, name: 'Practitioner', minScore: 8, requiredCorrect: 5, difficulty: 'medium' },
  { level: 4, name: 'Expert', minScore: 13, requiredCorrect: 7, difficulty: 'medium' },
  { level: 5, name: 'Master', minScore: 20, requiredCorrect: 10, difficulty: 'hard' },
]

// Rater tiers based on performance
const RATER_TIERS = [
  { tier: 'Bronze', minAccuracy: 0, color: 'text-orange-600', bgColor: 'bg-orange-100', icon: '🥉' },
  { tier: 'Silver', minAccuracy: 0.70, color: 'text-gray-500', bgColor: 'bg-gray-200', icon: '🥈' },
  { tier: 'Gold', minAccuracy: 0.85, color: 'text-yellow-600', bgColor: 'bg-yellow-100', icon: '🥇' },
]

// 7-point scale labels
const SCALE_LABELS = [
  { value: -3, label: 'A Much More Lame', short: '-3' },
  { value: -2, label: 'A More Lame', short: '-2' },
  { value: -1, label: 'A Slightly More Lame', short: '-1' },
  { value: 0, label: 'Equal / Cannot Decide', short: '0' },
  { value: 1, label: 'B Slightly More Lame', short: '+1' },
  { value: 2, label: 'B More Lame', short: '+2' },
  { value: 3, label: 'B Much More Lame', short: '+3' },
]

// Sound effects
const playSound = (type: 'correct' | 'incorrect' | 'levelup' | 'streak') => {
  try {
    const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)()
    const oscillator = audioContext.createOscillator()
    const gainNode = audioContext.createGain()

    oscillator.connect(gainNode)
    gainNode.connect(audioContext.destination)

    gainNode.gain.value = 0.1

    switch (type) {
      case 'correct':
        oscillator.frequency.value = 880
        oscillator.type = 'sine'
        break
      case 'incorrect':
        oscillator.frequency.value = 220
        oscillator.type = 'triangle'
        break
      case 'levelup':
        oscillator.frequency.value = 1047
        oscillator.type = 'sine'
        break
      case 'streak':
        oscillator.frequency.value = 1319
        oscillator.type = 'sine'
        break
    }

    oscillator.start()
    setTimeout(() => oscillator.stop(), 150)
  } catch (e) {
    // Audio not supported
  }
}

interface TrainingExample {
  id: string
  video_id_1?: string
  video_id_2?: string
  description: string
  hint: string
  correct_winner: number
  correct_degree: number
  difficulty: string
  is_auto_generated?: boolean
}

interface LeaderboardEntry {
  rank: number
  user_id: string
  username: string
  total_score: number
  accuracy: number
  rater_tier: string
  current_level: number
}

export default function TrainingModule() {
  const { user } = useAuth()

  // User progress state
  const [totalScore, setTotalScore] = useState(0)
  const [streak, setStreak] = useState(0)
  const [accuracy, setAccuracy] = useState(0)
  const [totalAttempts, setTotalAttempts] = useState(0)
  const [correctCount, setCorrectCount] = useState(0)
  const [currentLevel, setCurrentLevel] = useState(TRAINING_LEVELS[0])

  // Training examples from backend
  const [trainingExamples, setTrainingExamples] = useState<{
    easy: TrainingExample[]
    medium: TrainingExample[]
    hard: TrainingExample[]
  }>({ easy: [], medium: [], hard: [] })

  // Current training state
  const [currentExample, setCurrentExample] = useState<TrainingExample | null>(null)
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null)
  const [showFeedback, setShowFeedback] = useState(false)
  const [showHint, setShowHint] = useState(false)
  const [showLevelUp, setShowLevelUp] = useState(false)

  // Training mode: 'binary' (healthy/lame) or 'comparison' (7-point scale)
  const [trainingMode, setTrainingMode] = useState<'binary' | 'comparison'>('comparison')

  // UI state
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [view, setView] = useState<'training' | 'progress' | 'leaderboard' | 'setup'>('training')

  // Setup state (for admins)
  const [availableVideos, setAvailableVideos] = useState<Video[]>([])
  const [uploadingVideos, setUploadingVideos] = useState(false)
  const [uploadProgress, setUploadProgress] = useState<{ [key: string]: number }>({})
  const [creatingExample, setCreatingExample] = useState(false)
  const [newExample, setNewExample] = useState({
    video_id_1: '',
    video_id_2: '',
    correct_winner: 0,
    correct_degree: 1,
    difficulty: 'easy',
    description: '',
    hint: ''
  })
  const [hoveredVideoId, setHoveredVideoId] = useState<string | null>(null)
  const [exampleFilter, setExampleFilter] = useState<'all' | 'easy' | 'medium' | 'hard'>('all')
  const [showEditExampleModal, setShowEditExampleModal] = useState(false)
  const [editingExample, setEditingExample] = useState<TrainingExample | null>(null)
  const [savingExample, setSavingExample] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Video playback
  const videoRef1 = useRef<HTMLVideoElement>(null)
  const videoRef2 = useRef<HTMLVideoElement>(null)
  const [isPlaying, setIsPlaying] = useState(false)

  // Leaderboard
  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([])

  // Load progress and examples on mount
  useEffect(() => {
    loadProgress()
    loadExamples()
    loadLeaderboard()
    loadAvailableVideos()
  }, [])

  // Sync progress to backend when it changes
  useEffect(() => {
    if (totalAttempts > 0 && !loading) {
      syncProgress()
    }
  }, [totalScore, totalAttempts, correctCount, streak])

  const loadProgress = async () => {
    try {
      // First try to load from backend
      const userId = user?.id || localStorage.getItem('training_user_id') || 'anonymous'
      const data = await trainingApi.getLearnProgress(userId)

      if (data && data.total_attempts > 0) {
        setTotalScore(data.total_score)
        setTotalAttempts(data.total_attempts)
        setCorrectCount(data.correct_count)
        setStreak(data.streak || 0)
        setAccuracy(data.correct_count / data.total_attempts)

        const level = TRAINING_LEVELS.filter(l => data.total_score >= l.minScore).pop() || TRAINING_LEVELS[0]
        setCurrentLevel(level)
      } else {
        // Fall back to localStorage
        const localScore = parseInt(localStorage.getItem('training_score') || '0')
        const localAttempts = parseInt(localStorage.getItem('training_attempts') || '0')
        const localCorrect = parseInt(localStorage.getItem('training_correct') || '0')

        setTotalScore(localScore)
        setTotalAttempts(localAttempts)
        setCorrectCount(localCorrect)
        setAccuracy(localAttempts > 0 ? localCorrect / localAttempts : 0)

        const level = TRAINING_LEVELS.filter(l => localScore >= l.minScore).pop() || TRAINING_LEVELS[0]
        setCurrentLevel(level)
      }
    } catch (error) {
      console.error('Failed to load progress:', error)
      // Fall back to localStorage
      const localScore = parseInt(localStorage.getItem('training_score') || '0')
      setTotalScore(localScore)
    } finally {
      setLoading(false)
    }
  }

  const loadExamples = async () => {
    try {
      // Start with empty groups
      const grouped: { easy: TrainingExample[]; medium: TrainingExample[]; hard: TrainingExample[] } = {
        easy: [],
        medium: [],
        hard: []
      }

      // Try loading from tutorial API first
      const tutorialData = await tutorialApi.getExamples()
      if (tutorialData.examples && tutorialData.examples.length > 0) {
        // Group by difficulty
        tutorialData.examples.forEach((ex: any) => {
          const diff = ex.difficulty || 'medium'
          if (grouped[diff as keyof typeof grouped]) {
            grouped[diff as keyof typeof grouped].push({
              id: ex.id,
              video_id_1: ex.video_id_1,
              video_id_2: ex.video_id_2,
              description: ex.description || 'Compare these two cows',
              hint: ex.hint || 'Look at the gait patterns',
              correct_winner: ex.correct_answer > 0 ? 2 : ex.correct_answer < 0 ? 1 : 0,
              correct_degree: Math.abs(ex.correct_answer) || 1,
              difficulty: diff
            })
          }
        })
      }

      // Also load auto-generated examples to fill in missing difficulties
      const autoData = await trainingApi.getLearnExamples()
      if (autoData.examples && typeof autoData.examples === 'object') {
        // Merge auto-generated examples for difficulties that have no examples
        for (const diff of ['easy', 'medium', 'hard'] as const) {
          if (grouped[diff].length === 0 && autoData.examples[diff]?.length > 0) {
            grouped[diff] = autoData.examples[diff].map((ex: any) => ({
              id: ex.id,
              video_id_1: ex.video_id_1,
              video_id_2: ex.video_id_2,
              description: ex.description || 'Compare these two cows',
              hint: ex.hint || 'Look at the gait patterns',
              correct_winner: ex.correct_winner || 0,
              correct_degree: ex.correct_degree || 1,
              difficulty: diff,
              is_auto_generated: ex.is_auto_generated
            }))
          }
        }
      }

      setTrainingExamples(grouped)
    } catch (error) {
      console.error('Failed to load examples:', error)
    }
  }

  const loadLeaderboard = async () => {
    try {
      const data = await trainingApi.getLeaderboard(20)
      if (data.leaderboard) {
        setLeaderboard(data.leaderboard)
      }
    } catch (error) {
      console.error('Failed to load leaderboard:', error)
    }
  }

  const loadAvailableVideos = async () => {
    try {
      const data = await videosApi.list(0, 100)
      if (data.videos) {
        setAvailableVideos(data.videos)
      }
    } catch (error) {
      console.error('Failed to load videos:', error)
    }
  }

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files || files.length === 0) return

    setUploadingVideos(true)
    const fileArray = Array.from(files)

    for (let i = 0; i < fileArray.length; i++) {
      const file = fileArray[i]
      try {
        setUploadProgress(prev => ({ ...prev, [file.name]: 0 }))
        await videosApi.upload(file, undefined, (progress) => {
          setUploadProgress(prev => ({ ...prev, [file.name]: progress }))
        })
        setUploadProgress(prev => ({ ...prev, [file.name]: 100 }))
      } catch (error) {
        console.error(`Failed to upload ${file.name}:`, error)
        setUploadProgress(prev => ({ ...prev, [file.name]: -1 }))
      }
    }

    setUploadingVideos(false)
    // Reload videos after upload
    await loadAvailableVideos()
    // Clear file input
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
    setUploadProgress({})
  }

  const handleCreateExample = async () => {
    if (!newExample.video_id_1 || !newExample.video_id_2) {
      alert('Please select both videos')
      return
    }
    if (newExample.video_id_1 === newExample.video_id_2) {
      alert('Please select two different videos')
      return
    }

    setCreatingExample(true)
    try {
      await tutorialApi.createTask({
        video_id_1: newExample.video_id_1,
        video_id_2: newExample.video_id_2,
        correct_winner: newExample.correct_winner,
        correct_degree: newExample.correct_degree,
        difficulty: newExample.difficulty,
        description: newExample.description || `Compare these two cows (${newExample.difficulty} difficulty)`,
        hint: newExample.hint || 'Watch their walking patterns carefully',
        is_tutorial: true
      })

      // Reset form
      setNewExample({
        video_id_1: '',
        video_id_2: '',
        correct_winner: 0,
        correct_degree: 1,
        difficulty: 'easy',
        description: '',
        hint: ''
      })

      // Reload examples
      await loadExamples()
      alert('Training example created successfully!')
    } catch (error) {
      console.error('Failed to create example:', error)
      alert('Failed to create training example')
    } finally {
      setCreatingExample(false)
    }
  }

  const getTotalExamples = () => {
    return Object.values(trainingExamples).reduce((sum, arr) => sum + arr.length, 0)
  }

  const getFilteredExamples = (): TrainingExample[] => {
    if (exampleFilter === 'all') {
      return [...trainingExamples.easy, ...trainingExamples.medium, ...trainingExamples.hard]
    }
    return trainingExamples[exampleFilter] || []
  }

  const handleEditExample = (example: TrainingExample) => {
    setEditingExample({ ...example })
    setShowEditExampleModal(true)
  }

  const handleDeleteExample = async (exampleId: string) => {
    if (!confirm('Are you sure you want to delete this training example?')) return

    try {
      await tutorialApi.deleteTask(exampleId)
      await loadExamples()
    } catch (error) {
      console.error('Failed to delete example:', error)
      alert('Failed to delete training example')
    }
  }

  const handleSaveExample = async () => {
    if (!editingExample) return

    setSavingExample(true)
    try {
      await tutorialApi.updateTask(editingExample.id, {
        video_id_1: editingExample.video_id_1 || '',
        video_id_2: editingExample.video_id_2 || '',
        correct_winner: editingExample.correct_winner,
        correct_degree: editingExample.correct_degree,
        difficulty: editingExample.difficulty,
        description: editingExample.description,
        hint: editingExample.hint,
        is_tutorial: true
      })
      setShowEditExampleModal(false)
      setEditingExample(null)
      await loadExamples()
    } catch (error) {
      console.error('Failed to update example:', error)
      alert('Failed to update training example')
    } finally {
      setSavingExample(false)
    }
  }

  const syncProgress = async () => {
    setSaving(true)
    try {
      const userId = user?.id || localStorage.getItem('training_user_id') || 'anonymous'
      const tier = getCurrentTier()

      await trainingApi.saveLearnProgress({
        total_score: totalScore,
        total_attempts: totalAttempts,
        correct_count: correctCount,
        current_level: currentLevel.level,
        streak: streak,
        rater_tier: tier.tier.toLowerCase()
      }, userId)

      // Also save to localStorage as backup
      localStorage.setItem('training_score', totalScore.toString())
      localStorage.setItem('training_attempts', totalAttempts.toString())
      localStorage.setItem('training_correct', correctCount.toString())
    } catch (error) {
      console.error('Failed to sync progress:', error)
    } finally {
      setSaving(false)
    }
  }

  const loadNextExample = useCallback(() => {
    setSelectedAnswer(null)
    setShowFeedback(false)
    setShowHint(false)
    setIsPlaying(false)

    // Get examples for current difficulty
    const difficulty = currentLevel.difficulty as keyof typeof trainingExamples
    const examples = trainingExamples[difficulty] || []

    if (examples.length > 0) {
      const randomIndex = Math.floor(Math.random() * examples.length)
      setCurrentExample(examples[randomIndex])
    } else {
      // No examples available, show placeholder
      setCurrentExample({
        id: 'placeholder',
        description: 'No training examples available yet',
        hint: 'Ask an admin to create tutorial examples',
        correct_winner: 0,
        correct_degree: 1,
        difficulty: currentLevel.difficulty
      })
    }
  }, [currentLevel, trainingExamples])

  useEffect(() => {
    if (!loading && Object.values(trainingExamples).some(arr => arr.length > 0)) {
      loadNextExample()
    }
  }, [currentLevel, trainingExamples, loading, loadNextExample])

  const handleBinaryAnswer = (answer: number) => {
    if (showFeedback || !currentExample) return

    setSelectedAnswer(answer)
    setShowFeedback(true)

    // For binary mode: 0 = healthy, 1 = lame
    // Map to comparison: if correct_winner is 0 (equal), both answers are "correct"
    // If correct_winner is 1 or 2, check if user picked the lame one
    let isCorrect = false
    if (currentExample.correct_winner === 0) {
      isCorrect = answer === 0 // Equal means both are similar
    } else {
      isCorrect = answer === 1 // Lame is correct if there's a winner
    }

    processAnswer(isCorrect)
  }

  const handleComparisonAnswer = (value: number) => {
    if (showFeedback || !currentExample) return

    setSelectedAnswer(value)
    setShowFeedback(true)

    // Calculate correct answer from winner and degree
    let correctValue = 0
    if (currentExample.correct_winner === 1) {
      correctValue = -currentExample.correct_degree // A is more lame (negative)
    } else if (currentExample.correct_winner === 2) {
      correctValue = currentExample.correct_degree // B is more lame (positive)
    }

    // Check if answer is within acceptable range (exact or 1 step difference for partial credit)
    const diff = Math.abs(value - correctValue)
    const isCorrect = diff === 0
    const isPartiallyCorrect = diff === 1

    processAnswer(isCorrect, isPartiallyCorrect)
  }

  const processAnswer = (isCorrect: boolean, isPartiallyCorrect = false) => {
    const newAttempts = totalAttempts + 1
    const newCorrect = correctCount + (isCorrect ? 1 : 0)
    const newAccuracy = newCorrect / newAttempts

    setTotalAttempts(newAttempts)
    setCorrectCount(newCorrect)
    setAccuracy(newAccuracy)

    if (isCorrect) {
      playSound('correct')
      const points = currentLevel.level * 2
      const streakBonus = streak >= 3 ? Math.floor(streak / 3) : 0
      const newScore = totalScore + points + streakBonus
      setTotalScore(newScore)
      setStreak(prev => prev + 1)

      if (streak + 1 >= 5 && (streak + 1) % 5 === 0) {
        playSound('streak')
      }

      // Check for level up
      const nextLevel = TRAINING_LEVELS.find(l => l.minScore > totalScore && newScore >= l.minScore)
      if (nextLevel) {
        setCurrentLevel(nextLevel)
        setShowLevelUp(true)
        playSound('levelup')
        setTimeout(() => setShowLevelUp(false), 3000)
      }
    } else if (isPartiallyCorrect) {
      // Partial credit: 1 point, reset streak
      const newScore = totalScore + 1
      setTotalScore(newScore)
      setStreak(0)
    } else {
      playSound('incorrect')
      setStreak(0)
    }
  }

  const handleNext = () => {
    loadNextExample()
  }

  const togglePlayback = () => {
    const videos = [videoRef1.current, videoRef2.current].filter(Boolean)
    if (videos.length > 0) {
      if (isPlaying) {
        videos.forEach(v => v?.pause())
      } else {
        videos.forEach(v => v?.play())
      }
      setIsPlaying(!isPlaying)
    }
  }

  const restartVideos = () => {
    const videos = [videoRef1.current, videoRef2.current].filter(Boolean)
    videos.forEach(v => {
      if (v) {
        v.currentTime = 0
        v.play()
      }
    })
    setIsPlaying(true)
  }

  const getCurrentTier = () => {
    const tiers = RATER_TIERS.filter(t => accuracy >= t.minAccuracy)
    return tiers.length > 0 ? tiers[tiers.length - 1] : RATER_TIERS[0]
  }

  const getProgressToNextLevel = () => {
    const nextLevel = TRAINING_LEVELS.find(l => l.minScore > totalScore)
    if (!nextLevel) return 100
    const prevMinScore = currentLevel.minScore
    return ((totalScore - prevMinScore) / (nextLevel.minScore - prevMinScore)) * 100
  }

  const getCorrectAnswerValue = () => {
    if (!currentExample) return 0
    if (currentExample.correct_winner === 1) {
      return -currentExample.correct_degree
    } else if (currentExample.correct_winner === 2) {
      return currentExample.correct_degree
    }
    return 0
  }

  const tier = getCurrentTier()

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <div className="text-muted-foreground">Loading training module...</div>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      {/* Header with progress */}
      <div className="bg-card border border-border rounded-xl p-6">
        <div className="flex justify-between items-start">
          <div>
            <h1 className="text-3xl font-bold text-foreground">Lameness Training</h1>
            <p className="text-muted-foreground mt-1">Learn to assess cow lameness like an expert</p>
          </div>
          <div className="text-right">
            <div className={`inline-flex items-center gap-2 bg-primary/10 text-primary px-3 py-1 rounded-full border border-primary/20`}>
              <span className="text-xl">{tier.icon}</span>
              <span className="font-bold">{tier.tier} Rater</span>
            </div>
            {saving && <div className="text-xs text-muted-foreground mt-1">Saving...</div>}
          </div>
        </div>

        <div className="grid grid-cols-4 gap-4 mt-6">
          <div className="bg-muted/50 border border-border rounded-lg p-3 text-center">
            <div className="text-2xl font-bold text-foreground">{totalScore}</div>
            <div className="text-sm text-muted-foreground">Total Points</div>
          </div>
          <div className="bg-muted/50 border border-border rounded-lg p-3 text-center">
            <div className="text-2xl font-bold text-foreground">{streak} 🔥</div>
            <div className="text-sm text-muted-foreground">Streak</div>
          </div>
          <div className="bg-muted/50 border border-border rounded-lg p-3 text-center">
            <div className="text-2xl font-bold text-foreground">{(accuracy * 100).toFixed(0)}%</div>
            <div className="text-sm text-muted-foreground">Accuracy</div>
          </div>
          <div className="bg-muted/50 border border-border rounded-lg p-3 text-center">
            <div className="text-2xl font-bold text-foreground">Lv.{currentLevel.level}</div>
            <div className="text-sm text-muted-foreground">{currentLevel.name}</div>
          </div>
        </div>

        {/* Level progress bar */}
        <div className="mt-4">
          <div className="flex justify-between text-sm text-muted-foreground mb-1">
            <span>Level {currentLevel.level}</span>
            <span>Level {Math.min(currentLevel.level + 1, 5)}</span>
          </div>
          <div className="h-2 bg-muted rounded-full">
            <div
              className="h-2 bg-primary rounded-full transition-all"
              style={{ width: `${getProgressToNextLevel()}%` }}
            />
          </div>
        </div>
      </div>

      {/* View tabs */}
      <div className="flex border-b border-border">
        {['training', 'progress', 'leaderboard', ...(user?.role === 'admin' ? ['setup'] : [])].map((v) => (
          <button
            key={v}
            onClick={() => setView(v as any)}
            className={`px-6 py-3 font-medium capitalize ${
              view === v
                ? 'border-b-2 border-primary text-primary'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            {v === 'setup' ? 'Setup (Admin)' : v}
          </button>
        ))}
      </div>

      {/* Training View */}
      {view === 'training' && (
        <div className="space-y-6">
          {/* Mode selector */}
          <div className="flex justify-between items-center">
            <div className="flex gap-2">
              <button
                onClick={() => setTrainingMode('comparison')}
                className={`px-4 py-2 rounded-lg text-sm font-medium ${
                  trainingMode === 'comparison'
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-muted text-muted-foreground hover:bg-accent'
                }`}
              >
                7-Point Scale (Recommended)
              </button>
              <button
                onClick={() => setTrainingMode('binary')}
                className={`px-4 py-2 rounded-lg text-sm font-medium ${
                  trainingMode === 'binary'
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-muted text-muted-foreground hover:bg-accent'
                }`}
              >
                Binary (Healthy/Lame)
              </button>
            </div>
            <span className={`px-3 py-1 rounded-full text-sm font-medium ${
              currentLevel.difficulty === 'easy' ? 'bg-success/20 text-success' :
              currentLevel.difficulty === 'medium' ? 'bg-warning/20 text-warning' :
              'bg-destructive/20 text-destructive'
            }`}>
              {currentLevel.difficulty.charAt(0).toUpperCase() + currentLevel.difficulty.slice(1)} Difficulty
            </span>
          </div>

          {currentExample && (
            <>
              {/* Question */}
              <div className="border border-border rounded-lg p-6 bg-card">
                <h3 className="text-xl font-semibold mb-2 text-foreground">
                  {trainingMode === 'comparison'
                    ? 'Which cow is more lame?'
                    : 'Is this cow showing lameness?'}
                </h3>
                <p className="text-muted-foreground">{currentExample.description}</p>

                <div className="flex justify-between items-center mt-4">
                  <button
                    onClick={() => setShowHint(!showHint)}
                    className="text-sm text-blue-600 hover:underline"
                  >
                    {showHint ? 'Hide Hint' : 'Need a Hint?'}
                  </button>
                </div>

                {showHint && (
                  <div className="mt-4 p-3 bg-primary/10 rounded-lg text-primary text-sm">
                    💡 <strong>Hint:</strong> {currentExample.hint}
                  </div>
                )}
              </div>

              {/* Video comparison */}
              {trainingMode === 'comparison' && currentExample.video_id_1 && currentExample.video_id_2 ? (
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-center mb-2">
                      <span className="inline-block px-3 py-1 bg-primary/10 text-primary rounded-full text-sm font-medium">
                        Cow A
                      </span>
                    </div>
                    <div className="border border-border rounded-lg overflow-hidden bg-black">
                      <video
                        ref={videoRef1}
                        src={videosApi.getStreamUrl(currentExample.video_id_1)}
                        className="w-full aspect-video"
                        loop
                        muted
                        playsInline
                      />
                    </div>
                  </div>
                  <div>
                    <div className="text-center mb-2">
                      <span className="inline-block px-3 py-1 bg-secondary text-secondary-foreground rounded-full text-sm font-medium">
                        Cow B
                      </span>
                    </div>
                    <div className="border border-border rounded-lg overflow-hidden bg-black">
                      <video
                        ref={videoRef2}
                        src={videosApi.getStreamUrl(currentExample.video_id_2)}
                        className="w-full aspect-video"
                        loop
                        muted
                        playsInline
                      />
                    </div>
                  </div>
                </div>
              ) : (
                <div className="border rounded-lg overflow-hidden bg-gray-800">
                  <div className="aspect-video flex items-center justify-center">
                    <div className="text-center text-white">
                      <div className="text-6xl mb-4">🐄</div>
                      <p className="text-lg">Training Example</p>
                      <p className="text-sm text-gray-400 mt-1">{currentExample.description}</p>
                      {currentExample.is_auto_generated && (
                        <p className="text-xs text-yellow-400 mt-2">
                          (Auto-generated example - actual answer may vary)
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* Video controls */}
              {currentExample.video_id_1 && (
                <div className="flex justify-center gap-4">
                  <button
                    onClick={restartVideos}
                    className="px-6 py-2 border border-border rounded-lg hover:bg-accent"
                  >
                    ↺ Restart
                  </button>
                  <button
                    onClick={togglePlayback}
                    className="px-6 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90"
                  >
                    {isPlaying ? '⏸ Pause' : '▶ Play'}
                  </button>
                </div>
              )}

              {/* Answer buttons */}
              {trainingMode === 'comparison' ? (
                <div className="space-y-4">
                  <p className="text-center text-sm text-gray-500">
                    Select how much more lame one cow is compared to the other
                  </p>
                  <div className="grid grid-cols-7 gap-2">
                    {SCALE_LABELS.map((item) => {
                      const correctValue = getCorrectAnswerValue()
                      const isSelected = selectedAnswer === item.value
                      const isCorrect = showFeedback && item.value === correctValue
                      const isWrong = showFeedback && isSelected && item.value !== correctValue

                      return (
                        <button
                          key={item.value}
                          onClick={() => handleComparisonAnswer(item.value)}
                          disabled={showFeedback}
                          className={`p-3 rounded-lg border-2 transition-all text-center ${
                            showFeedback
                              ? isCorrect
                                ? 'border-success bg-success/10'
                                : isWrong
                                  ? 'border-destructive bg-destructive/10'
                                  : 'border-border opacity-50'
                              : isSelected
                                ? 'border-primary bg-primary/10'
                                : 'border-border hover:border-primary/50 hover:bg-primary/5'
                          }`}
                        >
                          <div className="text-lg font-bold">{item.short}</div>
                          <div className="text-xs text-muted-foreground mt-1">{item.label}</div>
                        </button>
                      )
                    })}
                  </div>
                  <div className="flex justify-between text-xs text-muted-foreground px-4">
                    <span>← A is more lame</span>
                    <span>B is more lame →</span>
                  </div>
                </div>
              ) : (
                <div className="grid grid-cols-2 gap-4">
                  <button
                    onClick={() => handleBinaryAnswer(0)}
                    disabled={showFeedback}
                    className={`p-6 rounded-xl border-2 transition-all ${
                      showFeedback
                        ? selectedAnswer === 0
                          ? currentExample.correct_winner === 0
                            ? 'border-success bg-success/10'
                            : 'border-destructive bg-destructive/10'
                          : currentExample.correct_winner === 0
                            ? 'border-success bg-success/10 opacity-50'
                            : 'border-border opacity-50'
                        : 'border-success/30 hover:border-success/60 hover:bg-success/5'
                    }`}
                  >
                    <div className="text-4xl mb-2">✓</div>
                    <div className="font-bold text-lg text-success">Healthy</div>
                    <div className="text-sm text-muted-foreground">No lameness detected</div>
                  </button>

                  <button
                    onClick={() => handleBinaryAnswer(1)}
                    disabled={showFeedback}
                    className={`p-6 rounded-xl border-2 transition-all ${
                      showFeedback
                        ? selectedAnswer === 1
                          ? currentExample.correct_winner !== 0
                            ? 'border-success bg-success/10'
                            : 'border-destructive bg-destructive/10'
                          : currentExample.correct_winner !== 0
                            ? 'border-success bg-success/10 opacity-50'
                            : 'border-border opacity-50'
                        : 'border-destructive/30 hover:border-destructive/60 hover:bg-destructive/5'
                    }`}
                  >
                    <div className="text-4xl mb-2">✗</div>
                    <div className="font-bold text-lg text-destructive">Lame</div>
                    <div className="text-sm text-muted-foreground">Shows lameness signs</div>
                  </button>
                </div>
              )}

              {/* Feedback */}
              {showFeedback && (
                <div className={`p-4 rounded-lg ${
                  selectedAnswer === getCorrectAnswerValue() ||
                  (trainingMode === 'binary' && (
                    (selectedAnswer === 0 && currentExample.correct_winner === 0) ||
                    (selectedAnswer === 1 && currentExample.correct_winner !== 0)
                  ))
                    ? 'bg-success/20 border border-success/40'
                    : Math.abs((selectedAnswer || 0) - getCorrectAnswerValue()) === 1
                      ? 'bg-warning/20 border border-warning/40'
                      : 'bg-destructive/20 border border-destructive/40'
                }`}>
                  <div className="flex items-start gap-3">
                    <span className="text-2xl">
                      {selectedAnswer === getCorrectAnswerValue() ||
                       (trainingMode === 'binary' && (
                         (selectedAnswer === 0 && currentExample.correct_winner === 0) ||
                         (selectedAnswer === 1 && currentExample.correct_winner !== 0)
                       ))
                        ? '🎉'
                        : Math.abs((selectedAnswer || 0) - getCorrectAnswerValue()) === 1
                          ? '👍'
                          : '📚'}
                    </span>
                    <div>
                      <h4 className={`font-bold ${
                        selectedAnswer === getCorrectAnswerValue() ||
                        (trainingMode === 'binary' && (
                          (selectedAnswer === 0 && currentExample.correct_winner === 0) ||
                          (selectedAnswer === 1 && currentExample.correct_winner !== 0)
                        ))
                          ? 'text-success'
                          : Math.abs((selectedAnswer || 0) - getCorrectAnswerValue()) === 1
                            ? 'text-warning'
                            : 'text-destructive'
                      }`}>
                        {selectedAnswer === getCorrectAnswerValue() ||
                         (trainingMode === 'binary' && (
                           (selectedAnswer === 0 && currentExample.correct_winner === 0) ||
                           (selectedAnswer === 1 && currentExample.correct_winner !== 0)
                         ))
                          ? 'Correct!'
                          : Math.abs((selectedAnswer || 0) - getCorrectAnswerValue()) === 1
                            ? 'Close! (+1 partial credit)'
                            : 'Not quite right'}
                      </h4>
                      <p className="text-sm mt-1">
                        {trainingMode === 'comparison' && (
                          <>The correct answer was: <strong>{SCALE_LABELS.find(s => s.value === getCorrectAnswerValue())?.label}</strong></>
                        )}
                      </p>

                      {streak >= 3 && selectedAnswer === getCorrectAnswerValue() && (
                        <div className="mt-2 text-warning font-medium">
                          🔥 {streak} streak! +{Math.floor(streak / 3)} bonus points!
                        </div>
                      )}
                    </div>
                  </div>

                  <button
                    onClick={handleNext}
                    className="mt-4 w-full py-3 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90"
                  >
                    Next Example →
                  </button>
                </div>
              )}
            </>
          )}

          {/* No examples message */}
          {!currentExample && !loading && (
            <div className="text-center py-12 border rounded-lg">
              <div className="text-4xl mb-4">📝</div>
              <h3 className="text-xl font-bold mb-2">No Training Examples Available</h3>
              <p className="text-gray-500">
                Ask an administrator to create tutorial examples in the Tutorial Management page.
              </p>
            </div>
          )}

          {/* Training tips */}
          <div className="bg-muted/50 rounded-lg p-4 text-sm text-muted-foreground">
            <h4 className="font-semibold mb-2">Training Tips</h4>
            <ul className="list-disc list-inside space-y-1">
              <li>Watch both videos completely before making a decision</li>
              <li>Look for: arched back, head bobbing, uneven stride, hesitation</li>
              <li>The 7-point scale matches real pairwise comparison tasks</li>
              <li>Reach 85% accuracy to become a Gold tier rater</li>
            </ul>
          </div>
        </div>
      )}

      {/* Progress View */}
      {view === 'progress' && (
        <div className="space-y-6">
          <div className="border border-border rounded-lg p-6 bg-card">
            <h3 className="text-xl font-bold mb-4 text-foreground">Your Progress</h3>

            <div className="grid grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium text-muted-foreground mb-2">Statistics</h4>
                <dl className="space-y-2">
                  <div className="flex justify-between">
                    <dt>Total Attempts</dt>
                    <dd className="font-bold">{totalAttempts}</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt>Correct Answers</dt>
                    <dd className="font-bold text-green-600">{correctCount}</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt>Accuracy Rate</dt>
                    <dd className="font-bold">{(accuracy * 100).toFixed(1)}%</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt>Current Streak</dt>
                    <dd className="font-bold text-orange-600">{streak} 🔥</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt>Total Points</dt>
                    <dd className="font-bold text-purple-600">{totalScore}</dd>
                  </div>
                </dl>
              </div>

              <div>
                <h4 className="font-medium text-muted-foreground mb-2">Rater Status</h4>
                <div className={`p-4 rounded-lg ${tier.bgColor}`}>
                  <div className="flex items-center gap-3">
                    <span className="text-4xl">{tier.icon}</span>
                    <div>
                      <div className={`text-xl font-bold ${tier.color}`}>{tier.tier} Rater</div>
                      <div className="text-sm text-muted-foreground">
                        {accuracy >= 0.85
                          ? 'Qualified for real labeling tasks!'
                          : `Need ${((0.85 - accuracy) * 100).toFixed(0)}% more accuracy to reach Gold`}
                      </div>
                    </div>
                  </div>
                </div>

                {accuracy >= 0.85 && (
                  <div className="mt-4 p-3 bg-success/10 rounded-lg">
                    <p className="text-success font-medium">
                      🎓 Congratulations! You're now qualified to participate in real pairwise comparisons.
                    </p>
                    <button
                      onClick={() => window.location.href = '/pairwise'}
                      className="mt-2 px-4 py-2 bg-success text-white rounded-lg text-sm"
                    >
                      Go to Pairwise Comparison
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Level progression */}
          <div className="border border-border rounded-lg p-6 bg-card">
            <h3 className="text-xl font-bold mb-4 text-foreground">Level Progression</h3>
            <div className="space-y-4">
              {TRAINING_LEVELS.map((level) => (
                <div
                  key={level.level}
                  className={`flex items-center gap-4 p-3 rounded-lg ${
                    currentLevel.level === level.level
                      ? 'bg-primary/10 border-2 border-primary/30'
                      : totalScore >= level.minScore
                        ? 'bg-success/10'
                        : 'bg-muted/50'
                  }`}
                >
                  <div className={`w-10 h-10 rounded-full flex items-center justify-center font-bold ${
                    totalScore >= level.minScore ? 'bg-success text-white' : 'bg-muted text-muted-foreground'
                  }`}>
                    {totalScore >= level.minScore ? '✓' : level.level}
                  </div>
                  <div className="flex-1">
                    <div className="font-medium text-foreground">{level.name}</div>
                    <div className="text-sm text-muted-foreground">
                      {level.minScore} points required • {level.difficulty} difficulty
                    </div>
                  </div>
                  {currentLevel.level === level.level && (
                    <span className="px-2 py-1 bg-primary text-primary-foreground text-xs rounded-full">Current</span>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Reset progress */}
          <button
            onClick={() => {
              if (confirm('Are you sure you want to reset all progress? This cannot be undone.')) {
                localStorage.removeItem('training_score')
                localStorage.removeItem('training_attempts')
                localStorage.removeItem('training_correct')
                setTotalScore(0)
                setTotalAttempts(0)
                setCorrectCount(0)
                setStreak(0)
                setAccuracy(0)
                setCurrentLevel(TRAINING_LEVELS[0])
              }
            }}
            className="text-destructive text-sm hover:underline"
          >
            Reset Progress
          </button>
        </div>
      )}

      {/* Leaderboard View */}
      {view === 'leaderboard' && (
        <div className="border border-border rounded-lg p-6 bg-card">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-xl font-bold">Leaderboard</h3>
            <button
              onClick={loadLeaderboard}
              className="text-sm text-blue-600 hover:underline"
            >
              Refresh
            </button>
          </div>

          {leaderboard.length > 0 ? (
            <div className="space-y-2">
              {leaderboard.map((entry, idx) => {
                const isCurrentUser = user?.id === entry.user_id ||
                  (!user && entry.user_id === 'anonymous')
                const tierIcon = RATER_TIERS.find(t => t.tier.toLowerCase() === entry.rater_tier)?.icon || '🥉'

                return (
                  <div
                    key={entry.user_id}
                    className={`flex items-center gap-3 p-3 rounded-lg ${
                      isCurrentUser ? 'bg-primary/10 border-2 border-primary/30' : 'bg-muted/50'
                    }`}
                  >
                    <div className={`w-8 text-center font-bold ${
                      entry.rank === 1 ? 'text-yellow-500' :
                      entry.rank === 2 ? 'text-gray-400' :
                      entry.rank === 3 ? 'text-orange-400' :
                      'text-gray-500'
                    }`}>
                      #{entry.rank}
                    </div>
                    <div className="text-xl">{tierIcon}</div>
                    <div className="flex-1">
                      <div className="font-medium">
                        {entry.username}
                        {isCurrentUser && <span className="ml-2 text-blue-600">(You)</span>}
                      </div>
                      <div className="text-xs text-gray-500">
                        Level {entry.current_level} • {(entry.accuracy * 100).toFixed(0)}% accuracy
                      </div>
                    </div>
                    <div className="font-bold text-purple-600">{entry.total_score} pts</div>
                  </div>
                )
              })}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              <p>No leaderboard data yet.</p>
              <p className="text-sm">Complete some training to appear on the leaderboard!</p>
            </div>
          )}

          {/* Add current user if not in leaderboard */}
          {totalAttempts > 0 && !leaderboard.some(e => e.user_id === (user?.id || 'anonymous')) && (
            <div className="mt-4 pt-4 border-t border-border">
              <div className="flex items-center gap-3 p-3 rounded-lg bg-primary/10 border-2 border-primary/30">
                <div className="w-8 text-center font-bold text-gray-500">—</div>
                <div className="text-xl">{tier.icon}</div>
                <div className="flex-1">
                  <div className="font-medium">
                    {user?.username || 'Anonymous'} <span className="text-blue-600">(You)</span>
                  </div>
                  <div className="text-xs text-gray-500">
                    Level {currentLevel.level} • {(accuracy * 100).toFixed(0)}% accuracy
                  </div>
                </div>
                <div className="font-bold text-purple-600">{totalScore} pts</div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Setup View (Admin Only) */}
      {view === 'setup' && user?.role === 'admin' && (
        <div className="space-y-6">
          {/* Status */}
          <div className="grid grid-cols-3 gap-4">
            <div className="border border-border rounded-lg p-4 text-center bg-card">
              <div className="text-3xl font-bold text-primary">{availableVideos.length}</div>
              <div className="text-sm text-muted-foreground">Videos Uploaded</div>
            </div>
            <div className="border border-border rounded-lg p-4 text-center bg-card">
              <div className="text-3xl font-bold text-success">{getTotalExamples()}</div>
              <div className="text-sm text-muted-foreground">Training Examples</div>
            </div>
            <div className="border border-border rounded-lg p-4 text-center bg-card">
              <div className="text-3xl font-bold text-secondary-foreground">
                {trainingExamples.easy.length} / {trainingExamples.medium.length} / {trainingExamples.hard.length}
              </div>
              <div className="text-sm text-muted-foreground">Easy / Medium / Hard</div>
            </div>
          </div>

          {/* Upload Videos Section */}
          <div className="border border-border rounded-lg p-6 bg-card">
            <h3 className="text-xl font-bold mb-4 text-foreground">Step 1: Upload Videos</h3>
            <p className="text-muted-foreground mb-4">
              Upload cow walking videos to use for training examples.
            </p>

            <div className="border-2 border-dashed border-border rounded-lg p-8 text-center bg-muted/30">
              <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                multiple
                onChange={handleFileUpload}
                className="hidden"
                id="video-upload"
              />
              <label
                htmlFor="video-upload"
                className="cursor-pointer"
              >
                <div className="text-4xl mb-2">📹</div>
                <div className="text-lg font-medium">
                  {uploadingVideos ? 'Uploading...' : 'Click to upload videos'}
                </div>
                <div className="text-sm text-gray-500 mt-1">
                  Supports MP4, AVI, MOV, MKV
                </div>
              </label>
            </div>

            {/* Upload Progress */}
            {Object.keys(uploadProgress).length > 0 && (
              <div className="mt-4 space-y-2">
                {Object.entries(uploadProgress).map(([name, progress]) => (
                  <div key={name} className="flex items-center gap-3">
                    <div className="flex-1 text-sm truncate">{name}</div>
                    <div className="w-32">
                      <div className="h-2 bg-muted rounded-full">
                        <div
                          className={`h-2 rounded-full ${progress === -1 ? 'bg-destructive' : 'bg-primary'}`}
                          style={{ width: `${progress === -1 ? 100 : progress}%` }}
                        />
                      </div>
                    </div>
                    <div className="w-12 text-sm text-right">
                      {progress === -1 ? 'Error' : `${progress}%`}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Create Training Example Section */}
          <div className="border border-border rounded-lg p-6 bg-card">
            <h3 className="text-xl font-bold mb-4 text-foreground">Step 2: Create Training Example</h3>
            <p className="text-muted-foreground mb-4">
              Drag videos from below to the drop zones, then set the correct answer.
            </p>

            <div className="grid grid-cols-2 gap-6">
              {/* Video A Drop Zone */}
              <div>
                <label className="block text-sm font-medium mb-2">
                  Video A (Cow A)
                  {newExample.video_id_1 && (
                    <button
                      onClick={() => setNewExample({ ...newExample, video_id_1: '' })}
                      className="ml-2 text-xs text-red-500 hover:text-red-700"
                    >
                      Clear
                    </button>
                  )}
                </label>
                <div
                  onDragOver={(e) => { e.preventDefault(); e.currentTarget.classList.add('border-blue-500', 'bg-blue-50') }}
                  onDragLeave={(e) => { e.currentTarget.classList.remove('border-blue-500', 'bg-blue-50') }}
                  onDrop={(e) => {
                    e.preventDefault()
                    e.currentTarget.classList.remove('border-blue-500', 'bg-blue-50')
                    const videoId = e.dataTransfer.getData('video_id')
                    if (videoId && videoId !== newExample.video_id_2) {
                      setNewExample({ ...newExample, video_id_1: videoId })
                    }
                  }}
                  className={`border-2 border-dashed rounded-lg overflow-hidden transition-all ${
                    newExample.video_id_1 ? 'border-blue-400 bg-blue-50' : 'border-gray-300'
                  }`}
                >
                  {newExample.video_id_1 ? (
                    <div className="relative">
                      <video
                        src={videosApi.getStreamUrl(newExample.video_id_1)}
                        className="w-full aspect-video object-cover"
                        controls
                        muted
                      />
                      <div className="absolute top-2 left-2 px-2 py-1 bg-primary text-primary-foreground text-xs font-bold rounded">
                        A
                      </div>
                    </div>
                  ) : (
                    <div className="aspect-video flex flex-col items-center justify-center text-muted-foreground">
                      <span className="text-4xl mb-2">📥</span>
                      <span className="text-sm">Drop Video A here</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Video B Drop Zone */}
              <div>
                <label className="block text-sm font-medium mb-2">
                  Video B (Cow B)
                  {newExample.video_id_2 && (
                    <button
                      onClick={() => setNewExample({ ...newExample, video_id_2: '' })}
                      className="ml-2 text-xs text-red-500 hover:text-red-700"
                    >
                      Clear
                    </button>
                  )}
                </label>
                <div
                  onDragOver={(e) => { e.preventDefault(); e.currentTarget.classList.add('border-green-500', 'bg-green-50') }}
                  onDragLeave={(e) => { e.currentTarget.classList.remove('border-green-500', 'bg-green-50') }}
                  onDrop={(e) => {
                    e.preventDefault()
                    e.currentTarget.classList.remove('border-green-500', 'bg-green-50')
                    const videoId = e.dataTransfer.getData('video_id')
                    if (videoId && videoId !== newExample.video_id_1) {
                      setNewExample({ ...newExample, video_id_2: videoId })
                    }
                  }}
                  className={`border-2 border-dashed rounded-lg overflow-hidden transition-all ${
                    newExample.video_id_2 ? 'border-green-400 bg-green-50' : 'border-gray-300'
                  }`}
                >
                  {newExample.video_id_2 ? (
                    <div className="relative">
                      <video
                        src={videosApi.getStreamUrl(newExample.video_id_2)}
                        className="w-full aspect-video object-cover"
                        controls
                        muted
                      />
                      <div className="absolute top-2 left-2 px-2 py-1 bg-green-500 text-white text-xs font-bold rounded">
                        B
                      </div>
                    </div>
                  ) : (
                    <div className="aspect-video flex flex-col items-center justify-center text-gray-400">
                      <span className="text-4xl mb-2">📥</span>
                      <span className="text-sm">Drop Video B here</span>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Available Videos - Draggable */}
            {availableVideos.length > 0 && (
              <div className="mt-6 border-t border-border pt-4">
                <h4 className="font-medium mb-2 text-foreground">Available Videos ({availableVideos.length}) - Drag to drop zones above</h4>
                <div className="grid grid-cols-6 gap-2 max-h-64 overflow-y-auto p-2 bg-muted/50 rounded-lg">
                  {availableVideos.map((video) => {
                    const isSelectedA = newExample.video_id_1 === video.video_id
                    const isSelectedB = newExample.video_id_2 === video.video_id
                    return (
                      <div
                        key={video.video_id}
                        draggable
                        onDragStart={(e) => {
                          e.dataTransfer.setData('video_id', video.video_id)
                          e.dataTransfer.effectAllowed = 'copy'
                        }}
                        className={`relative border-2 rounded-lg overflow-hidden cursor-grab active:cursor-grabbing transition-all ${
                          isSelectedA ? 'border-primary ring-2 ring-primary/30' :
                          isSelectedB ? 'border-success ring-2 ring-success/30' :
                          'border-border hover:border-muted-foreground hover:shadow-md bg-card'
                        }`}
                      >
                        <div className="relative">
                          <video
                            src={videosApi.getStreamUrl(video.video_id)}
                            className="w-full aspect-video object-cover bg-gray-100"
                            muted
                            onMouseEnter={(e) => e.currentTarget.play()}
                            onMouseLeave={(e) => { e.currentTarget.pause(); e.currentTarget.currentTime = 0 }}
                          />
                          {(isSelectedA || isSelectedB) && (
                            <div className={`absolute top-1 right-1 px-1.5 py-0.5 rounded text-xs font-bold text-white ${
                              isSelectedA ? 'bg-primary' : 'bg-success'
                            }`}>
                              {isSelectedA ? 'A' : 'B'}
                            </div>
                          )}
                        </div>
                        <div className="p-1 text-[10px] truncate bg-card border-t border-border text-center text-foreground">
                          {(video.filename || video.video_id).slice(0, 8)}...
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {/* Correct Answer */}
            <div className="mt-6">
              <label className="block text-sm font-medium mb-2 text-foreground">Correct Answer (Which cow is more lame?)</label>
              <select
                value={`${newExample.correct_winner}:${newExample.correct_degree}`}
                onChange={(e) => {
                  const [winner, degree] = e.target.value.split(':').map(Number)
                  setNewExample({ ...newExample, correct_winner: winner, correct_degree: degree })
                }}
                className="w-full p-2 border border-border rounded-lg bg-background text-foreground"
              >
                <option value="1:3">A Much More Lame (-3)</option>
                <option value="1:2">A More Lame (-2)</option>
                <option value="1:1">A Slightly More Lame (-1)</option>
                <option value="0:1">Equal / Cannot Decide (0)</option>
                <option value="2:1">B Slightly More Lame (+1)</option>
                <option value="2:2">B More Lame (+2)</option>
                <option value="2:3">B Much More Lame (+3)</option>
              </select>
            </div>

            {/* Difficulty */}
            <div className="mt-4">
              <label className="block text-sm font-medium mb-2 text-foreground">Difficulty Level</label>
              <select
                value={newExample.difficulty}
                onChange={(e) => setNewExample({ ...newExample, difficulty: e.target.value })}
                className="w-full p-2 border border-border rounded-lg bg-background text-foreground"
              >
                <option value="easy">Easy (Obvious difference)</option>
                <option value="medium">Medium (Moderate difference)</option>
                <option value="hard">Hard (Subtle difference)</option>
              </select>
            </div>

            {/* Optional Fields */}
            <div className="mt-4 grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-2 text-foreground">Description (Optional)</label>
                <input
                  type="text"
                  value={newExample.description}
                  onChange={(e) => setNewExample({ ...newExample, description: e.target.value })}
                  placeholder="e.g., Compare these two cows walking"
                  className="w-full p-2 border border-border rounded-lg bg-background text-foreground"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2 text-foreground">Hint (Optional)</label>
                <input
                  type="text"
                  value={newExample.hint}
                  onChange={(e) => setNewExample({ ...newExample, hint: e.target.value })}
                  placeholder="e.g., Look at the back arch"
                  className="w-full p-2 border border-border rounded-lg bg-background text-foreground"
                />
              </div>
            </div>

            {/* Create Button */}
            <div className="mt-6">
              <button
                onClick={handleCreateExample}
                disabled={creatingExample || !newExample.video_id_1 || !newExample.video_id_2}
                className="w-full py-3 bg-success text-white rounded-lg font-medium hover:bg-success/90 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {creatingExample ? 'Creating...' : 'Create Training Example'}
              </button>
            </div>
          </div>

          {/* Existing Examples - Table Format Like TutorialManagement */}
          <div className="border border-border rounded-lg p-6 bg-card">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-xl font-bold">Existing Training Examples</h3>
              <button
                onClick={loadExamples}
                className="text-sm text-blue-600 hover:underline"
              >
                Refresh
              </button>
            </div>

            {/* Filter tabs by difficulty */}
            <div className="flex gap-2 border-b border-border mb-4">
              {(['all', 'easy', 'medium', 'hard'] as const).map((tab) => {
                const count = tab === 'all'
                  ? getTotalExamples()
                  : trainingExamples[tab as keyof typeof trainingExamples]?.length || 0
                return (
                  <button
                    key={tab}
                    onClick={() => setExampleFilter(tab)}
                    className={`px-4 py-2 text-sm font-medium transition-colors ${
                      exampleFilter === tab
                        ? 'text-primary border-b-2 border-primary'
                        : 'text-muted-foreground hover:text-foreground'
                    }`}
                  >
                    {tab === 'all' ? 'All' : tab.charAt(0).toUpperCase() + tab.slice(1)} ({count})
                  </button>
                )
              })}
            </div>

            {getTotalExamples() === 0 ? (
              <div className="text-center py-8 text-gray-500">
                <p>No training examples yet.</p>
                <p className="text-sm">Upload videos and create examples above.</p>
              </div>
            ) : (
              <div className="border border-border rounded-lg overflow-hidden">
                <table className="w-full">
                  <thead className="bg-muted/50">
                    <tr>
                      <th className="px-4 py-3 text-left text-sm font-medium text-foreground">Videos</th>
                      <th className="px-4 py-3 text-left text-sm font-medium text-foreground">Correct Answer</th>
                      <th className="px-4 py-3 text-left text-sm font-medium text-foreground">Difficulty</th>
                      <th className="px-4 py-3 text-left text-sm font-medium text-foreground">Description</th>
                      <th className="px-4 py-3 text-left text-sm font-medium text-foreground">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {getFilteredExamples().map((ex) => {
                      const correctValue = ex.correct_winner === 1 ? -ex.correct_degree :
                                          ex.correct_winner === 2 ? ex.correct_degree : 0
                      const answerLabel = SCALE_LABELS.find(s => s.value === correctValue)?.label || 'Equal'

                      return (
                        <tr key={ex.id} className="hover:bg-muted/30">
                          <td className="px-4 py-3">
                            <div className="flex items-center gap-2">
                              {ex.video_id_1 ? (
                                <div className="w-24 h-14 bg-gray-100 rounded overflow-hidden relative group">
                                  <video
                                    src={videosApi.getStreamUrl(ex.video_id_1)}
                                    className="w-full h-full object-cover"
                                    muted
                                    onMouseEnter={(e) => e.currentTarget.play()}
                                    onMouseLeave={(e) => { e.currentTarget.pause(); e.currentTarget.currentTime = 0 }}
                                  />
                                  <div className="absolute top-1 left-1 px-1 py-0.5 bg-primary text-primary-foreground text-xs font-bold rounded">A</div>
                                </div>
                              ) : (
                                <div className="w-24 h-14 bg-muted rounded flex items-center justify-center text-muted-foreground">A</div>
                              )}
                              <span className="text-muted-foreground">vs</span>
                              {ex.video_id_2 ? (
                                <div className="w-24 h-14 bg-muted rounded overflow-hidden relative group">
                                  <video
                                    src={videosApi.getStreamUrl(ex.video_id_2)}
                                    className="w-full h-full object-cover"
                                    muted
                                    onMouseEnter={(e) => e.currentTarget.play()}
                                    onMouseLeave={(e) => { e.currentTarget.pause(); e.currentTarget.currentTime = 0 }}
                                  />
                                  <div className="absolute top-1 left-1 px-1 py-0.5 bg-success text-primary-foreground text-xs font-bold rounded">B</div>
                                </div>
                              ) : (
                                <div className="w-24 h-14 bg-muted rounded flex items-center justify-center text-muted-foreground">B</div>
                              )}
                            </div>
                          </td>
                          <td className="px-4 py-3 text-sm">{answerLabel}</td>
                          <td className="px-4 py-3">
                            <span className={`px-2 py-1 rounded-full text-xs font-medium capitalize ${
                              ex.difficulty === 'easy' ? 'bg-success/20 text-success' :
                              ex.difficulty === 'medium' ? 'bg-warning/20 text-warning' :
                              'bg-destructive/20 text-destructive'
                            }`}>
                              {ex.difficulty}
                            </span>
                            {ex.is_auto_generated && (
                              <span className="ml-1 px-1 py-0.5 bg-muted text-muted-foreground text-xs rounded">Auto</span>
                            )}
                          </td>
                          <td className="px-4 py-3 text-sm text-muted-foreground max-w-xs truncate">
                            {ex.description || '-'}
                          </td>
                          <td className="px-4 py-3">
                            <div className="flex gap-2">
                              <button
                                onClick={() => handleEditExample(ex)}
                                className="px-3 py-1 text-sm border rounded hover:bg-accent"
                              >
                                Edit
                              </button>
                              <button
                                onClick={() => handleDeleteExample(ex.id)}
                                className="px-3 py-1 text-sm text-destructive border border-destructive/30 rounded hover:bg-destructive/10"
                              >
                                Delete
                              </button>
                            </div>
                          </td>
                        </tr>
                      )
                    })}
                    {getFilteredExamples().length === 0 && (
                      <tr>
                        <td colSpan={5} className="px-4 py-8 text-center text-muted-foreground">
                          No examples found for this filter.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Edit Example Modal */}
          {showEditExampleModal && editingExample && (
            <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
              <div className="bg-card border border-border rounded-lg p-6 max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto shadow-xl">
                <h3 className="text-lg font-semibold mb-4 text-foreground">Edit Training Example</h3>

                <div className="space-y-4">
                  {/* Video Previews */}
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium mb-1 text-foreground">Video A</label>
                      {editingExample.video_id_1 ? (
                        <div className="aspect-video bg-black rounded overflow-hidden">
                          <video
                            src={videosApi.getStreamUrl(editingExample.video_id_1)}
                            className="w-full h-full object-contain"
                            controls
                            muted
                          />
                        </div>
                      ) : (
                        <div className="aspect-video bg-muted rounded flex items-center justify-center text-muted-foreground">
                          No video
                        </div>
                      )}
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1 text-foreground">Video B</label>
                      {editingExample.video_id_2 ? (
                        <div className="aspect-video bg-black rounded overflow-hidden">
                          <video
                            src={videosApi.getStreamUrl(editingExample.video_id_2)}
                            className="w-full h-full object-contain"
                            controls
                            muted
                          />
                        </div>
                      ) : (
                        <div className="aspect-video bg-muted rounded flex items-center justify-center text-muted-foreground">
                          No video
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Correct Answer */}
                  <div>
                    <label className="block text-sm font-medium mb-1 text-foreground">Correct Answer</label>
                    <select
                      value={`${editingExample.correct_winner}:${editingExample.correct_degree}`}
                      onChange={(e) => {
                        const [winner, degree] = e.target.value.split(':').map(Number)
                        setEditingExample({ ...editingExample, correct_winner: winner, correct_degree: degree })
                      }}
                      className="w-full p-2 border border-border rounded-lg bg-background text-foreground"
                    >
                      <option value="1:3">A Much More Lame (-3)</option>
                      <option value="1:2">A More Lame (-2)</option>
                      <option value="1:1">A Slightly More Lame (-1)</option>
                      <option value="0:1">Equal / Cannot Decide (0)</option>
                      <option value="2:1">B Slightly More Lame (+1)</option>
                      <option value="2:2">B More Lame (+2)</option>
                      <option value="2:3">B Much More Lame (+3)</option>
                    </select>
                  </div>

                  {/* Difficulty */}
                  <div>
                    <label className="block text-sm font-medium mb-1 text-foreground">Difficulty</label>
                    <select
                      value={editingExample.difficulty}
                      onChange={(e) => setEditingExample({ ...editingExample, difficulty: e.target.value })}
                      className="w-full p-2 border border-border rounded-lg bg-background text-foreground"
                    >
                      <option value="easy">Easy - Obvious difference</option>
                      <option value="medium">Medium - Moderate difference</option>
                      <option value="hard">Hard - Subtle difference</option>
                    </select>
                  </div>

                  {/* Description */}
                  <div>
                    <label className="block text-sm font-medium mb-1 text-foreground">Description</label>
                    <textarea
                      value={editingExample.description}
                      onChange={(e) => setEditingExample({ ...editingExample, description: e.target.value })}
                      className="w-full p-2 border border-border rounded-lg bg-background text-foreground"
                      rows={2}
                      placeholder="Describe what the rater should observe..."
                    />
                  </div>

                  {/* Hint */}
                  <div>
                    <label className="block text-sm font-medium mb-1 text-foreground">Hint</label>
                    <textarea
                      value={editingExample.hint}
                      onChange={(e) => setEditingExample({ ...editingExample, hint: e.target.value })}
                      className="w-full p-2 border border-border rounded-lg bg-background text-foreground"
                      rows={2}
                      placeholder="Explain why this is the correct answer..."
                    />
                  </div>
                </div>

                <div className="flex justify-end gap-2 mt-6">
                  <button
                    onClick={() => {
                      setShowEditExampleModal(false)
                      setEditingExample(null)
                    }}
                    className="px-4 py-2 border rounded-lg hover:bg-accent"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleSaveExample}
                    disabled={savingExample}
                    className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50"
                  >
                    {savingExample ? 'Saving...' : 'Save Changes'}
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Level up modal */}
      {showLevelUp && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-card border border-border rounded-xl p-8 text-center animate-bounce shadow-xl">
            <div className="text-6xl mb-4">🎉</div>
            <h2 className="text-2xl font-bold mb-2 text-foreground">Level Up!</h2>
            <p className="text-muted-foreground mb-4">
              You've reached <span className="font-bold text-primary">Level {currentLevel.level}: {currentLevel.name}</span>
            </p>
            <button
              onClick={() => setShowLevelUp(false)}
              className="px-6 py-3 bg-primary text-primary-foreground rounded-lg font-medium"
            >
              Continue Training
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
