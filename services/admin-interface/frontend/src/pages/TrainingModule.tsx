import { useEffect, useState, useRef, useCallback } from 'react'
import { videosApi } from '@/api/client'

// Training levels and their configurations
const TRAINING_LEVELS = [
  { level: 1, name: 'Beginner', minScore: 0, requiredCorrect: 3, difficulty: 'easy' },
  { level: 2, name: 'Apprentice', minScore: 3, requiredCorrect: 5, difficulty: 'easy' },
  { level: 3, name: 'Practitioner', minScore: 8, requiredCorrect: 5, difficulty: 'medium' },
  { level: 4, name: 'Expert', minScore: 13, requiredCorrect: 7, difficulty: 'medium' },
  { level: 5, name: 'Master', minScore: 20, requiredCorrect: 10, difficulty: 'hard' },
]

// Example training data (in production, load from backend)
const TRAINING_EXAMPLES = {
  easy: [
    {
      id: 'easy_1',
      description: 'Clear arched back while walking',
      correctAnswer: 1, // lame
      feedback: 'Correct! This cow shows a pronounced arched back, a classic sign of lameness.',
      hint: 'Look at the spine curvature during walking.',
    },
    {
      id: 'easy_2',
      description: 'Normal straight-back gait',
      correctAnswer: 0, // healthy
      feedback: 'Correct! This cow walks with a straight back and even stride - signs of good health.',
      hint: 'Notice the smooth, even gait pattern.',
    },
    {
      id: 'easy_3',
      description: 'Severe head bobbing',
      correctAnswer: 1,
      feedback: 'Correct! Excessive head bobbing often indicates pain in the front legs.',
      hint: 'Watch the head movement relative to the body.',
    },
  ],
  medium: [
    {
      id: 'med_1',
      description: 'Subtle uneven stride',
      correctAnswer: 1,
      feedback: 'Good catch! This subtle unevenness indicates early-stage lameness.',
      hint: 'Compare the stride length of front vs back legs.',
    },
    {
      id: 'med_2',
      description: 'Slightly hesitant gait',
      correctAnswer: 1,
      feedback: 'Correct! Hesitation and slower movement can indicate discomfort.',
      hint: 'Watch for any pause or reluctance in movement.',
    },
    {
      id: 'med_3',
      description: 'Normal but cautious walking',
      correctAnswer: 0,
      feedback: 'Correct! This cow is walking normally, just at a slower pace.',
      hint: 'Look for actual gait abnormalities, not just speed.',
    },
  ],
  hard: [
    {
      id: 'hard_1',
      description: 'Borderline case - minimal back arch',
      correctAnswer: 1,
      feedback: 'Excellent! Even subtle arch changes can indicate early lameness.',
      hint: 'Focus on the angle of the spine at the withers.',
    },
    {
      id: 'hard_2',
      description: 'Variable gait pattern',
      correctAnswer: 1,
      feedback: 'Great observation! Inconsistent gait patterns often indicate intermittent pain.',
      hint: 'Watch the full duration and look for pattern changes.',
    },
    {
      id: 'hard_3',
      description: 'Quick confident walking',
      correctAnswer: 0,
      feedback: 'Correct! Despite the quick pace, this cow shows no lameness indicators.',
      hint: 'Speed alone is not an indicator - look for gait quality.',
    },
  ],
}

// Rater tiers based on performance
const RATER_TIERS = [
  { tier: 'Bronze', minAccuracy: 0, color: 'text-orange-600', bgColor: 'bg-orange-100', icon: '🥉' },
  { tier: 'Silver', minAccuracy: 0.70, color: 'text-gray-500', bgColor: 'bg-gray-200', icon: '🥈' },
  { tier: 'Gold', minAccuracy: 0.85, color: 'text-yellow-600', bgColor: 'bg-yellow-100', icon: '🥇' },
]

// Sound effects (would be actual audio files in production)
const playSound = (type: 'correct' | 'incorrect' | 'levelup' | 'streak') => {
  // In production, use actual audio files
  console.log(`Playing sound: ${type}`)
  
  // Browser audio feedback using Web Audio API
  try {
    const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)()
    const oscillator = audioContext.createOscillator()
    const gainNode = audioContext.createGain()
    
    oscillator.connect(gainNode)
    gainNode.connect(audioContext.destination)
    
    gainNode.gain.value = 0.1
    
    switch (type) {
      case 'correct':
        oscillator.frequency.value = 880 // A5
        oscillator.type = 'sine'
        break
      case 'incorrect':
        oscillator.frequency.value = 220 // A3
        oscillator.type = 'triangle'
        break
      case 'levelup':
        oscillator.frequency.value = 1047 // C6
        oscillator.type = 'sine'
        break
      case 'streak':
        oscillator.frequency.value = 1319 // E6
        oscillator.type = 'sine'
        break
    }
    
    oscillator.start()
    setTimeout(() => oscillator.stop(), 150)
  } catch (e) {
    // Audio not supported
  }
}

export default function TrainingModule() {
  // User progress state
  const [totalScore, setTotalScore] = useState(() => 
    parseInt(localStorage.getItem('training_score') || '0')
  )
  const [streak, setStreak] = useState(0)
  const [accuracy, setAccuracy] = useState(() => 
    parseFloat(localStorage.getItem('training_accuracy') || '0')
  )
  const [totalAttempts, setTotalAttempts] = useState(() => 
    parseInt(localStorage.getItem('training_attempts') || '0')
  )
  const [correctCount, setCorrectCount] = useState(() => 
    parseInt(localStorage.getItem('training_correct') || '0')
  )
  
  // Current training state
  const [currentLevel, setCurrentLevel] = useState(() => {
    const score = parseInt(localStorage.getItem('training_score') || '0')
    const levels = TRAINING_LEVELS.filter((l: { minScore: number }) => score >= l.minScore)
    const level = levels.length > 0 ? levels[levels.length - 1] : undefined
    return level || TRAINING_LEVELS[0]
  })
  const [currentExample, setCurrentExample] = useState<any>(null)
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null)
  const [showFeedback, setShowFeedback] = useState(false)
  const [showHint, setShowHint] = useState(false)
  const [showLevelUp, setShowLevelUp] = useState(false)
  
  // UI state
  const [loading, setLoading] = useState(false)
  const [view, setView] = useState<'training' | 'progress' | 'leaderboard'>('training')
  
  const videoRef = useRef<HTMLVideoElement>(null)

  useEffect(() => {
    loadNextExample()
  }, [currentLevel])

  useEffect(() => {
    // Save progress to localStorage
    localStorage.setItem('training_score', totalScore.toString())
    localStorage.setItem('training_attempts', totalAttempts.toString())
    localStorage.setItem('training_correct', correctCount.toString())
    localStorage.setItem('training_accuracy', accuracy.toString())
  }, [totalScore, totalAttempts, correctCount, accuracy])

  const loadNextExample = () => {
    setSelectedAnswer(null)
    setShowFeedback(false)
    setShowHint(false)
    
    // Get examples for current difficulty
    const examples = TRAINING_EXAMPLES[currentLevel.difficulty as keyof typeof TRAINING_EXAMPLES]
    const randomIndex = Math.floor(Math.random() * examples.length)
    setCurrentExample(examples[randomIndex])
  }

  const handleAnswer = (answer: number) => {
    if (showFeedback) return
    
    setSelectedAnswer(answer)
    setShowFeedback(true)
    
    const isCorrect = answer === currentExample.correctAnswer
    const newAttempts = totalAttempts + 1
    const newCorrect = correctCount + (isCorrect ? 1 : 0)
    const newAccuracy = newCorrect / newAttempts
    
    setTotalAttempts(newAttempts)
    setCorrectCount(newCorrect)
    setAccuracy(newAccuracy)
    
    if (isCorrect) {
      playSound('correct')
      const points = currentLevel.level * 2 // More points at higher levels
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
    } else {
      playSound('incorrect')
      setStreak(0)
    }
  }

  const handleNext = () => {
    loadNextExample()
  }

  const getCurrentTier = () => {
    const tiers = RATER_TIERS.filter((t: { minAccuracy: number }) => accuracy >= t.minAccuracy)
    return tiers.length > 0 ? tiers[tiers.length - 1] : RATER_TIERS[0]
  }

  const getProgressToNextLevel = () => {
    const nextLevel = TRAINING_LEVELS.find(l => l.minScore > totalScore)
    if (!nextLevel) return 100
    const prevMinScore = currentLevel.minScore
    return ((totalScore - prevMinScore) / (nextLevel.minScore - prevMinScore)) * 100
  }

  const tier = getCurrentTier()

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header with progress */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl p-6 text-white">
        <div className="flex justify-between items-start">
          <div>
            <h1 className="text-3xl font-bold">Lameness Training</h1>
            <p className="text-blue-100 mt-1">Learn to assess cow lameness like an expert</p>
          </div>
          <div className="text-right">
            <div className={`inline-flex items-center gap-2 ${tier.bgColor} ${tier.color} px-3 py-1 rounded-full`}>
              <span className="text-xl">{tier.icon}</span>
              <span className="font-bold">{tier.tier} Rater</span>
            </div>
          </div>
        </div>
        
        <div className="grid grid-cols-4 gap-4 mt-6">
          <div className="bg-white/20 rounded-lg p-3 text-center">
            <div className="text-2xl font-bold">{totalScore}</div>
            <div className="text-sm text-blue-100">Total Points</div>
          </div>
          <div className="bg-white/20 rounded-lg p-3 text-center">
            <div className="text-2xl font-bold">{streak} 🔥</div>
            <div className="text-sm text-blue-100">Streak</div>
          </div>
          <div className="bg-white/20 rounded-lg p-3 text-center">
            <div className="text-2xl font-bold">{(accuracy * 100).toFixed(0)}%</div>
            <div className="text-sm text-blue-100">Accuracy</div>
          </div>
          <div className="bg-white/20 rounded-lg p-3 text-center">
            <div className="text-2xl font-bold">Lv.{currentLevel.level}</div>
            <div className="text-sm text-blue-100">{currentLevel.name}</div>
          </div>
        </div>
        
        {/* Level progress bar */}
        <div className="mt-4">
          <div className="flex justify-between text-sm text-blue-100 mb-1">
            <span>Level {currentLevel.level}</span>
            <span>Level {currentLevel.level + 1}</span>
          </div>
          <div className="h-2 bg-white/30 rounded-full">
            <div 
              className="h-2 bg-yellow-400 rounded-full transition-all"
              style={{ width: `${getProgressToNextLevel()}%` }}
            />
          </div>
        </div>
      </div>

      {/* View tabs */}
      <div className="flex border-b">
        {['training', 'progress', 'leaderboard'].map((v) => (
          <button
            key={v}
            onClick={() => setView(v as any)}
            className={`px-6 py-3 font-medium capitalize ${
              view === v 
                ? 'border-b-2 border-blue-600 text-blue-600'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            {v}
          </button>
        ))}
      </div>

      {/* Training View */}
      {view === 'training' && currentExample && (
        <div className="space-y-6">
          {/* Difficulty indicator */}
          <div className="flex justify-between items-center">
            <span className={`px-3 py-1 rounded-full text-sm font-medium ${
              currentLevel.difficulty === 'easy' ? 'bg-green-100 text-green-700' :
              currentLevel.difficulty === 'medium' ? 'bg-yellow-100 text-yellow-700' :
              'bg-red-100 text-red-700'
            }`}>
              {currentLevel.difficulty.charAt(0).toUpperCase() + currentLevel.difficulty.slice(1)} Difficulty
            </span>
            <button
              onClick={() => setShowHint(!showHint)}
              className="text-sm text-blue-600 hover:underline"
            >
              {showHint ? 'Hide Hint' : 'Need a Hint?'}
            </button>
          </div>

          {/* Question */}
          <div className="border rounded-lg p-6 bg-white">
            <h3 className="text-xl font-semibold mb-2">Is this cow lame?</h3>
            <p className="text-gray-600">{currentExample.description}</p>
            
            {showHint && (
              <div className="mt-4 p-3 bg-blue-50 rounded-lg text-blue-700 text-sm">
                💡 <strong>Hint:</strong> {currentExample.hint}
              </div>
            )}
          </div>

          {/* Video */}
          <div className="border rounded-lg overflow-hidden bg-black">
            {/* In production, load actual video */}
            <div className="aspect-video flex items-center justify-center bg-gray-800">
              <div className="text-center text-white">
                <div className="text-4xl mb-2">🐄</div>
                <p>Training Video: {currentExample.id}</p>
                <p className="text-sm text-gray-400 mt-1">(Demo - actual video would play here)</p>
              </div>
            </div>
          </div>

          {/* Answer buttons */}
          <div className="grid grid-cols-2 gap-4">
            <button
              onClick={() => handleAnswer(0)}
              disabled={showFeedback}
              className={`p-6 rounded-xl border-2 transition-all ${
                showFeedback
                  ? selectedAnswer === 0
                    ? currentExample.correctAnswer === 0
                      ? 'border-green-500 bg-green-50'
                      : 'border-red-500 bg-red-50'
                    : currentExample.correctAnswer === 0
                      ? 'border-green-500 bg-green-50 opacity-50'
                      : 'border-gray-200 opacity-50'
                  : 'border-green-200 hover:border-green-400 hover:bg-green-50'
              }`}
            >
              <div className="text-4xl mb-2">✓</div>
              <div className="font-bold text-lg text-green-700">Healthy</div>
              <div className="text-sm text-gray-500">No lameness detected</div>
            </button>
            
            <button
              onClick={() => handleAnswer(1)}
              disabled={showFeedback}
              className={`p-6 rounded-xl border-2 transition-all ${
                showFeedback
                  ? selectedAnswer === 1
                    ? currentExample.correctAnswer === 1
                      ? 'border-green-500 bg-green-50'
                      : 'border-red-500 bg-red-50'
                    : currentExample.correctAnswer === 1
                      ? 'border-green-500 bg-green-50 opacity-50'
                      : 'border-gray-200 opacity-50'
                  : 'border-red-200 hover:border-red-400 hover:bg-red-50'
              }`}
            >
              <div className="text-4xl mb-2">✗</div>
              <div className="font-bold text-lg text-red-700">Lame</div>
              <div className="text-sm text-gray-500">Shows lameness signs</div>
            </button>
          </div>

          {/* Feedback */}
          {showFeedback && (
            <div className={`p-4 rounded-lg ${
              selectedAnswer === currentExample.correctAnswer
                ? 'bg-green-100 border border-green-300'
                : 'bg-red-100 border border-red-300'
            }`}>
              <div className="flex items-start gap-3">
                <span className="text-2xl">
                  {selectedAnswer === currentExample.correctAnswer ? '🎉' : '📚'}
                </span>
                <div>
                  <h4 className={`font-bold ${
                    selectedAnswer === currentExample.correctAnswer ? 'text-green-800' : 'text-red-800'
                  }`}>
                    {selectedAnswer === currentExample.correctAnswer ? 'Correct!' : 'Not quite right'}
                  </h4>
                  <p className="text-sm mt-1">{currentExample.feedback}</p>
                  
                  {selectedAnswer === currentExample.correctAnswer && streak >= 3 && (
                    <div className="mt-2 text-orange-600 font-medium">
                      🔥 {streak} streak! +{Math.floor(streak / 3)} bonus points!
                    </div>
                  )}
                </div>
              </div>
              
              <button
                onClick={handleNext}
                className="mt-4 w-full py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700"
              >
                Next Example →
              </button>
            </div>
          )}

          {/* Keyboard hints */}
          <div className="text-center text-xs text-gray-400">
            Press <kbd className="px-1 bg-gray-100 rounded">H</kbd> for healthy,{' '}
            <kbd className="px-1 bg-gray-100 rounded">L</kbd> for lame,{' '}
            <kbd className="px-1 bg-gray-100 rounded">?</kbd> for hint
          </div>
        </div>
      )}

      {/* Progress View */}
      {view === 'progress' && (
        <div className="space-y-6">
          <div className="border rounded-lg p-6">
            <h3 className="text-xl font-bold mb-4">Your Progress</h3>
            
            <div className="grid grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium text-gray-500 mb-2">Statistics</h4>
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
                <h4 className="font-medium text-gray-500 mb-2">Rater Status</h4>
                <div className={`p-4 rounded-lg ${tier.bgColor}`}>
                  <div className="flex items-center gap-3">
                    <span className="text-4xl">{tier.icon}</span>
                    <div>
                      <div className={`text-xl font-bold ${tier.color}`}>{tier.tier} Rater</div>
                      <div className="text-sm text-gray-600">
                        {accuracy >= 0.85 
                          ? 'Qualified for real labeling tasks!'
                          : `Need ${((0.85 - accuracy) * 100).toFixed(0)}% more to reach Gold`}
                      </div>
                    </div>
                  </div>
                </div>
                
                {accuracy >= 0.85 && (
                  <div className="mt-4 p-3 bg-green-100 rounded-lg">
                    <p className="text-green-800 font-medium">
                      🎓 Congratulations! You're now qualified to participate in real pairwise comparisons.
                    </p>
                    <button
                      onClick={() => window.location.href = '/pairwise'}
                      className="mt-2 px-4 py-2 bg-green-600 text-white rounded-lg text-sm"
                    >
                      Go to Pairwise Comparison
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Level progression */}
          <div className="border rounded-lg p-6">
            <h3 className="text-xl font-bold mb-4">Level Progression</h3>
            <div className="space-y-4">
              {TRAINING_LEVELS.map((level) => (
                <div 
                  key={level.level}
                  className={`flex items-center gap-4 p-3 rounded-lg ${
                    currentLevel.level === level.level 
                      ? 'bg-blue-100 border-2 border-blue-300'
                      : totalScore >= level.minScore 
                        ? 'bg-green-50'
                        : 'bg-gray-50'
                  }`}
                >
                  <div className={`w-10 h-10 rounded-full flex items-center justify-center font-bold ${
                    totalScore >= level.minScore ? 'bg-green-500 text-white' : 'bg-gray-300 text-gray-600'
                  }`}>
                    {totalScore >= level.minScore ? '✓' : level.level}
                  </div>
                  <div className="flex-1">
                    <div className="font-medium">{level.name}</div>
                    <div className="text-sm text-gray-500">
                      {level.minScore} points required • {level.difficulty} difficulty
                    </div>
                  </div>
                  {currentLevel.level === level.level && (
                    <span className="px-2 py-1 bg-blue-600 text-white text-xs rounded-full">Current</span>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Reset progress */}
          <button
            onClick={() => {
              if (confirm('Are you sure you want to reset all progress?')) {
                localStorage.removeItem('training_score')
                localStorage.removeItem('training_attempts')
                localStorage.removeItem('training_correct')
                localStorage.removeItem('training_accuracy')
                window.location.reload()
              }
            }}
            className="text-red-600 text-sm hover:underline"
          >
            Reset Progress
          </button>
        </div>
      )}

      {/* Leaderboard View */}
      {view === 'leaderboard' && (
        <div className="border rounded-lg p-6">
          <h3 className="text-xl font-bold mb-4">Leaderboard</h3>
          <p className="text-gray-500 mb-4">Coming soon! Compete with other trainees.</p>
          
          {/* Placeholder leaderboard */}
          <div className="space-y-2">
            {[
              { rank: 1, name: 'ExpertRater', score: 500, tier: '🥇' },
              { rank: 2, name: 'CowObserver', score: 350, tier: '🥈' },
              { rank: 3, name: 'VetStudent', score: 280, tier: '🥉' },
              { rank: 4, name: 'You', score: totalScore, tier: tier.icon, isYou: true },
              { rank: 5, name: 'Learner123', score: 100, tier: '🥉' },
            ].sort((a, b) => b.score - a.score).map((entry, idx) => (
              <div 
                key={entry.name}
                className={`flex items-center gap-3 p-3 rounded-lg ${
                  entry.isYou ? 'bg-blue-100 border-2 border-blue-300' : 'bg-gray-50'
                }`}
              >
                <div className="w-8 text-center font-bold text-gray-500">#{idx + 1}</div>
                <div className="text-xl">{entry.tier}</div>
                <div className="flex-1 font-medium">
                  {entry.name}
                  {entry.isYou && <span className="ml-2 text-blue-600">(You)</span>}
                </div>
                <div className="font-bold text-purple-600">{entry.score} pts</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Level up modal */}
      {showLevelUp && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl p-8 text-center animate-bounce">
            <div className="text-6xl mb-4">🎉</div>
            <h2 className="text-2xl font-bold mb-2">Level Up!</h2>
            <p className="text-gray-600 mb-4">
              You've reached <span className="font-bold text-purple-600">Level {currentLevel.level}: {currentLevel.name}</span>
            </p>
            <button
              onClick={() => setShowLevelUp(false)}
              className="px-6 py-3 bg-purple-600 text-white rounded-lg font-medium"
            >
              Continue Training
            </button>
          </div>
        </div>
      )}

      {/* Keyboard shortcuts */}
      {view === 'training' && (
        <script dangerouslySetInnerHTML={{
          __html: `
            document.addEventListener('keydown', function(e) {
              if (e.target.tagName === 'INPUT') return;
              if (e.key === 'h' || e.key === 'H') document.querySelector('[data-answer="0"]')?.click();
              if (e.key === 'l' || e.key === 'L') document.querySelector('[data-answer="1"]')?.click();
              if (e.key === '?') document.querySelector('[data-hint]')?.click();
            });
          `
        }} />
      )}
    </div>
  )
}

