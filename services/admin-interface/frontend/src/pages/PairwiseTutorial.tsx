import { useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { cn } from '@/lib/utils'
import { useLanguage } from '@/contexts/LanguageContext'
import {
  computeFeedback,
  getValidDemoPairs,
  type DemoConsensusPair,
  type FeedbackResult,
} from '@/utils/pairwiseConsensus'

const COMPARISON_SCALE = [
  { value: -3, label: 'Left Much More Lame' },
  { value: -2, label: 'Left More Lame' },
  { value: -1, label: 'Left Slightly More Lame' },
  { value: 0, label: 'Equal / Cannot Decide' },
  { value: 1, label: 'Right Slightly More Lame' },
  { value: 2, label: 'Right More Lame' },
  { value: 3, label: 'Right Much More Lame' },
]

interface TutorialPair {
  pairKey: string
  leftCowId: string
  rightCowId: string
  leftUrl: string
  rightUrl: string
  source: DemoConsensusPair
}

function toDisplayScaleValue(pair: TutorialPair, canonicalValue: number): number {
  const [minCow, maxCow] = pair.pairKey.split('_')
  return pair.leftCowId === minCow ? canonicalValue : -canonicalValue
}

function buildDisplayHistogram(pair: TutorialPair): Array<{ x: number; count: number }> {
  const buckets = new Map<number, number>()
  for (let x = -3; x <= 3; x++) buckets.set(x, 0)
  for (const s of pair.source.consensusData.scores) {
    const x = toDisplayScaleValue(pair, s)
    buckets.set(x, (buckets.get(x) ?? 0) + 1)
  }
  return Array.from(buckets.entries()).map(([x, count]) => ({ x, count }))
}

function buildTutorialPairs(): TutorialPair[] {
  const base = getValidDemoPairs()
    .filter(p => !p.consensusData.allEqual)
    .sort((a, b) => {
      const strengthDiff = Math.abs(b.consensusData.mean) - Math.abs(a.consensusData.mean)
      if (Math.abs(strengthDiff) > 1e-6) return strengthDiff
      return b.consensusData.agreePercent - a.consensusData.agreePercent
    })
    .slice(0, 5)

  return base.map((pair) => {
    const flipped = Math.random() < 0.5
    return flipped
      ? {
          pairKey: pair.pairKey,
          leftCowId: pair.cow_R,
          rightCowId: pair.cow_L,
          leftUrl: pair.cow_R_URL,
          rightUrl: pair.cow_L_URL,
          source: pair,
        }
      : {
          pairKey: pair.pairKey,
          leftCowId: pair.cow_L,
          rightCowId: pair.cow_R,
          leftUrl: pair.cow_L_URL,
          rightUrl: pair.cow_R_URL,
          source: pair,
        }
  })
}

export default function PairwiseTutorial() {
  const navigate = useNavigate()
  const { t } = useLanguage()
  const video1Ref = useRef<HTMLVideoElement>(null)
  const video2Ref = useRef<HTMLVideoElement>(null)

  const pairs = useMemo(() => buildTutorialPairs(), [])
  const [index, setIndex] = useState(0)
  const [selectedValue, setSelectedValue] = useState<number | null>(null)
  const [feedback, setFeedback] = useState<FeedbackResult | null>(null)
  const [isPlaying, setIsPlaying] = useState(true)
  const [answers, setAnswers] = useState<Record<string, number>>({})

  const current = pairs[index]
  const completedCount = Object.keys(answers).length
  const consensusDisplayMean = current
    ? toDisplayScaleValue(current, current.source.consensusData.mean)
    : 0
  const histogram = current ? buildDisplayHistogram(current) : []
  const maxBin = histogram.reduce((m, b) => Math.max(m, b.count), 1)
  const userDistance =
    selectedValue === null ? null : Math.abs(selectedValue - consensusDisplayMean)

  const onSelect = (value: number) => {
    if (!current) return
    setSelectedValue(value)
    setAnswers(prev => ({ ...prev, [current.pairKey]: value }))
    setFeedback(computeFeedback(current.leftCowId, current.rightCowId, value))
  }

  const nextPair = () => {
    if (index >= pairs.length - 1) {
      navigate('/pairwise')
      return
    }
    setIndex(prev => prev + 1)
    setSelectedValue(null)
    setFeedback(null)
    setIsPlaying(true)
  }

  const togglePlayback = () => {
    if (!video1Ref.current || !video2Ref.current) return
    if (isPlaying) {
      video1Ref.current.pause()
      video2Ref.current.pause()
    } else {
      video1Ref.current.play()
      video2Ref.current.play()
    }
    setIsPlaying(!isPlaying)
  }

  const restartVideos = () => {
    if (!video1Ref.current || !video2Ref.current) return
    video1Ref.current.currentTime = 0
    video2Ref.current.currentTime = 0
    video1Ref.current.play()
    video2Ref.current.play()
    setIsPlaying(true)
  }

  if (!current) {
    return (
      <div className="text-center py-16">
        <h2 className="text-2xl font-bold mb-2">Tutorial Unavailable</h2>
        <p className="text-muted-foreground mb-6">No tutorial pairs found in demo data.</p>
        <button
          onClick={() => navigate('/pairwise')}
          className="px-5 py-2 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90"
        >
          Back to Pairwise
        </button>
      </div>
    )
  }

  return (
    <div className="max-w-5xl mx-auto space-y-5">
      <div className="rounded-xl border border-primary/30 bg-primary/10 p-5">
        <h2 className="text-2xl font-bold">{t('tutorial.title')}</h2>
        <p className="text-sm text-muted-foreground mt-1">
          {t('tutorial.subtitle')}
        </p>
        <div className="mt-3 flex items-center gap-3 text-sm">
          <span className="font-medium">{t('tutorial.step')} {index + 1} {t('tutorial.of')} {pairs.length}</span>
          <span className="text-muted-foreground">{t('tutorial.answered')} {completedCount}/{pairs.length}</span>
        </div>
        <div className="mt-2 h-2 rounded-full bg-primary/20 overflow-hidden">
          <div
            className="h-full bg-primary rounded-full transition-all"
            style={{ width: `${((index + 1) / pairs.length) * 100}%` }}
          />
        </div>
      </div>

      <div className="bg-muted/50 rounded-lg p-4 text-sm">
        <h4 className="font-semibold mb-2">{t('tutorial.whatToLookFor')}</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="flex items-start gap-2">
            <span className="text-red-500">●</span>
            <div>
              <div className="font-medium">{t('tutorial.archedBack')}</div>
              <div className="text-muted-foreground">{t('tutorial.archedBackDesc')}</div>
            </div>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-orange-500">●</span>
            <div>
              <div className="font-medium">{t('tutorial.headBobbing')}</div>
              <div className="text-muted-foreground">{t('tutorial.headBobbingDesc')}</div>
            </div>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-yellow-500">●</span>
            <div>
              <div className="font-medium">{t('tutorial.unevenStride')}</div>
              <div className="text-muted-foreground">{t('tutorial.unevenStrideDesc')}</div>
            </div>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-blue-500">●</span>
            <div>
              <div className="font-medium">{t('tutorial.slowMovement')}</div>
              <div className="text-muted-foreground">{t('tutorial.slowMovementDesc')}</div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-1">
          <div className="text-center font-semibold">{t('pairwise.leftCow')} ({current.leftCowId})</div>
          <video
            ref={video1Ref}
            src={current.leftUrl}
            className="w-full aspect-video bg-black rounded-lg"
            loop
            muted
            playsInline
            controls
            autoPlay
          />
        </div>
        <div className="space-y-1">
          <div className="text-center font-semibold">{t('pairwise.rightCow')} ({current.rightCowId})</div>
          <video
            ref={video2Ref}
            src={current.rightUrl}
            className="w-full aspect-video bg-black rounded-lg"
            loop
            muted
            playsInline
            controls
            autoPlay
          />
        </div>
      </div>

      <div className="flex justify-center gap-3">
        <button
          onClick={togglePlayback}
          className="px-5 py-1.5 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 text-sm"
        >
          {isPlaying ? t('pairwise.pause') : t('pairwise.play')}
        </button>
        <button
          onClick={nextPair}
          disabled={selectedValue === null}
          className="px-5 py-1.5 border border-border rounded-lg hover:bg-accent text-sm disabled:opacity-50"
        >
          {index === pairs.length - 1 ? t('tutorial.finish') : t('tutorial.nextPair')}
        </button>
      </div>

      <div className="space-y-2">
        <label className="block text-center font-medium text-sm">
          Select lameness difference
        </label>
        <div className="flex items-center justify-center gap-8 py-2">
          {COMPARISON_SCALE.map((option, idx) => {
            const sizeClass =
              idx === 0 || idx === 6 ? 'w-12 h-12' :
              idx === 1 || idx === 5 ? 'w-10 h-10' :
              idx === 2 || idx === 4 ? 'w-9 h-9' :
              'w-7 h-7'
            const colorClass =
              selectedValue === option.value
                ? option.value < 0 ? 'bg-blue-600 border-blue-700' :
                  option.value > 0 ? 'bg-orange-600 border-orange-700' :
                  'bg-gray-500 border-gray-600'
                : option.value < 0 ? 'bg-blue-500/50 hover:bg-blue-500/70 border-blue-500/60' :
                  option.value > 0 ? 'bg-orange-500/50 hover:bg-orange-500/70 border-orange-500/60' :
                  'bg-gray-400/40 hover:bg-gray-400/60 border-gray-400/50'
            return (
              <button
                key={option.value}
                onClick={() => onSelect(option.value)}
                className={cn(
                  `rounded-full border-2 ${sizeClass} ${colorClass} transition-all`,
                  selectedValue === option.value && 'ring-2 ring-primary ring-offset-2 scale-110',
                )}
                title={option.label}
              >
                <span className="sr-only">{option.label}</span>
              </button>
            )
          })}
        </div>
      </div>

      {feedback && (
        <div
          className={cn(
            'rounded-xl border p-4 text-center',
            feedback.type === 'great'
              ? 'bg-emerald-500/10 border-emerald-500/30'
              : feedback.type === 'good'
              ? 'bg-blue-500/10 border-blue-500/30'
              : feedback.type === 'interesting'
              ? 'bg-amber-500/10 border-amber-500/30'
              : 'bg-red-500/10 border-red-500/30',
          )}
        >
          <div className="text-3xl">{feedback.emoji}</div>
          <div className="font-semibold mt-1">{feedback.message}</div>
          <p className="text-sm text-muted-foreground mt-1">{feedback.details}</p>

          <div className="mt-4 space-y-2">
            <div className="text-xs text-muted-foreground">
              Crowd center: <span className="font-semibold text-foreground">{consensusDisplayMean.toFixed(1)}</span>
              {userDistance !== null && (
                <>
                  {' '}· Your distance: <span className="font-semibold text-foreground">{userDistance.toFixed(1)}</span>
                </>
              )}
            </div>

            <div className="flex items-end justify-center gap-1 h-16">
              {histogram.map((bin) => (
                <div key={bin.x} className="w-7 flex flex-col items-center gap-1">
                  <div
                    className={cn(
                      'w-5 rounded-sm transition-all',
                      bin.x < 0 ? 'bg-blue-500/70' : bin.x > 0 ? 'bg-orange-500/70' : 'bg-gray-500/70',
                    )}
                    style={{ height: `${Math.max(2, (bin.count / maxBin) * 44)}px` }}
                    title={`score ${bin.x}: ${bin.count} votes`}
                  />
                  <span className="text-[10px] text-muted-foreground">{bin.x}</span>
                </div>
              ))}
            </div>

            <div className="relative max-w-sm mx-auto h-5">
              <div className="absolute left-0 right-0 top-2 h-0.5 bg-muted rounded" />
              <div
                className="absolute top-0 h-4 w-0.5 bg-primary"
                style={{ left: `${((consensusDisplayMean + 3) / 6) * 100}%` }}
                title={`Crowd center ${consensusDisplayMean.toFixed(1)}`}
              />
              {selectedValue !== null && (
                <div
                  className="absolute top-0 h-4 w-0.5 bg-foreground"
                  style={{ left: `${((selectedValue + 3) / 6) * 100}%` }}
                  title={`Your answer ${selectedValue}`}
                />
              )}
            </div>
          </div>
        </div>
      )}

      <div className="flex justify-center gap-3">
        <button
          onClick={() => navigate('/pairwise')}
          className="px-5 py-2 rounded-lg border border-border hover:bg-accent"
        >
          {t('pairwise.skipTutorial')}
        </button>
      </div>
    </div>
  )
}
