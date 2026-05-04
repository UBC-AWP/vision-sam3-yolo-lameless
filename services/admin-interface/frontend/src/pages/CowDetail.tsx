import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { 
  cowsApi, 
  CowPrediction, 
  LamenessTimelineEntry, 
  CowVideo 
} from '@/api/client'
import LLMExplanation from '@/components/LLMExplanation'
import { getDemoCows } from '@/utils/demoData'
import { getCowRankings, getConsensusData } from '@/utils/pairwiseConsensus'
import { useAuth } from '@/contexts/AuthContext'

interface CowDetails {
  id: string
  cow_id: string
  tag_number?: string | null
  total_sightings: number
  first_seen?: string | null
  last_seen?: string | null
  is_active: boolean
  notes?: string | null
  embedding_version?: string
  video_count: number
  lameness_record_count: number
  current_prediction?: CowPrediction | null
  last_prediction_update?: string | null
}

export default function CowDetail() {
  const { user } = useAuth()
  const useDemo = user?.id === 'guest' || user?.role === 'rater'
  const { cowId } = useParams<{ cowId: string }>()
  const [cow, setCow] = useState<CowDetails | null>(null)
  const [timeline, setTimeline] = useState<LamenessTimelineEntry[]>([])
  const [trend, setTrend] = useState<string>('unknown')
  const [videos, setVideos] = useState<CowVideo[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'timeline' | 'videos' | 'details'>('timeline')
  const [weeksRange, setWeeksRange] = useState(4)
  const [demoExplanation, setDemoExplanation] = useState<any>(null)
  
  // Edit mode
  const [isEditing, setIsEditing] = useState(false)
  const [editTag, setEditTag] = useState('')
  const [editNotes, setEditNotes] = useState('')

  useEffect(() => {
    if (cowId) {
      loadCowData()
    }
  }, [cowId, weeksRange])

  const loadCowData = async () => {
    if (!cowId) return
    
    try {
      setLoading(true)

      // ── Check pairwise-ranked cows first (real CSV data) ──────────────────
      const rankings = getCowRankings()
      const ranking = rankings.find(r => r.cowId === cowId)

      if (ranking) {
        const consensus = getConsensusData()
        // Collect pair-level data involving this cow for a pseudo-timeline
        const now = Date.now()
        const oneDay = 24 * 60 * 60 * 1000
        const pairEntries: any[] = []
        let pairIdx = 0
        for (const [, pair] of consensus.entries()) {
          if (pair.minCow !== cowId && pair.maxCow !== cowId) continue
          // canonical mean: positive = maxCow more lame; flip for minCow
          const rawMean = pair.maxCow === cowId ? pair.mean : -pair.mean
          // Map to 0-1 (raw mean is -3..+3, but in practice smaller)
          const score = Math.max(0, Math.min(1, (rawMean + 3) / 6))
          pairEntries.push({
            id: `pair-${pairIdx++}`,
            video_id: cowId,
            date: new Date(now - pairIdx * 3 * oneDay).toISOString(),
            fusion_score: parseFloat(score.toFixed(3)),
            pipeline_scores: {},
            is_lame: score > 0.5,
            severity_level: score >= 0.75 ? 'severe' : score >= 0.5 ? 'moderate' : score >= 0.25 ? 'mild' : 'healthy',
            human_validated: true,
            human_label: score > 0.5,
            confidence: pair.agreePercent / 100,
          })
        }
        pairEntries.sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())

        const score = ranking.normalizedScore
        setCow({
          id: ranking.cowId,
          cow_id: ranking.cowId,
          tag_number: `#${ranking.cowId}`,
          total_sightings: ranking.comparisons,
          first_seen: null,
          last_seen: null,
          is_active: true,
          notes: `Rank #${ranking.rank} — ${ranking.wins}W / ${ranking.losses}L / ${ranking.ties}T across ${ranking.comparisons} pairwise judgments`,
          embedding_version: 'pairwise-consensus',
          video_count: ranking.videoUrl ? 1 : 0,
          lameness_record_count: pairEntries.length,
          current_prediction: {
            aggregated_score: score,
            is_lame: score > 0.5,
            confidence: 0.9,
            severity_level: ranking.severity,
            observation_date: new Date().toISOString(),
            num_videos: ranking.videoUrl ? 1 : 0,
          } as any,
          last_prediction_update: new Date().toISOString(),
        })
        setTimeline(pairEntries as any)
        setTrend(score > 0.5 ? 'worsening' : 'stable')
        if (ranking.videoUrl) {
          setVideos([{
            video_id: ranking.cowId,
            s3_url: ranking.videoUrl,
            recorded_date: new Date().toISOString(),
            lameness_score: score,
            created_at: new Date().toISOString(),
          } as any])
        } else {
          setVideos([])
        }
        setEditTag(`#${ranking.cowId}`)
        setEditNotes(`Rank #${ranking.rank} — ${ranking.wins}W / ${ranking.losses}L / ${ranking.ties}T`)

        const totalCows = rankings.length
        const winPct = ranking.comparisons > 0
          ? ((ranking.wins / ranking.comparisons) * 100).toFixed(0)
          : '0'
        const severityLabel = ranking.severity.charAt(0).toUpperCase() + ranking.severity.slice(1)
        const isLame = score > 0.5

        setDemoExplanation({
          video_id: ranking.cowId,
          explanation: `Cow ${ranking.cowId} is ranked **#${ranking.rank}** out of ${totalCows} cows based on ${ranking.comparisons} pairwise human judgments. Consensus lameness score: **${(score * 100).toFixed(0)}%** (${severityLabel}).`,
          sections: {
            executive_summary: `Cow **${ranking.cowId}** received a consensus lameness score of **${(score * 100).toFixed(0)}%**, placing it at rank **#${ranking.rank}** out of ${totalCows} cows. It was judged more lame in ${ranking.losses} out of ${ranking.comparisons} comparisons (win rate: ${winPct}%).`,
            key_evidence: `- **${ranking.losses} losses** (judged more lame) vs **${ranking.wins} wins** (judged less lame)\n- **${ranking.ties} ties** (annotators could not decide)\n- Derived from **${ranking.comparisons} individual pairwise judgments** by human annotators\n- Severity category: **${severityLabel}** (normalized score ${(score * 100).toFixed(1)}%)`,
            uncertainties: ranking.ties > 0
              ? `${ranking.ties} comparisons ended in a tie, indicating some ambiguity in judging this cow's lameness relative to others. Scores are relative within this cohort of ${totalCows} cows.`
              : `All comparisons produced a clear winner/loser verdict. Scores are relative within this cohort of ${totalCows} cows.`,
            recommended_action: isLame
              ? `**Veterinary review recommended.** Cow ${ranking.cowId} consistently ranks among the more lame animals in the herd. A physical gait assessment and possible hoof inspection are advised.`
              : `**Continue routine monitoring.** Cow ${ranking.cowId} is among the healthier animals in pairwise comparisons. Maintain regular observation schedule.`,
          },
          llm_provider: 'pairwise-consensus',
          llm_model: 'Human Annotators',
          fusion_summary: {
            prediction: isLame ? 'Lame' : 'Sound',
            probability: score,
            confidence: ranking.wins + ranking.losses > 0
              ? ranking.losses / (ranking.wins + ranking.losses)
              : 0.5,
            decision_mode: 'human',
          },
        })

        setError(null)
        setLoading(false)
        return
      }

      // ── Fallback: legacy demo_cows.csv ─────────────────────────────────────
      const demoCows = getDemoCows()
      const demoCow = demoCows.find(c => c.id === cowId)
      
      if (demoCow) {
        // Load demo data
        const tags = ['#A101', '#B205', '#C330', '#D412', '#E508']
        const now = Date.now()
        const oneDay = 24 * 60 * 60 * 1000
        const daysAgo = Math.floor(Math.random() * 30)
        
        // Calculate score based on severity
        let score = 0
        if (demoCow.severity === 'healthy') score = 0.15
        else if (demoCow.severity === 'mild') score = 0.4
        else if (demoCow.severity === 'moderate') score = 0.65
        else if (demoCow.severity === 'severe') score = 0.85
        
        setCow({
          id: demoCow.id,
          cow_id: demoCow.id,
          tag_number: tags[Math.floor(Math.random() * tags.length)],
          total_sightings: 15,
          first_seen: new Date(now - 60 * oneDay).toISOString(),
          last_seen: new Date(now - daysAgo * oneDay).toISOString(),
          is_active: true,
          notes: 'Demo cow from demo_cows.csv',
          embedding_version: 'dinov3-base',
          video_count: 3,
          lameness_record_count: 3,
          current_prediction: {
            fusion_score: score,
            aggregated_score: score, // Add for display in Lameness Score card
            tleap_score: score + (Math.random() - 0.5) * 0.1,
            tcn_score: score + (Math.random() - 0.5) * 0.1,
            transformer_score: score + (Math.random() - 0.5) * 0.1,
            is_lame: demoCow.severity === 'severe' || demoCow.severity === 'moderate',
            confidence: 0.85,
            severity_level: demoCow.severity,
            observation_date: new Date(now - daysAgo * oneDay).toISOString()
          } as any,
          last_prediction_update: new Date(now - daysAgo * oneDay).toISOString()
        })
        
        // Generate timeline (data points in weekly intervals)
        const numDataPoints = Math.min(weeksRange, 10) // Up to 10 data points
        const timelineEntries: LamenessTimelineEntry[] = Array.from({ length: numDataPoints }, (_, i) => {
          // Generate realistic score progression
          let weekScore = score
          if (demoCow.severity === 'severe' || demoCow.severity === 'moderate') {
            // Gradual worsening for lame cows
            weekScore = score - (i * 0.03) + (Math.random() - 0.5) * 0.1
          } else {
            // Stable/slight improvement for healthy cows
            weekScore = score + (Math.random() - 0.5) * 0.12
          }
          weekScore = Math.max(0, Math.min(1, weekScore)) // Clamp 0-1
          
          return {
            id: `timeline-${i}`,
            video_id: demoCow.id,
            date: new Date(now - (i * 7 * oneDay)).toISOString(), // Weekly intervals
            fusion_score: parseFloat(weekScore.toFixed(3)),
            tleap_score: weekScore + (Math.random() - 0.5) * 0.08,
            tcn_score: weekScore + (Math.random() - 0.5) * 0.08,
            transformer_score: weekScore + (Math.random() - 0.5) * 0.08,
            is_lame: weekScore > 0.5,
            severity_level: weekScore > 0.75 ? 'severe' : weekScore > 0.5 ? 'moderate' : weekScore > 0.3 ? 'mild' : 'healthy',
            human_validated: i === 0 || i === 3, // Mark some as validated
            confidence: 0.75 + Math.random() * 0.2
          } as any
        }).reverse() // Oldest to newest for chart
        setTimeline(timelineEntries)
        setTrend(Math.random() > 0.5 ? 'stable' : 'improving')
        
        // Generate demo videos
        setVideos([{
          video_id: demoCow.id,
          filename: `cow_${demoCow.id}_demo.mp4`,
          s3_url: demoCow.videoUrl,
          recorded_date: new Date(now - daysAgo * oneDay).toISOString(),
          lameness_score: score,
          created_at: new Date(now - daysAgo * oneDay).toISOString()
        } as any])
        
        setEditTag(tags[Math.floor(Math.random() * tags.length)])
        setEditNotes('Demo cow from demo_cows.csv')

        const severityLabel = demoCow.severity.charAt(0).toUpperCase() + demoCow.severity.slice(1)
        const isLame = demoCow.severity === 'severe' || demoCow.severity === 'moderate'
        setDemoExplanation({
          video_id: demoCow.id,
          explanation: `Cow ${demoCow.id} has a **${severityLabel}** lameness severity based on expert annotation.`,
          sections: {
            executive_summary: `Cow **${demoCow.id}** has been assessed with **${severityLabel}** severity lameness. The lameness score is **${(score * 100).toFixed(0)}%**.`,
            key_evidence: `- Severity category: **${severityLabel}**\n- Lameness score: **${(score * 100).toFixed(0)}%**\n- Annotation source: expert-labeled demo dataset`,
            uncertainties: `This is demo data from a labeled dataset. Scores represent expert annotations rather than real-time AI inference.`,
            recommended_action: isLame
              ? `**Veterinary review recommended.** Cow ${demoCow.id} shows ${demoCow.severity} lameness. A physical gait assessment and possible hoof inspection are advised.`
              : `**Continue routine monitoring.** Cow ${demoCow.id} shows ${demoCow.severity} lameness severity. Maintain regular observation schedule.`,
          },
          llm_provider: 'demo-annotation',
          llm_model: 'Expert Annotator',
          fusion_summary: {
            prediction: isLame ? 'Lame' : 'Sound',
            probability: score,
            confidence: 0.85,
            decision_mode: 'annotation',
          },
        })

        setError(null)
        setLoading(false)
        return
      }
      
      const [cowData, lamenessData, videosData] = await Promise.all([
        cowsApi.get(cowId),
        cowsApi.getLameness(cowId, weeksRange * 7), // Convert weeks to days for API
        cowsApi.getVideos(cowId, { limit: 50 })
      ])
      
      setCow(cowData)
      setTimeline(lamenessData.timeline)
      setTrend(lamenessData.trend)
      setVideos(videosData.videos)
      setEditTag(cowData.tag_number || '')
      setEditNotes(cowData.notes || '')
      setError(null)
    } catch (err: any) {
      console.error('Failed to load cow data:', err)
      setError(err.response?.data?.detail || 'Failed to load cow data')
    } finally {
      setLoading(false)
    }
  }

  const handleSaveEdit = async () => {
    if (!cowId) return
    
    try {
      await cowsApi.update(cowId, {
        tag_number: editTag || null,
        notes: editNotes || null
      })
      await loadCowData()
      setIsEditing(false)
    } catch (err: any) {
      console.error('Failed to update cow:', err)
    }
  }

  const getSeverityColor = (severity: string | null | undefined): string => {
    switch (severity) {
      case 'healthy': return 'bg-success/20 text-success border-success/30'
      case 'mild': return 'bg-warning/20 text-warning border-warning/30'
      case 'moderate': return 'bg-orange-500/20 text-orange-500 border-orange-500/30'
      case 'severe': return 'bg-destructive/20 text-destructive border-destructive/30'
      default: return 'bg-muted text-muted-foreground border-border'
    }
  }

  const getTrendInfo = (trend: string): { icon: string; color: string; text: string } => {
    switch (trend) {
      case 'improving': return { icon: '📈', color: 'text-success', text: 'Improving' }
      case 'worsening': return { icon: '📉', color: 'text-destructive', text: 'Worsening' }
      case 'stable': return { icon: '➡️', color: 'text-primary', text: 'Stable' }
      default: return { icon: '❓', color: 'text-muted-foreground', text: 'Unknown' }
    }
  }

  const formatDate = (dateStr: string | null | undefined): string => {
    if (!dateStr) return 'Never'
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const formatShortDate = (dateStr: string | null | undefined): string => {
    if (!dateStr) return '—'
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric'
    })
  }

  if (loading && !cow) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <div className="text-muted-foreground">Loading cow details...</div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="text-center py-12">
        <div className="text-4xl mb-4">❌</div>
        <h4 className="text-lg font-semibold mb-2">Error Loading Cow</h4>
        <p className="text-muted-foreground mb-4">{error}</p>
        <Link to="/cows" className="text-primary hover:underline">
          ← Back to Cow Registry
        </Link>
      </div>
    )
  }

  if (!cow) {
    return (
      <div className="text-center py-12">
        <div className="text-4xl mb-4">🐮</div>
        <h4 className="text-lg font-semibold mb-2">Cow Not Found</h4>
        <Link to="/cows" className="text-primary hover:underline">
          ← Back to Cow Registry
        </Link>
      </div>
    )
  }

  const prediction = cow.current_prediction
  const trendInfo = getTrendInfo(trend)

  return (
    <div className="space-y-6">
      {/* Breadcrumb */}
      <div className="text-sm text-muted-foreground">
        <Link to="/cows" className="hover:text-primary">Cow Registry</Link>
        <span className="mx-2">/</span>
        <span>{cow.cow_id.slice(0, 8)}...</span>
      </div>

      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <div className="flex items-center gap-3">
            <h2 className="text-3xl font-bold tracking-tight">
              {cow.tag_number ? `🏷️ ${cow.tag_number}` : `🐮 ${cow.cow_id.slice(0, 12)}`}
            </h2>
            {cow.is_active ? (
              <span className="px-3 py-1 text-sm bg-success/20 text-success rounded-full">
                Active
              </span>
            ) : (
              <span className="px-3 py-1 text-sm bg-muted text-muted-foreground rounded-full">
                Inactive
              </span>
            )}
          </div>
          <p className="text-muted-foreground mt-1 font-mono text-sm">
            ID: {cow.cow_id}
          </p>
        </div>
        
        {!useDemo && (
          <button
            onClick={() => setIsEditing(!isEditing)}
            className="px-4 py-2 border rounded-lg hover:bg-accent transition-colors"
          >
            {isEditing ? 'Cancel' : '✏️ Edit'}
          </button>
        )}
      </div>

      {/* Edit Form */}
      {isEditing && !useDemo && (
        <div className="border border-border rounded-lg p-6 bg-muted/50">
          <h3 className="text-lg font-semibold mb-4">Edit Cow Details</h3>
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1">Tag Number</label>
              <input
                type="text"
                value={editTag}
                onChange={(e) => setEditTag(e.target.value)}
                placeholder="e.g., 1234"
                className="w-full px-4 py-2 border border-border rounded-lg bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Notes</label>
              <input
                type="text"
                value={editNotes}
                onChange={(e) => setEditNotes(e.target.value)}
                placeholder="Any notes about this cow..."
                className="w-full px-4 py-2 border border-border rounded-lg bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
              />
            </div>
          </div>
          <div className="flex gap-2 mt-4">
            <button
              onClick={handleSaveEdit}
              className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90"
            >
              Save Changes
            </button>
            <button
              onClick={() => setIsEditing(false)}
              className="px-4 py-2 border rounded-lg hover:bg-accent"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {/* Current Severity */}
        <div className="border border-border rounded-lg p-6 bg-card">
          <p className="text-sm text-muted-foreground mb-2">Current Status</p>
          <div className={`inline-flex items-center gap-2 px-3 py-2 rounded-lg text-lg font-medium border ${
            getSeverityColor(prediction?.severity_level)
          }`}>
            <span className="text-2xl">
              {prediction?.severity_level === 'healthy' ? '🐄' :
               prediction?.severity_level === 'mild' ? '🟡' :
               prediction?.severity_level === 'moderate' ? '🟠' :
               prediction?.severity_level === 'severe' ? '🔴' : '❓'}
            </span>
            <span className="capitalize">{prediction?.severity_level || 'Unknown'}</span>
          </div>
        </div>

        {/* Health Score */}
        <div className="border border-border rounded-lg p-6 bg-card">
          <p className="text-sm text-muted-foreground mb-2">{useDemo ? 'Health Score' : 'Lameness Score'}</p>
          {prediction?.aggregated_score !== undefined ? (
            <>
              <p className="text-3xl font-bold">
                {(prediction.aggregated_score * 100).toFixed(0)}%
              </p>
              <div className="w-full bg-muted rounded-full h-2 mt-2">
                <div
                  className={`h-2 rounded-full transition-all ${
                    prediction.aggregated_score < 0.3 ? 'bg-success' :
                    prediction.aggregated_score < 0.5 ? 'bg-warning' :
                    prediction.aggregated_score < 0.7 ? 'bg-orange-500' : 'bg-destructive'
                  }`}
                  style={{ width: `${prediction.aggregated_score * 100}%` }}
                />
              </div>
            </>
          ) : (
            <p className="text-3xl font-bold text-muted-foreground">—</p>
          )}
        </div>

        {/* Trend */}
        <div className="border border-border rounded-lg p-6 bg-card">
          <p className="text-sm text-muted-foreground mb-2">Trend ({weeksRange} weeks)</p>
          <div className={`flex items-center gap-2 text-xl font-medium ${trendInfo.color}`}>
            <span className="text-2xl">{trendInfo.icon}</span>
            <span>{trendInfo.text}</span>
          </div>
        </div>

        {/* Videos — only shown to admin/researcher */}
        {!useDemo && (
          <div className="border border-border rounded-lg p-6 bg-card">
            <p className="text-sm text-muted-foreground mb-2">Total Videos</p>
            <p className="text-3xl font-bold">{cow.video_count}</p>
            <p className="text-sm text-muted-foreground mt-1">
              {cow.lameness_record_count} records
            </p>
          </div>
        )}
      </div>

      {/* Dates Info — only shown to admin/researcher */}
      {!useDemo && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div className="border border-border rounded-lg p-4 bg-card">
            <span className="text-muted-foreground">First Seen:</span>
            <span className="ml-2 font-medium">{formatShortDate(cow.first_seen)}</span>
          </div>
          <div className="border border-border rounded-lg p-4 bg-card">
            <span className="text-muted-foreground">Last Seen:</span>
            <span className="ml-2 font-medium">{formatShortDate(cow.last_seen)}</span>
          </div>
          <div className="border border-border rounded-lg p-4 bg-card">
            <span className="text-muted-foreground">Total Sightings:</span>
            <span className="ml-2 font-medium">{cow.total_sightings}</span>
          </div>
          <div className="border border-border rounded-lg p-4 bg-card">
            <span className="text-muted-foreground">Confidence:</span>
            <span className="ml-2 font-medium">
              {prediction?.confidence ? `${(prediction.confidence * 100).toFixed(0)}%` : '—'}
            </span>
          </div>
        </div>
      )}

      {/* Latest AI Explanation / Pairwise Summary */}
      {(videos.length > 0 || demoExplanation) && (
        <div className="space-y-2">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            {demoExplanation ? '📊 Pairwise Consensus Summary' : '🤖 Latest AI Analysis'}
            <span className="text-sm font-normal text-muted-foreground">
              {demoExplanation ? 'derived from human annotations' : 'from most recent video'}
            </span>
          </h3>
          <LLMExplanation
            videoId={videos[0]?.video_id ?? cowId ?? ''}
            overrideData={demoExplanation ?? undefined}
          />
        </div>
      )}

      {/* Tabs — hidden for rater/public view */}
      {!useDemo && <>
      <div className="border-b">
        <div className="flex gap-4">
          {['timeline', 'videos', 'details'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab as typeof activeTab)}
              className={`px-4 py-3 font-medium border-b-2 transition-colors capitalize ${
                activeTab === tab
                  ? 'border-primary text-primary'
                  : 'border-transparent text-muted-foreground hover:text-foreground'
              }`}
            >
              {tab === 'timeline' && '📊 '}
              {tab === 'videos' && '📹 '}
              {tab === 'details' && 'ℹ️ '}
              {tab}
            </button>
          ))}
        </div>
      </div>

      {/* Tab Content */}
      {activeTab === 'timeline' && (
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <h3 className="text-lg font-semibold">Lameness Timeline</h3>
            <select
              value={weeksRange}
              onChange={(e) => setWeeksRange(Number(e.target.value))}
              className="px-3 py-1.5 border border-border rounded-lg text-sm bg-card text-foreground"
            >
              <option value={4}>Last 4 weeks</option>
              <option value={10}>Last 10 weeks</option>
              <option value={26}>Last 6 months</option>
              <option value={52}>Last year</option>
            </select>
          </div>

          {/* Timeline Chart (Simplified bar chart) */}
          {timeline.length > 0 && (
            <div className="border border-border rounded-lg p-6 bg-card">
              <div className="flex items-end gap-1 h-32 mb-4">
                {timeline.slice(0, 30).map((entry, idx) => {
                  const score = entry.fusion_score ?? 0.5
                  return (
                    <div
                      key={entry.id || idx}
                      className="flex-1 min-w-[8px] transition-all hover:opacity-80 cursor-pointer relative group"
                    >
                      <div
                        className={`w-full rounded-t ${
                          score < 0.3 ? 'bg-success' :
                          score < 0.5 ? 'bg-warning' :
                          score < 0.7 ? 'bg-orange-500' : 'bg-destructive'
                        } ${entry.human_validated ? 'ring-2 ring-primary' : ''}`}
                        style={{ height: `${score * 100}%` }}
                      />
                      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block bg-popover text-popover-foreground text-xs rounded px-2 py-1 whitespace-nowrap z-10 shadow-lg border border-border">
                        {formatShortDate(entry.date)}: {(score * 100).toFixed(0)}%
                        {entry.human_validated && ' ✓'}
                      </div>
                    </div>
                  )
                })}
              </div>
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>{formatShortDate(timeline[timeline.length - 1]?.date)}</span>
                <span>{formatShortDate(timeline[0]?.date)}</span>
              </div>
            </div>
          )}

          {/* Timeline Table */}
          {timeline.length === 0 ? (
            <div className="text-center py-8 border border-border rounded-lg bg-muted/50">
              <p className="text-muted-foreground">No lameness records in this period</p>
            </div>
          ) : (
            <div className="border border-border rounded-lg overflow-hidden bg-card">
              <table className="w-full">
                <thead className="bg-muted/50 border-b border-border">
                  <tr>
                    <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Date</th>
                    {!useDemo && <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Video</th>}
                    <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">{useDemo ? 'Health Score' : 'Score'}</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Status</th>
                    {!useDemo && <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Validated</th>}
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {timeline.slice(0, 20).map((entry) => (
                    <tr key={entry.id} className="hover:bg-accent/50">
                      <td className="py-3 px-4 text-sm">
                        {formatDate(entry.date)}
                      </td>
                      {!useDemo && (
                        <td className="py-3 px-4">
                          <Link 
                            to={`/results/${entry.video_id}`}
                            className="font-mono text-sm text-primary hover:underline"
                          >
                            {entry.video_id.slice(0, 8)}...
                          </Link>
                        </td>
                      )}
                      <td className="py-3 px-4">
                        <div className="flex items-center gap-2">
                          <div className="w-12 bg-muted rounded-full h-2">
                            <div
                              className={`h-2 rounded-full ${
                                (entry.fusion_score ?? 0) < 0.3 ? 'bg-success' :
                                (entry.fusion_score ?? 0) < 0.5 ? 'bg-warning' :
                                (entry.fusion_score ?? 0) < 0.7 ? 'bg-orange-500' : 'bg-destructive'
                              }`}
                              style={{ width: `${(entry.fusion_score ?? 0) * 100}%` }}
                            />
                          </div>
                          <span className="text-sm font-mono">
                            {entry.fusion_score !== null ? `${(entry.fusion_score * 100).toFixed(0)}%` : '—'}
                          </span>
                        </div>
                      </td>
                      <td className="py-3 px-4">
                        <span className={`inline-flex px-2 py-1 rounded text-xs font-medium border ${
                          getSeverityColor(entry.severity_level)
                        }`}>
                          {useDemo
                            ? (entry.severity_level === 'healthy' ? '✅ Healthy' :
                               entry.severity_level === 'mild' ? '🟡 Mild' :
                               entry.severity_level === 'moderate' ? '🟠 Moderate' :
                               entry.severity_level === 'severe' ? '🔴 Lame' :
                               'Unknown')
                            : (entry.severity_level || 'Unknown')}
                        </span>
                      </td>
                      {!useDemo && (
                        <td className="py-3 px-4">
                          {entry.human_validated ? (
                            <span className="text-success">
                              ✓ {entry.human_label ? 'Lame' : 'Sound'}
                            </span>
                          ) : (
                            <span className="text-muted-foreground text-sm">Pending</span>
                          )}
                        </td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {activeTab === 'videos' && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold">Video History</h3>
          
          {videos.length === 0 ? (
            <div className="text-center py-8 border border-border rounded-lg bg-muted/50">
              <p className="text-muted-foreground">No videos found for this cow</p>
            </div>
          ) : (
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
              {videos.map((video) => (
                <div key={video.video_id} className="border border-border rounded-lg p-4 bg-card hover:shadow-md transition-shadow">
                  <div className="flex justify-between items-start mb-2">
                    <Link 
                      to={`/results/${video.video_id}`}
                      className="font-mono text-sm text-primary hover:underline"
                    >
                      {video.video_id.slice(0, 12)}...
                    </Link>
                    {video.lameness_score !== null && video.lameness_score !== undefined && (
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        video.lameness_score < 0.3 ? 'bg-success/20 text-success' :
                        video.lameness_score < 0.5 ? 'bg-warning/20 text-warning' :
                        video.lameness_score < 0.7 ? 'bg-orange-500/20 text-orange-500' :
                        'bg-destructive/20 text-destructive'
                      }`}>
                        {(video.lameness_score * 100).toFixed(0)}%
                      </span>
                    )}
                  </div>
                  <div className="text-sm text-muted-foreground space-y-1">
                    <p>Track #{video.track_id}</p>
                    {video.total_frames && <p>{video.total_frames} frames</p>}
                    {video.reid_confidence && (
                      <p>Re-ID: {(video.reid_confidence * 100).toFixed(0)}%</p>
                    )}
                    <p className="text-xs">{formatShortDate(video.created_at)}</p>
                  </div>
                  <Link
                    to={`/video/${video.video_id}`}
                    className="block mt-3 text-center py-2 border border-border rounded hover:bg-accent transition-colors text-sm"
                  >
                    View Video →
                  </Link>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {activeTab === 'details' && (
        <div className="space-y-6">
          <h3 className="text-lg font-semibold">Technical Details</h3>
          
          <div className="grid md:grid-cols-2 gap-6">
            {/* Identity Info */}
            <div className="border border-border rounded-lg p-6 bg-card">
              <h4 className="font-medium mb-4">Identity Information</h4>
              <dl className="space-y-3">
                <div className="flex justify-between">
                  <dt className="text-muted-foreground">Internal ID</dt>
                  <dd className="font-mono text-sm">{cow.id}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-muted-foreground">Cow ID (Re-ID)</dt>
                  <dd className="font-mono text-sm">{cow.cow_id}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-muted-foreground">Tag Number</dt>
                  <dd>{cow.tag_number || '—'}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-muted-foreground">Embedding Version</dt>
                  <dd className="font-mono text-sm">{cow.embedding_version || 'dinov3-base'}</dd>
                </div>
              </dl>
            </div>

            {/* Statistics */}
            <div className="border border-border rounded-lg p-6 bg-card">
              <h4 className="font-medium mb-4">Statistics</h4>
              <dl className="space-y-3">
                <div className="flex justify-between">
                  <dt className="text-muted-foreground">Total Sightings</dt>
                  <dd className="font-bold">{cow.total_sightings}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-muted-foreground">Videos Analyzed</dt>
                  <dd className="font-bold">{cow.video_count}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-muted-foreground">Lameness Records</dt>
                  <dd className="font-bold">{cow.lameness_record_count}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-muted-foreground">Status</dt>
                  <dd>{cow.is_active ? '🟢 Active' : '⚪ Inactive'}</dd>
                </div>
              </dl>
            </div>

            {/* Pipeline Scores (if available) */}
            {prediction && (
              <div className="border border-border rounded-lg p-6 bg-card md:col-span-2">
                <h4 className="font-medium mb-4">Current Prediction Details</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center p-3 bg-muted/50 rounded">
                    <p className="text-2xl font-bold">
                      {(prediction.aggregated_score * 100).toFixed(0)}%
                    </p>
                    <p className="text-sm text-muted-foreground">Aggregated Score</p>
                  </div>
                  <div className="text-center p-3 bg-muted/50 rounded">
                    <p className="text-2xl font-bold">
                      {(prediction.confidence * 100).toFixed(0)}%
                    </p>
                    <p className="text-sm text-muted-foreground">Confidence</p>
                  </div>
                  <div className="text-center p-3 bg-muted/50 rounded">
                    <p className="text-2xl font-bold">{prediction.num_videos}</p>
                    <p className="text-sm text-muted-foreground">Videos Used</p>
                  </div>
                  <div className="text-center p-3 bg-muted/50 rounded">
                    <p className="text-2xl font-bold capitalize">{prediction.severity_level}</p>
                    <p className="text-sm text-muted-foreground">Severity</p>
                  </div>
                </div>
              </div>
            )}

            {/* Notes */}
            {cow.notes && (
              <div className="border border-border rounded-lg p-6 bg-card md:col-span-2">
                <h4 className="font-medium mb-2">Notes</h4>
                <p className="text-muted-foreground">{cow.notes}</p>
              </div>
            )}
          </div>
        </div>
      )}
      </>}
    </div>
  )
}

