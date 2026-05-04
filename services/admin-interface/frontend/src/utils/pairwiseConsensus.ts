import rawCSV from '../demo_data/winner_loser_sampled_exchange0_55HITs.csv?raw'

// Eagerly import all local cow videos – Vite resolves these at build time
const cowVideoFilesMP4 = import.meta.glob<string>(
  '../demo_data/compressed_cow/*.MP4',
  { eager: true, query: '?url', import: 'default' }
)
const cowVideoFilesmp4 = import.meta.glob<string>(
  '../demo_data/compressed_cow/*.mp4',
  { eager: true, query: '?url', import: 'default' }
)
const cowVideoFiles: Record<string, string> = { ...cowVideoFilesMP4, ...cowVideoFilesmp4 }

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ConsensusData {
  pairKey: string
  minCow: string
  maxCow: string
  /** Canonical signed scores: +degree means maxCow more lame, -degree means minCow more lame */
  scores: number[]
  mean: number
  median: number
  stdev: number
  /** % of responses whose direction matches the majority */
  agreePercent: number
  count: number
  /** True when every annotator rated this pair as degree=0 (equally lame) */
  allEqual: boolean
}

export interface FeedbackResult {
  message: string
  emoji: string
  type: 'great' | 'good' | 'interesting' | 'unique'
  details: string
  /** Which side the consensus favors (relative to left/right display) */
  consensusFavor: 'left' | 'right' | 'tied'
  consensusMean: number
  userCanonical: number
  agreePercent: number
}

export interface DemoConsensusPair {
  cow_L: string
  cow_R: string
  cow_L_URL: string
  cow_R_URL: string
  pairKey: string
  consensusData: ConsensusData
}

// ---------------------------------------------------------------------------
// Build cow-ID → video URL map
// ---------------------------------------------------------------------------

function buildCowVideoMap(): Map<string, string> {
  const map = new Map<string, string>()
  for (const [filePath, url] of Object.entries(cowVideoFiles)) {
    // filename pattern: compressed_..._<clip>_<cowId>.<ext>
    const match = filePath.match(/_(\d{4})\.[Mm][Pp]4$/)
    if (match) {
      const cowId = match[1]
      if (!map.has(cowId)) {
        map.set(cowId, url)
      }
    }
  }
  return map
}

// ---------------------------------------------------------------------------
// Parse CSV and compute consensus per canonical pair
// ---------------------------------------------------------------------------

function parseConsensus(): Map<string, ConsensusData> {
  const lines = rawCSV.trim().split('\n')
  const pairScores = new Map<string, number[]>()

  for (const line of lines.slice(1)) {
    const parts = line.trim().split(',')
    if (parts.length < 3) continue

    const winner = parts[0].trim()
    const loser = parts[1].trim()
    const degree = parseInt(parts[2].trim(), 10)
    if (isNaN(degree) || !winner || !loser) continue

    const wNum = parseInt(winner, 10)
    const lNum = parseInt(loser, 10)
    const minNum = Math.min(wNum, lNum)
    const maxNum = Math.max(wNum, lNum)
    const minCow = String(minNum)
    const maxCow = String(maxNum)
    const key = `${minCow}_${maxCow}`

    // Canonical score convention:
    //   positive → maxCow is more lame
    //   negative → minCow is more lame
    // In this dataset, `winner` means "more lame cow".
    const canonicalScore = winner === minCow ? -degree : +degree

    const existing = pairScores.get(key)
    if (existing) {
      existing.push(canonicalScore)
    } else {
      pairScores.set(key, [canonicalScore])
    }
  }

  const consensus = new Map<string, ConsensusData>()

  for (const [key, scores] of pairScores.entries()) {
    const [minCow, maxCow] = key.split('_')
    const n = scores.length
    const mean = scores.reduce((a, b) => a + b, 0) / n

    const sorted = [...scores].sort((a, b) => a - b)
    const mid = Math.floor(sorted.length / 2)
    const median =
      sorted.length % 2 === 0
        ? (sorted[mid - 1] + sorted[mid]) / 2
        : sorted[mid]

    const stdev = Math.sqrt(
      scores.reduce((sum, s) => sum + Math.pow(s - mean, 2), 0) / n
    )

    // Agreement: what fraction match the majority direction?
    const majoritySign = Math.sign(mean)
    const agreedCount =
      majoritySign === 0
        ? scores.filter(s => s === 0).length
        : scores.filter(s => Math.sign(s) === majoritySign).length
    const agreePercent = (agreedCount / n) * 100

    consensus.set(key, {
      pairKey: key,
      minCow,
      maxCow,
      scores,
      mean,
      median,
      stdev,
      agreePercent,
      count: n,
      allEqual: scores.every(s => s === 0),
    })
  }

  return consensus
}

// ---------------------------------------------------------------------------
// Cow ranking
// ---------------------------------------------------------------------------

export interface CowRanking {
  cowId: string
  /** 1 = most lame */
  rank: number
  /** Average Elo rating across 100 random orderings (higher = more lame) */
  rawScore: number
  /** 0–1 normalized across all cows; 1 = most lame */
  normalizedScore: number
  severity: 'healthy' | 'mild' | 'moderate' | 'severe'
  /** Total individual judgments this cow participated in */
  comparisons: number
  /** Times judged as the more-lame cow */
  wins: number
  /** Times judged as the less-lame / healthier cow */
  losses: number
  /** Times judged equal (degree 0) */
  ties: number
  videoUrl?: string
}

// ---------------------------------------------------------------------------
// Elo rating (mirrors elo_rating_randomize_lame_rank.R)
// ---------------------------------------------------------------------------

/**
 * Parse raw CSV judgments into a flat list, replicating each row by its
 * degree (degree=2 → 2 entries, degree=3 → 3 entries), matching
 * replicate_row_df() in the R script.  degree=0 → 1 draw entry.
 */
function parseRawJudgments(): Array<{ winner: string; loser: string; isDraw: boolean }> {
  const lines = rawCSV.trim().split('\n')
  const result: Array<{ winner: string; loser: string; isDraw: boolean }> = []

  for (const line of lines.slice(1)) {
    const parts = line.trim().split(',')
    if (parts.length < 3) continue
    const winner = parts[0].trim()
    const loser  = parts[1].trim()
    const degree = parseInt(parts[2].trim(), 10)
    if (isNaN(degree) || !winner || !loser) continue

    const isDraw = degree === 0
    const reps   = isDraw ? 1 : degree   // degree=1→1×, degree=2→2×, degree=3→3×
    for (let r = 0; r < reps; r++) {
      result.push({ winner, loser, isDraw })
    }
  }
  return result
}

/**
 * Run Elo rating over `judgments` in a shuffled order.
 * winner = more-lame cow (gains Elo), loser = healthier cow (loses Elo).
 * k=20 matches the R script default.
 */
function runEloIteration(
  judgments: Array<{ winner: string; loser: string; isDraw: boolean }>,
  cows: Set<string>,
  k = 20,
): Map<string, number> {
  const elo = new Map<string, number>()
  for (const cow of cows) elo.set(cow, 1500)

  // Fisher-Yates shuffle
  const shuffled = [...judgments]
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]]
  }

  for (const { winner, loser, isDraw } of shuffled) {
    const rW = elo.get(winner)!
    const rL = elo.get(loser)!
    const eW = 1 / (1 + Math.pow(10, (rL - rW) / 400))
    const eL = 1 - eW
    if (isDraw) {
      elo.set(winner, rW + k * (0.5 - eW))
      elo.set(loser,  rL + k * (0.5 - eL))
    } else {
      elo.set(winner, rW + k * (1   - eW))
      elo.set(loser,  rL + k * (0   - eL))
    }
  }
  return elo
}

/**
 * Compute averaged Elo ratings across `numIterations` random orderings.
 * Mirrors the 100-iteration averaging in the R script that removes order bias.
 */
function computeAverageElo(numIterations = 100): Map<string, number> {
  const judgments = parseRawJudgments()

  const cows = new Set<string>()
  for (const { winner, loser } of judgments) {
    cows.add(winner)
    cows.add(loser)
  }

  const sums = new Map<string, number>()
  for (const cow of cows) sums.set(cow, 0)

  for (let i = 0; i < numIterations; i++) {
    const iter = runEloIteration(judgments, cows)
    for (const [cow, rating] of iter) {
      sums.set(cow, sums.get(cow)! + rating)
    }
  }

  const avg = new Map<string, number>()
  for (const [cow, total] of sums) {
    avg.set(cow, total / numIterations)
  }
  return avg
}

// ---------------------------------------------------------------------------
// Cow ranking
// ---------------------------------------------------------------------------

/**
 * Derive a global lameness ranking using Elo rating, matching the methodology
 * in elo_rating_randomize_lame_rank.R:
 *  - Each row is replicated by its degree (higher certainty = more evidence)
 *  - degree=0 → draw (both cows equally lame)
 *  - 100 random orderings are averaged to remove order bias (k=20)
 *  - Winner = more-lame cow → higher Elo
 *  - normalizedScore: 0=healthiest, 1=most lame
 */
export function getCowRankings(): CowRanking[] {
  if (_cowRankings) return _cowRankings
  _cowRankings = _computeCowRankings()
  return _cowRankings
}

function _computeCowRankings(): CowRanking[] {
  const videoMap = getCowVideoMap()
  const avgElo   = computeAverageElo(100)

  // Per-cow wins/losses/ties from raw CSV (one entry per original row)
  const stats = new Map<string, { wins: number; losses: number; ties: number; comparisons: number }>()
  const lines = rawCSV.trim().split('\n')
  for (const line of lines.slice(1)) {
    const parts = line.trim().split(',')
    if (parts.length < 3) continue
    const winner = parts[0].trim()
    const loser  = parts[1].trim()
    const degree = parseInt(parts[2].trim(), 10)
    if (isNaN(degree) || !winner || !loser) continue

    for (const cow of [winner, loser]) {
      if (!stats.has(cow)) stats.set(cow, { wins: 0, losses: 0, ties: 0, comparisons: 0 })
    }
    const ws = stats.get(winner)!
    const ls = stats.get(loser)!
    if (degree === 0) {
      ws.ties++; ws.comparisons++
      ls.ties++; ls.comparisons++
    } else {
      ws.wins++;   ws.comparisons++
      ls.losses++; ls.comparisons++
    }
  }

  // Sort by Elo descending → most lame first (highest Elo = most-lame)
  const sorted = [...avgElo.entries()].sort((a, b) => b[1] - a[1])

  const maxElo = sorted[0]?.[1] ?? 1
  const minElo = sorted[sorted.length - 1]?.[1] ?? 0
  const range  = maxElo - minElo || 1

  return sorted.map(([cowId, eloRating], idx) => {
    // normalizedScore: 1 = most lame (highest Elo), 0 = healthiest (lowest Elo)
    const n = (eloRating - minElo) / range

    let severity: CowRanking['severity']
    if (n >= 0.75) severity = 'severe'
    else if (n >= 0.50) severity = 'moderate'
    else if (n >= 0.25) severity = 'mild'
    else severity = 'healthy'

    const s = stats.get(cowId) ?? { wins: 0, losses: 0, ties: 0, comparisons: 0 }

    return {
      cowId,
      rank: idx + 1,
      rawScore: eloRating,
      normalizedScore: n,
      severity,
      comparisons: s.comparisons,
      wins:   s.wins,
      losses: s.losses,
      ties:   s.ties,
      videoUrl: videoMap.get(cowId),
    }
  })
}

// ---------------------------------------------------------------------------
// Module-level caches (initialised lazily)
// ---------------------------------------------------------------------------

let _cowVideoMap: Map<string, string> | null = null
let _consensusData: Map<string, ConsensusData> | null = null
let _demoPairs: DemoConsensusPair[] | null = null
let _cowRankings: CowRanking[] | null = null

export function getCowVideoMap(): Map<string, string> {
  if (!_cowVideoMap) _cowVideoMap = buildCowVideoMap()
  return _cowVideoMap
}

export function getConsensusData(): Map<string, ConsensusData> {
  if (!_consensusData) _consensusData = parseConsensus()
  return _consensusData
}

/**
 * Returns all valid demo pairs: both cows have a local video AND real human
 * comparison data exists in the CSV.  Pairs are returned in a stable order.
 */
export function getValidDemoPairs(): DemoConsensusPair[] {
  if (_demoPairs) return _demoPairs

  const videoMap = getCowVideoMap()
  const consensus = getConsensusData()

  _demoPairs = []
  for (const [key, data] of consensus.entries()) {
    const cow_L_URL = videoMap.get(data.minCow)
    const cow_R_URL = videoMap.get(data.maxCow)
    if (!cow_L_URL || !cow_R_URL) continue

    _demoPairs.push({
      cow_L: data.minCow,
      cow_R: data.maxCow,
      cow_L_URL,
      cow_R_URL,
      pairKey: key,
      consensusData: data,
    })
  }

  return _demoPairs
}

// ---------------------------------------------------------------------------
// Feedback computation
// ---------------------------------------------------------------------------

/**
 * Compare a user's submitted answer against the crowd consensus for a pair.
 *
 * @param leftCowId  - The cow ID shown on the LEFT side
 * @param rightCowId - The cow ID shown on the RIGHT side
 * @param userValue  - The value on the 7-point scale:
 *                     negative (-1 to -3) = LEFT cow more lame
 *                     0 = equal / cannot decide
 *                     positive (1 to 3)   = RIGHT cow more lame
 */
export function computeFeedback(
  leftCowId: string,
  rightCowId: string,
  userValue: number
): FeedbackResult | null {
  const consensus = getConsensusData()

  const lNum = parseInt(leftCowId, 10)
  const rNum = parseInt(rightCowId, 10)
  const minCow = String(Math.min(lNum, rNum))
  const maxCow = String(Math.max(lNum, rNum))
  const key = `${minCow}_${maxCow}`

  const data = consensus.get(key)
  if (!data) return null

  // Convert userValue → canonical score
  // canonical positive = maxCow is more lame
  // if leftCow is minCow: user positive (right more lame = maxCow) → same sign
  // if leftCow is maxCow: user positive (right more lame = minCow) → flip sign
  const userCanonical = leftCowId === minCow ? userValue : -userValue

  const distance = Math.abs(userCanonical - data.mean)
  const directionMatch =
    userCanonical === 0 ||
    data.mean === 0 ||
    Math.sign(userCanonical) === Math.sign(data.mean)

  // Which side does the consensus say is more lame? (relative to display order)
  let consensusFavor: FeedbackResult['consensusFavor']
  if (Math.abs(data.mean) < 0.3) {
    consensusFavor = 'tied'
  } else if (leftCowId === minCow) {
    consensusFavor = data.mean > 0 ? 'right' : 'left'
  } else {
    consensusFavor = data.mean > 0 ? 'left' : 'right'
  }

  // Determine feedback tier
  let type: FeedbackResult['type']
  let message: string
  let emoji: string

  if (directionMatch && distance <= 1.0) {
    type = 'great'
    emoji = '🎯'
    message = "You're very close to what everyone else believes!"
  } else if (directionMatch && distance <= 2.0) {
    type = 'good'
    emoji = '👍'
    message = 'Your take aligns with the majority on this pair!'
  } else if (directionMatch && distance <= 3.5) {
    type = 'interesting'
    emoji = '🤔'
    message = 'You agree on which cow is lamer, but see the degree a bit differently.'
  } else if (!directionMatch && distance <= 2.5) {
    type = 'interesting'
    emoji = '🤔'
    message = 'Interesting perspective! Many raters actually saw this the other way.'
  } else {
    type = 'unique'
    emoji = '😮'
    message = 'Oops... this is quite a unique point of view!'
  }

  // Build human-readable details
  const agreePct = Math.round(data.agreePercent)
  const avgStrength = Math.abs(data.mean).toFixed(1)
  const favorStr =
    consensusFavor === 'tied'
      ? 'Raters were split – no clear consensus direction.'
      : `${agreePct}% of raters thought the ${consensusFavor} cow was more lame (avg. strength ${avgStrength}/3).`

  const details = favorStr

  return {
    message,
    emoji,
    type,
    details,
    consensusFavor,
    consensusMean: data.mean,
    userCanonical,
    agreePercent: data.agreePercent,
  }
}
