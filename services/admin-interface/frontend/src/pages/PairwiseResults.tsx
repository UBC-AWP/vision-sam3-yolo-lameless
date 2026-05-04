import { useState, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  BarChart,
  Bar,
  XAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts'
import {
  ArrowLeft,
  ArrowRight,
  Minus,
  Search,
  ChevronUp,
  ChevronDown,
  GitCompare,
  Users,
  TrendingUp,
  CheckCircle2,
  ArrowUpDown,
} from 'lucide-react'
import { getConsensusData, type ConsensusData } from '@/utils/pairwiseConsensus'

import { useLanguage } from '@/contexts/LanguageContext'

// ---------------------------------------------------------------------------
// Types & helpers
// ---------------------------------------------------------------------------

type SortKey = 'agreePercent' | 'absMean' | 'count' | 'pairKey'
type SortDir = 'asc' | 'desc'

function agreementColor(pct: number): string {
  if (pct >= 80) return 'text-emerald-500'
  if (pct >= 65) return 'text-amber-500'
  return 'text-rose-500'
}

function agreementBg(pct: number): string {
  if (pct >= 80) return 'bg-emerald-500/15 border-emerald-500/30'
  if (pct >= 65) return 'bg-amber-500/15 border-amber-500/30'
  return 'bg-rose-500/15 border-rose-500/30'
}

function consensusLabel(data: ConsensusData) {
  if (Math.abs(data.mean) < 0.25) return { text: 'Tied', icon: 'tie', cow: '' }
  if (data.mean > 0) return { text: `Cow ${data.maxCow} more lame`, icon: 'right', cow: data.maxCow }
  return { text: `Cow ${data.minCow} more lame`, icon: 'left', cow: data.minCow }
}

// Build histogram data for score distribution (-3 … +3)
function buildHistogram(scores: number[]) {
  const bins: { score: number; count: number }[] = []
  for (let s = -3; s <= 3; s++) {
    bins.push({ score: s, count: scores.filter(x => x === s).length })
  }
  return bins
}

function barColor(score: number) {
  if (score < 0) return '#3b82f6'   // blue  → left cow more lame
  if (score > 0) return '#f97316'   // orange → right cow more lame
  return '#6b7280'                   // gray  → tied
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function MiniHistogram({ scores }: { scores: number[] }) {
  const data = buildHistogram(scores)
  return (
    <ResponsiveContainer width="100%" height={40}>
      <BarChart data={data} margin={{ top: 2, right: 0, left: 0, bottom: 0 }} barSize={8}>
        <XAxis dataKey="score" tick={{ fontSize: 8 }} tickLine={false} axisLine={false} />
        <Tooltip
          contentStyle={{ fontSize: 11, padding: '4px 8px', borderRadius: 6 }}
          formatter={(val: number) => [`${val}×`, '']}
          labelFormatter={(l: number) => `Score ${l}`}
        />
        <Bar dataKey="count" radius={[2, 2, 0, 0]}>
          {data.map((entry) => (
            <Cell key={entry.score} fill={barColor(entry.score)} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

function SortHeader({
  label,
  sortKey,
  current,
  dir,
  onSort,
}: {
  label: string
  sortKey: SortKey
  current: SortKey
  dir: SortDir
  onSort: (k: SortKey) => void
}) {
  const active = current === sortKey
  return (
    <button
      onClick={() => onSort(sortKey)}
      className="flex items-center gap-1 text-xs font-semibold uppercase tracking-wider text-muted-foreground hover:text-foreground transition-colors group"
    >
      {label}
      {active ? (
        dir === 'desc' ? (
          <ChevronDown className="h-3 w-3 text-primary" />
        ) : (
          <ChevronUp className="h-3 w-3 text-primary" />
        )
      ) : (
        <ArrowUpDown className="h-3 w-3 opacity-30 group-hover:opacity-70" />
      )}
    </button>
  )
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function PairwiseResults() {
  const navigate = useNavigate()
  const { t } = useLanguage()
  const [search, setSearch] = useState('')
  const [sortKey, setSortKey] = useState<SortKey>('absMean')
  const [sortDir, setSortDir] = useState<SortDir>('desc')

  // Load consensus data (cached, computed once)
  const allPairs = useMemo(() => {
    const map = getConsensusData()
    return Array.from(map.values())
  }, [])

  // Summary stats
  const summary = useMemo(() => {
    const totalJudgments = allPairs.reduce((s, p) => s + p.count, 0)
    const directionalPairs = allPairs.filter(p => !p.allEqual)
    const avgAgree = directionalPairs.length
      ? directionalPairs.reduce((s, p) => s + p.agreePercent, 0) / directionalPairs.length
      : 0
    const strongConsensus = directionalPairs.filter(p => p.agreePercent >= 75).length
    const equalPairs = allPairs.filter(p => p.allEqual).length
    return { totalPairs: allPairs.length, totalJudgments, avgAgree, strongConsensus, equalPairs }
  }, [allPairs])

  // Filter + sort
  const displayed = useMemo(() => {
    const q = search.toLowerCase().trim()
    let filtered = q
      ? allPairs.filter(p => p.minCow.includes(q) || p.maxCow.includes(q))
      : allPairs

    filtered = [...filtered].sort((a, b) => {
      let va: number, vb: number
      switch (sortKey) {
        case 'agreePercent': va = a.agreePercent; vb = b.agreePercent; break
        case 'absMean':      va = Math.abs(a.mean); vb = Math.abs(b.mean); break
        case 'count':        va = a.count; vb = b.count; break
        default:             va = parseInt(a.minCow); vb = parseInt(b.minCow)
      }
      return sortDir === 'desc' ? vb - va : va - vb
    })

    return filtered
  }, [allPairs, search, sortKey, sortDir])

  const handleSort = (key: SortKey) => {
    if (key === sortKey) {
      setSortDir(d => d === 'desc' ? 'asc' : 'desc')
    } else {
      setSortKey(key)
      setSortDir('desc')
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <GitCompare className="h-6 w-6 text-primary" />
            {t('pairResults.title')}
          </h2>
          <p className="text-sm text-muted-foreground mt-1">
            {t('pairResults.subtitle')}
          </p>
        </div>
        <button
          onClick={() => navigate('/pairwise')}
          className="px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-medium hover:bg-primary/90 flex items-center gap-2"
        >
          <GitCompare className="h-4 w-4" />
          {t('dashboard.continueComparing')}
        </button>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        {[
          {
            label: t('pairResults.totalPairs'),
            value: summary.totalPairs,
            icon: GitCompare,
            color: 'text-blue-500',
            bg: 'bg-blue-500/10',
          },
          {
            label: t('pairResults.totalJudgments'),
            value: summary.totalJudgments,
            icon: Users,
            color: 'text-purple-500',
            bg: 'bg-purple-500/10',
          },
          {
            label: t('pairResults.avgAgreement'),
            value: `${summary.avgAgree.toFixed(1)}%`,
            icon: TrendingUp,
            color: 'text-amber-500',
            bg: 'bg-amber-500/10',
          },
          {
            label: 'Strong Consensus (≥75%)',
            value: `${summary.strongConsensus} pairs`,
            icon: CheckCircle2,
            color: 'text-emerald-500',
            bg: 'bg-emerald-500/10',
          },
          {
            label: 'All Equal Pairs',
            value: `${summary.equalPairs} pairs`,
            icon: Minus,
            color: 'text-slate-500',
            bg: 'bg-slate-500/10',
          },
        ].map(card => {
          const Icon = card.icon
          return (
            <div key={card.label} className="bg-card border border-border/50 rounded-xl p-4 flex items-center gap-3">
              <div className={`w-10 h-10 rounded-lg ${card.bg} flex items-center justify-center flex-shrink-0`}>
                <Icon className={`h-5 w-5 ${card.color}`} />
              </div>
              <div>
                <div className="text-xl font-bold">{card.value}</div>
                <div className="text-xs text-muted-foreground">{card.label}</div>
              </div>
            </div>
          )
        })}
      </div>

      {/* Search + sort bar */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="relative flex-1 min-w-48 max-w-sm">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <input
            type="text"
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder={t('pairResults.searchPlaceholder')}
            className="w-full pl-9 pr-3 py-2 bg-muted/50 border border-border/50 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
          />
        </div>
        <span className="text-sm text-muted-foreground">
          {t('dashboard.showing')} <span className="font-medium text-foreground">{displayed.length}</span> {t('dashboard.of')} {allPairs.length} {t('pairwise.pairs')}
        </span>
      </div>

      {/* Table */}
      <div className="bg-card border border-border/50 rounded-xl overflow-hidden">
        {/* Table header */}
        <div className="grid grid-cols-[160px_1fr_100px_110px_70px_160px] gap-4 px-4 py-3 border-b border-border/50 bg-muted/30">
          <SortHeader label={t('pairResults.cow')} sortKey="pairKey" current={sortKey} dir={sortDir} onSort={handleSort} />
          <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">{t('pairResults.consensus')}</div>
          <SortHeader label={t('pairResults.agreement')} sortKey="agreePercent" current={sortKey} dir={sortDir} onSort={handleSort} />
          <SortHeader label={t('pairResults.meanScore')} sortKey="absMean" current={sortKey} dir={sortDir} onSort={handleSort} />
          <SortHeader label="N" sortKey="count" current={sortKey} dir={sortDir} onSort={handleSort} />
          <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Distribution</div>
        </div>

        {/* Rows */}
        <div className="divide-y divide-border/30 max-h-[60vh] overflow-y-auto">
          {displayed.length === 0 && (
            <div className="py-12 text-center text-muted-foreground text-sm">
              {t('pairResults.noPairsMatch')}
            </div>
          )}
          {displayed.map(pair => {
            const consensus = consensusLabel(pair)
            const absMean = Math.abs(pair.mean)

            return (
              <div
                key={pair.pairKey}
                className="grid grid-cols-[160px_1fr_100px_110px_70px_160px] gap-4 px-4 py-3 hover:bg-muted/20 transition-colors items-center"
              >
                {/* Pair IDs */}
                <div className="flex items-center gap-1.5 font-mono text-sm">
                  <span className="px-2 py-0.5 bg-blue-500/10 text-blue-600 dark:text-blue-400 rounded font-medium">
                    {pair.minCow}
                  </span>
                  <span className="text-muted-foreground text-xs">vs</span>
                  <span className="px-2 py-0.5 bg-orange-500/10 text-orange-600 dark:text-orange-400 rounded font-medium">
                    {pair.maxCow}
                  </span>
                </div>

                {/* Consensus direction */}
                <div className="flex items-center gap-1.5 min-w-0">
                  {consensus.icon === 'left' && (
                    <ArrowLeft className="h-3.5 w-3.5 text-blue-500 flex-shrink-0" />
                  )}
                  {consensus.icon === 'right' && (
                    <ArrowRight className="h-3.5 w-3.5 text-orange-500 flex-shrink-0" />
                  )}
                  {consensus.icon === 'tie' && (
                    <Minus className="h-3.5 w-3.5 text-muted-foreground flex-shrink-0" />
                  )}
                  <span className={`text-sm truncate ${
                    consensus.icon === 'left' ? 'text-blue-600 dark:text-blue-400' :
                    consensus.icon === 'right' ? 'text-orange-600 dark:text-orange-400' :
                    'text-muted-foreground'
                  }`}>
                    {consensus.text}
                  </span>
                </div>

                {/* Agreement % */}
                <div>
                  {pair.allEqual ? (
                    <div className="inline-flex items-center px-2 py-0.5 rounded-full border text-xs font-bold bg-slate-500/10 text-slate-500 border-slate-500/30">
                      All Equal
                    </div>
                  ) : (
                    <div className={`inline-flex items-center px-2 py-0.5 rounded-full border text-xs font-bold ${agreementBg(pair.agreePercent)} ${agreementColor(pair.agreePercent)}`}>
                      {pair.agreePercent.toFixed(0)}%
                    </div>
                  )}
                </div>

                {/* Mean ± stdev */}
                <div className="text-sm">
                  <span className={`font-semibold ${
                    absMean >= 2 ? 'text-foreground' : 'text-muted-foreground'
                  }`}>
                    {absMean.toFixed(2)}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {' '}±{pair.stdev.toFixed(1)}
                  </span>
                  <div className="mt-0.5 w-full bg-muted rounded-full h-1.5 overflow-hidden">
                    <div
                      className={`h-1.5 rounded-full ${
                        absMean >= 2 ? 'bg-primary' : 'bg-muted-foreground/40'
                      }`}
                      style={{ width: `${(absMean / 3) * 100}%` }}
                    />
                  </div>
                </div>

                {/* Count */}
                <div className="text-sm text-muted-foreground font-medium">
                  {pair.count}
                </div>

                {/* Mini distribution chart */}
                <div className="w-full">
                  <MiniHistogram scores={pair.scores} />
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-6 text-xs text-muted-foreground flex-wrap">
        <div className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm bg-blue-500/70 inline-block" />
          Histogram bars left of 0 = left/min cow more lame
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm bg-orange-500/70 inline-block" />
          Bars right of 0 = right/max cow more lame
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm bg-gray-400/70 inline-block" />
          0 = equal / cannot decide
        </div>
        <div className="flex items-center gap-1.5">
          <span className="font-medium text-emerald-500">≥80%</span>
          <span className="font-medium text-amber-500 ml-2">65–79%</span>
          <span className="font-medium text-rose-500 ml-2">&lt;65%</span>
          agreement tiers (directional pairs only)
        </div>
        <div className="flex items-center gap-1.5">
          <span className="font-medium text-slate-500">All Equal</span>
          = all annotators rated this pair as equally lame (degree=0)
        </div>
      </div>
    </div>
  )
}
