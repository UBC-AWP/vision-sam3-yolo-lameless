import { useEffect, useState } from 'react'
import { useLanguage } from '@/contexts/LanguageContext'

export default function Performance() {
  const { t } = useLanguage()
  const [stats, setStats] = useState({
    totalRatings: 47,
    alignment: 85.4,  // Combined accuracy and consistency metric
    meanDeviation: 0.8,
    standardDeviation: 1.2,
  })

  const percentileOutperformed = 77 // User outperformed 77% of other raters

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-3xl font-bold">{t('performance.title')}</h2>
        <p className="text-muted-foreground mt-1">
          {t('performance.subtitle')}
        </p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-card border border-border rounded-lg p-6">
          <div className="text-sm text-muted-foreground mb-1">{t('performance.totalCompleted')}</div>
          <div className="text-4xl font-bold mt-2">{stats.totalRatings}</div>
          <p className="text-xs text-muted-foreground mt-2">{t('performance.totalCompletedDesc')}</p>
        </div>
        <div className="bg-card border border-border rounded-lg p-6">
          <div className="text-sm text-muted-foreground mb-1">{t('performance.agreementScore')}</div>
          <div className="text-4xl font-bold mt-2 text-success">{stats.alignment}%</div>
          <p className="text-xs text-muted-foreground mt-2">{t('performance.agreementScoreDesc')}</p>
        </div>
      </div>

      {/* Performance Ranking */}
      <div className="bg-card border border-border rounded-lg p-8">
        <div className="text-center mb-6">
          <div className="text-6xl font-bold text-success mb-3">{percentileOutperformed}%</div>
          <p className="text-xl font-medium mb-2">
            {t('performance.topPercent')} <span className="text-success">{100 - percentileOutperformed}%</span>
          </p>
          <p className="text-muted-foreground">
            {t('performance.rateMoreAccurately')} {percentileOutperformed} {t('performance.outOf100')}
          </p>
        </div>
        
        {/* Simple visual ranking bar */}
        <div className="relative w-full h-16 mt-8 bg-muted rounded-full overflow-hidden">
          {/* Progress fill */}
          <div 
            className="absolute top-0 left-0 h-full bg-gradient-to-r from-success/40 to-success/60 rounded-full"
            style={{ width: `${percentileOutperformed}%` }}
          />
          
          {/* User position marker */}
          <div 
            className="absolute top-1/2 -translate-y-1/2 w-4 h-4 bg-success rounded-full border-4 border-white shadow-lg"
            style={{ left: `${percentileOutperformed}%`, transform: 'translate(-50%, -50%)' }}
          />
          
          {/* Labels */}
          <div className="absolute top-1/2 left-4 -translate-y-1/2 text-sm font-medium text-muted-foreground">
            0%
          </div>
          <div className="absolute top-1/2 right-4 -translate-y-1/2 text-sm font-medium text-muted-foreground">
            100%
          </div>
          <div 
            className="absolute -top-8 text-sm font-bold text-success"
            style={{ left: `${percentileOutperformed}%`, transform: 'translateX(-50%)' }}
          >
            {t('performance.youAreHere')}
          </div>
        </div>
      </div>

      {/* What This Means */}
      <div className="bg-card border border-border rounded-lg p-6">
        <h3 className="text-xl font-semibold mb-4">{t('performance.whatThisMeans')}</h3>
        <div className="space-y-4">
          <div className="flex items-start gap-4 p-4 bg-success/10 border border-success/30 rounded-lg">
            <div className="text-3xl">✓</div>
            <div>
              <div className="font-semibold text-success mb-1">{t('performance.doingGreat')}</div>
              <div className="text-sm text-muted-foreground">
                {t('performance.doingGreatDesc')}
              </div>
            </div>
          </div>
          <div className="flex items-start gap-4 p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
            <div className="text-3xl">🎯</div>
            <div>
              <div className="font-semibold text-blue-500 mb-1">{t('performance.highAgreement')}</div>
              <div className="text-sm text-muted-foreground">
                {t('performance.highAgreementDesc1')}{stats.alignment}{t('performance.highAgreementDesc2')}
              </div>
            </div>
          </div>
          <div className="flex items-start gap-4 p-4 bg-muted/50 border border-border rounded-lg">
            <div className="text-3xl">📈</div>
            <div>
              <div className="font-semibold mb-1">{t('performance.keepRating')}</div>
              <div className="text-sm text-muted-foreground">
                {t('performance.keepRatingDesc')}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
