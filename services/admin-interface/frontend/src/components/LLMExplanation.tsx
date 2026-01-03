/**
 * LLM Explanation Component
 * Displays AI-generated explanations for lameness predictions
 */
import { useState, useEffect } from 'react'
import { 
  Brain, 
  AlertTriangle, 
  CheckCircle2, 
  HelpCircle, 
  Lightbulb,
  RefreshCw,
  Sparkles,
  ClipboardList,
  AlertCircle,
  ArrowRight
} from 'lucide-react'
import { cn } from '../lib/utils'

interface ExplanationSection {
  executive_summary: string
  key_evidence: string
  uncertainties: string
  recommended_action: string
}

interface FusionSummary {
  prediction: string
  probability: number
  confidence: number
  decision_mode: string
}

interface LLMExplanationData {
  video_id: string
  status?: string
  message?: string
  explanation: string
  sections: ExplanationSection
  llm_provider: string
  fusion_summary: FusionSummary
  prompt_used?: string
}

interface LLMExplanationProps {
  videoId: string
  className?: string
}

export function LLMExplanation({ videoId, className }: LLMExplanationProps) {
  const [data, setData] = useState<LLMExplanationData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [showPrompt, setShowPrompt] = useState(false)

  const fetchExplanation = async () => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await fetch(`/api/analysis/${videoId}/explanation`)
      if (!response.ok) {
        throw new Error('Failed to fetch explanation')
      }
      const result = await response.json()
      setData(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (videoId) {
      fetchExplanation()
    }
  }, [videoId])

  if (loading) {
    return (
      <div className={cn("border border-border rounded-xl bg-card p-6", className)}>
        <div className="flex items-center gap-3">
          <div className="animate-spin">
            <Brain className="h-5 w-5 text-primary" />
          </div>
          <span className="text-muted-foreground">Loading AI explanation...</span>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className={cn("border border-destructive/30 rounded-xl bg-destructive/10 p-6", className)}>
        <div className="flex items-center gap-3 text-destructive">
          <AlertCircle className="h-5 w-5" />
          <span>Failed to load explanation: {error}</span>
        </div>
        <button
          onClick={fetchExplanation}
          className="mt-3 flex items-center gap-2 text-sm text-primary hover:underline"
        >
          <RefreshCw className="h-4 w-4" />
          Retry
        </button>
      </div>
    )
  }

  if (!data) return null

  const isPending = data.status === 'pending'
  const isLame = data.fusion_summary?.prediction === 'Lame'
  const confidence = data.fusion_summary?.confidence || 0

  return (
    <div className={cn("border border-border rounded-xl bg-card overflow-hidden", className)}>
      {/* Header */}
      <div className={cn(
        "p-4 border-b border-border flex items-center justify-between",
        isLame ? "bg-destructive/10" : "bg-success/10"
      )}>
        <div className="flex items-center gap-3">
          <div className={cn(
            "w-10 h-10 rounded-xl flex items-center justify-center",
            isLame ? "bg-destructive/20" : "bg-success/20"
          )}>
            <Brain className={cn(
              "h-5 w-5",
              isLame ? "text-destructive" : "text-success"
            )} />
          </div>
          <div>
            <h3 className="font-semibold text-foreground flex items-center gap-2">
              AI Explanation
              {data.llm_provider !== 'none' && data.llm_provider !== 'template' && (
                <span className="text-xs px-2 py-0.5 rounded-full bg-primary/20 text-primary">
                  <Sparkles className="h-3 w-3 inline mr-1" />
                  {data.llm_provider}
                </span>
              )}
            </h3>
            <p className="text-sm text-muted-foreground">
              {isPending ? 'Basic summary (full analysis pending)' : 'Generated analysis'}
            </p>
          </div>
        </div>
        
        {/* Prediction Badge */}
        <div className={cn(
          "px-4 py-2 rounded-lg font-semibold text-sm",
          isLame 
            ? "bg-destructive text-destructive-foreground" 
            : "bg-success text-success-foreground"
        )}>
          {data.fusion_summary?.prediction} ({(data.fusion_summary?.probability * 100).toFixed(0)}%)
        </div>
      </div>

      {/* Content */}
      <div className="p-4 space-y-4">
        {/* Executive Summary */}
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-sm font-medium text-foreground">
            <CheckCircle2 className="h-4 w-4 text-primary" />
            Executive Summary
          </div>
          <p className="text-sm text-foreground bg-muted/50 rounded-lg p-3">
            {data.sections?.executive_summary || data.explanation}
          </p>
        </div>

        {/* Key Evidence */}
        {data.sections?.key_evidence && (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm font-medium text-foreground">
              <ClipboardList className="h-4 w-4 text-info" />
              Key Evidence
            </div>
            <div className="text-sm text-muted-foreground bg-muted/30 rounded-lg p-3 whitespace-pre-line">
              {data.sections.key_evidence}
            </div>
          </div>
        )}

        {/* Uncertainties */}
        {data.sections?.uncertainties && (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm font-medium text-foreground">
              <HelpCircle className="h-4 w-4 text-warning" />
              Uncertainties
            </div>
            <div className="text-sm text-muted-foreground bg-warning/10 border border-warning/20 rounded-lg p-3 whitespace-pre-line">
              {data.sections.uncertainties}
            </div>
          </div>
        )}

        {/* Recommended Action */}
        {data.sections?.recommended_action && (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm font-medium text-foreground">
              <Lightbulb className="h-4 w-4 text-success" />
              Recommended Action
            </div>
            <div className={cn(
              "text-sm rounded-lg p-3 border",
              isLame 
                ? "bg-destructive/10 border-destructive/20 text-foreground" 
                : "bg-success/10 border-success/20 text-foreground"
            )}>
              <div className="flex items-start gap-2">
                <ArrowRight className="h-4 w-4 mt-0.5 flex-shrink-0" />
                <span>{data.sections.recommended_action}</span>
              </div>
            </div>
          </div>
        )}

        {/* Decision Mode & Confidence */}
        <div className="flex flex-wrap gap-3 pt-2 border-t border-border">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span className="font-medium">Decision Mode:</span>
            <span className={cn(
              "px-2 py-0.5 rounded-full capitalize",
              data.fusion_summary?.decision_mode === 'human' && "bg-purple-500/20 text-purple-600",
              data.fusion_summary?.decision_mode === 'automated' && "bg-blue-500/20 text-blue-600",
              data.fusion_summary?.decision_mode === 'hybrid' && "bg-amber-500/20 text-amber-600",
              data.fusion_summary?.decision_mode === 'uncertain' && "bg-gray-500/20 text-gray-600"
            )}>
              {data.fusion_summary?.decision_mode || 'unknown'}
            </span>
          </div>
          
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span className="font-medium">Confidence:</span>
            <div className="flex items-center gap-1">
              <div className="w-16 h-2 bg-muted rounded-full overflow-hidden">
                <div 
                  className={cn(
                    "h-full rounded-full",
                    confidence > 0.7 ? "bg-success" : confidence > 0.4 ? "bg-warning" : "bg-destructive"
                  )}
                  style={{ width: `${confidence * 100}%` }}
                />
              </div>
              <span>{(confidence * 100).toFixed(0)}%</span>
            </div>
          </div>
        </div>

        {/* Show Prompt Toggle (for debugging) */}
        {data.prompt_used && (
          <div className="pt-2">
            <button
              onClick={() => setShowPrompt(!showPrompt)}
              className="text-xs text-muted-foreground hover:text-foreground flex items-center gap-1"
            >
              {showPrompt ? 'Hide' : 'Show'} prompt used
            </button>
            {showPrompt && (
              <pre className="mt-2 text-xs bg-muted/50 p-3 rounded-lg overflow-x-auto max-h-60 overflow-y-auto">
                {data.prompt_used}
              </pre>
            )}
          </div>
        )}
      </div>

      {/* Footer - Pending Status */}
      {isPending && (
        <div className="px-4 py-3 bg-muted/50 border-t border-border flex items-center justify-between">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <AlertTriangle className="h-4 w-4 text-warning" />
            Full LLM analysis not yet generated
          </div>
          <button
            onClick={fetchExplanation}
            className="flex items-center gap-2 text-sm text-primary hover:text-primary/80"
          >
            <RefreshCw className="h-4 w-4" />
            Refresh
          </button>
        </div>
      )}
    </div>
  )
}

export default LLMExplanation

