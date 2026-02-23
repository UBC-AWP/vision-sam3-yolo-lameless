import { useEffect, useState } from 'react'
import { shapApi } from '@/api/client'

interface ShapValue {
  feature: string
  value: number
  shap_value: number
  contribution: number
}

interface ShapExplanationProps {
  videoId: string
  compact?: boolean
}

// Feature descriptions for tooltips
const FEATURE_DESCRIPTIONS: Record<string, string> = {
  stride_length: 'Length of stride during walking - shorter strides may indicate lameness',
  stride_regularity: 'Consistency of stride pattern - irregular strides suggest gait problems',
  back_arch: 'Curvature of the back - arched back is a common lameness indicator',
  head_bob: 'Up/down movement of head while walking - excessive bobbing indicates pain',
  limb_asymmetry: 'Difference in movement between left and right limbs',
  yolo_confidence_mean: 'Average detection confidence from YOLO model',
  yolo_detection_count: 'Number of cow detections across frames',
  dinov3_embedding_norm: 'Magnitude of visual embedding features',
  dinov3_similarity_score: 'Similarity to known patterns in the database',
  fusion_probability: 'Combined prediction from all pipelines',
}

export default function ShapExplanation({ videoId, compact = false }: ShapExplanationProps) {
  const [shapData, setShapData] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const loadShapData = async () => {
      try {
        const data = await shapApi.getForcePlot(videoId)
        setShapData(data)
      } catch (err: any) {
        setError(err.response?.data?.detail || 'Failed to load SHAP explanation')
      } finally {
        setLoading(false)
      }
    }
    loadShapData()
  }, [videoId])

  if (loading) {
    return (
      <div className="p-4 text-center text-muted-foreground">
        Loading explanation...
      </div>
    )
  }

  if (error || !shapData) {
    return (
      <div className="p-4 text-center text-muted-foreground">
        {error || 'No explanation available'}
      </div>
    )
  }

  const { base_value, prediction, features } = shapData
  const shapValues: ShapValue[] = features || []
  
  // Sort by absolute contribution
  const sortedFeatures = [...shapValues].sort((a, b) => 
    Math.abs(b.shap_value) - Math.abs(a.shap_value)
  )

  // Separate positive and negative contributions
  const positiveFeatures = sortedFeatures.filter(f => f.shap_value > 0)
  const negativeFeatures = sortedFeatures.filter(f => f.shap_value < 0)

  // Calculate cumulative values for force plot
  const calculateCumulativeValues = () => {
    let cumulative = base_value
    const segments = []
    
    // Add positive contributions first (pushing toward lame)
    for (const feat of positiveFeatures) {
      segments.push({
        ...feat,
        start: cumulative,
        end: cumulative + feat.shap_value,
        direction: 'positive'
      })
      cumulative += feat.shap_value
    }
    
    // Then negative contributions (pushing toward sound)
    for (const feat of negativeFeatures) {
      segments.push({
        ...feat,
        start: cumulative,
        end: cumulative + feat.shap_value,
        direction: 'negative'
      })
      cumulative += feat.shap_value
    }
    
    return segments
  }

  const segments = calculateCumulativeValues()
  const maxValue = Math.max(prediction, 1 - base_value + 0.1)
  const minValue = Math.min(0, base_value - 0.5)
  const range = maxValue - minValue

  if (compact) {
    // Compact view - just top features
    return (
      <div className="space-y-2">
        <div className="flex justify-between items-center">
          <span className="text-sm font-medium">Why this prediction?</span>
          <span className={`text-sm font-bold ${prediction > 0.5 ? 'text-destructive' : 'text-success'}`}>
            {(prediction * 100).toFixed(0)}% {prediction > 0.5 ? 'Lame' : 'Sound'}
          </span>
        </div>
        <div className="space-y-1">
          {sortedFeatures.slice(0, 3).map((feat, idx) => (
            <div key={idx} className="flex items-center gap-2 text-xs">
              <span className={`w-2 h-2 rounded-full ${feat.shap_value > 0 ? 'bg-destructive' : 'bg-success'}`} />
              <span className="flex-1 truncate" title={FEATURE_DESCRIPTIONS[feat.feature]}>
                {feat.feature.replace(/_/g, ' ')}
              </span>
              <span className={feat.shap_value > 0 ? 'text-destructive' : 'text-success'}>
                {feat.shap_value > 0 ? '+' : ''}{(feat.shap_value * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold">SHAP Explanation</h3>
        <div className="text-right">
          <div className="text-sm text-muted-foreground">Prediction</div>
          <div className={`text-2xl font-bold ${prediction > 0.5 ? 'text-destructive' : 'text-success'}`}>
            {(prediction * 100).toFixed(1)}% {prediction > 0.5 ? 'Lame' : 'Sound'}
          </div>
        </div>
      </div>

      {/* Force Plot Visualization */}
      <div className="bg-muted/50 rounded-lg p-4">
        <div className="text-sm text-muted-foreground mb-2">Force Plot</div>
        
        {/* Scale */}
        <div className="relative h-8 mb-2">
          <div className="absolute inset-x-0 top-1/2 h-1 bg-muted-foreground/30 rounded" />
          
          {/* Base value marker */}
          <div 
            className="absolute top-0 w-0.5 h-full bg-muted-foreground/60"
            style={{ left: `${((base_value - minValue) / range) * 100}%` }}
          >
            <div className="absolute -top-6 left-1/2 -translate-x-1/2 text-xs text-muted-foreground whitespace-nowrap">
              Base: {(base_value * 100).toFixed(0)}%
            </div>
          </div>
          
          {/* Prediction marker */}
          <div 
            className="absolute top-0 w-1 h-full bg-foreground rounded"
            style={{ left: `${((prediction - minValue) / range) * 100}%` }}
          >
            <div className="absolute -bottom-6 left-1/2 -translate-x-1/2 text-xs font-bold whitespace-nowrap">
              {(prediction * 100).toFixed(0)}%
            </div>
          </div>
        </div>

        {/* Labels */}
        <div className="flex justify-between text-xs text-muted-foreground mt-8">
          <span className="text-success">← Sound</span>
          <span className="text-destructive">Lame →</span>
        </div>
      </div>

      {/* Feature Contributions */}
      <div className="space-y-4">
        <div className="text-sm font-medium">Feature Contributions</div>
        
        {/* Pushing toward Lame (positive) */}
        {positiveFeatures.length > 0 && (
          <div className="space-y-2">
            <div className="text-xs text-destructive font-medium flex items-center gap-2">
              <span className="w-3 h-3 bg-destructive rounded" />
              Pushing toward Lame
            </div>
            {positiveFeatures.map((feat, idx) => (
              <FeatureBar
                key={idx}
                feature={feat}
                maxContribution={Math.max(...shapValues.map(f => Math.abs(f.shap_value)))}
                direction="positive"
              />
            ))}
          </div>
        )}

        {/* Pushing toward Sound (negative) */}
        {negativeFeatures.length > 0 && (
          <div className="space-y-2">
            <div className="text-xs text-success font-medium flex items-center gap-2">
              <span className="w-3 h-3 bg-success rounded" />
              Pushing toward Sound
            </div>
            {negativeFeatures.map((feat, idx) => (
              <FeatureBar
                key={idx}
                feature={feat}
                maxContribution={Math.max(...shapValues.map(f => Math.abs(f.shap_value)))}
                direction="negative"
              />
            ))}
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="text-xs text-muted-foreground border-t pt-4">
        <p>
          SHAP values show how each feature contributes to the prediction.
          Positive values (red) push the prediction toward "Lame", 
          while negative values (green) push toward "Sound".
        </p>
      </div>
    </div>
  )
}

interface FeatureBarProps {
  feature: ShapValue
  maxContribution: number
  direction: 'positive' | 'negative'
}

function FeatureBar({ feature, maxContribution, direction }: FeatureBarProps) {
  const percentage = Math.abs(feature.shap_value) / maxContribution * 100
  const description = FEATURE_DESCRIPTIONS[feature.feature]
  
  return (
    <div className="group relative">
      <div className="flex items-center gap-2">
        <div className="w-32 text-xs truncate" title={description}>
          {feature.feature.replace(/_/g, ' ')}
        </div>
        <div className="flex-1 h-4 bg-muted rounded overflow-hidden">
          <div
            className={`h-full rounded transition-all ${
              direction === 'positive' ? 'bg-destructive/70' : 'bg-success/70'
            }`}
            style={{ width: `${percentage}%` }}
          />
        </div>
        <div className={`w-16 text-xs text-right font-medium ${
          direction === 'positive' ? 'text-destructive' : 'text-success'
        }`}>
          {direction === 'positive' ? '+' : ''}{(feature.shap_value * 100).toFixed(1)}%
        </div>
      </div>
      
      {/* Tooltip */}
      {description && (
        <div className="absolute left-0 bottom-full mb-1 hidden group-hover:block z-10">
          <div className="bg-popover text-popover-foreground text-xs rounded px-2 py-1 max-w-xs shadow-lg border border-border">
            <div className="font-medium mb-1">{feature.feature.replace(/_/g, ' ')}</div>
            <div className="text-muted-foreground">{description}</div>
            <div className="mt-1">Value: {feature.value.toFixed(3)}</div>
          </div>
        </div>
      )}
    </div>
  )
}

