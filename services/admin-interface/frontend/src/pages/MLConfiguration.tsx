import { useEffect, useState, useCallback } from 'react'
import { mlConfigApi, trainingApi } from '@/api/client'

interface ParamDescription {
  name: string
  description: string
  category: string
  default: number | string
  range?: [number, number]
  options?: string[]
}

interface ModelConfig {
  [key: string]: number | string | boolean
}

type TabType = 'catboost' | 'xgboost' | 'lightgbm' | 'ensemble' | 'training'

export default function MLConfiguration() {
  const [activeTab, setActiveTab] = useState<TabType>('catboost')
  const [config, setConfig] = useState<any>(null)
  const [descriptions, setDescriptions] = useState<any>(null)
  const [modelsStatus, setModelsStatus] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)
  const [hasChanges, setHasChanges] = useState(false)

  const loadData = useCallback(async () => {
    try {
      const [configData, descriptionsData, statusData] = await Promise.all([
        mlConfigApi.getConfig(),
        mlConfigApi.getParameterDescriptions(),
        mlConfigApi.getModelsStatus(),
      ])
      setConfig(configData.config)
      setDescriptions(descriptionsData)
      setModelsStatus(statusData)
    } catch (error) {
      console.error('Failed to load ML configuration:', error)
      setMessage({ type: 'error', text: 'Failed to load configuration' })
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadData()
  }, [loadData])

  const handleParamChange = (tab: TabType, param: string, value: number | string | boolean) => {
    setConfig((prev: any) => ({
      ...prev,
      [tab]: {
        ...prev[tab],
        [param]: value,
      },
    }))
    setHasChanges(true)
  }

  const handleSave = async (tab: TabType) => {
    setSaving(true)
    setMessage(null)
    try {
      switch (tab) {
        case 'catboost':
          await mlConfigApi.updateCatBoostConfig(config.catboost)
          break
        case 'xgboost':
          await mlConfigApi.updateXGBoostConfig(config.xgboost)
          break
        case 'lightgbm':
          await mlConfigApi.updateLightGBMConfig(config.lightgbm)
          break
        case 'ensemble':
          await mlConfigApi.updateEnsembleConfig(config.ensemble)
          break
        case 'training':
          await mlConfigApi.updateTrainingConfig(config.training)
          break
      }
      setMessage({ type: 'success', text: `${tab.charAt(0).toUpperCase() + tab.slice(1)} configuration saved!` })
      setHasChanges(false)
    } catch (error: any) {
      setMessage({ type: 'error', text: error.response?.data?.detail || 'Failed to save configuration' })
    } finally {
      setSaving(false)
    }
  }

  const handleReset = async () => {
    if (!window.confirm('Reset all configurations to defaults?')) return
    setSaving(true)
    try {
      const result = await mlConfigApi.resetToDefaults()
      setConfig(result.config)
      setMessage({ type: 'success', text: 'Configuration reset to defaults' })
      setHasChanges(false)
    } catch (error: any) {
      setMessage({ type: 'error', text: 'Failed to reset configuration' })
    } finally {
      setSaving(false)
    }
  }

  const handleStartTraining = async () => {
    if (!window.confirm('Start ML model training with current configuration?')) return
    try {
      await trainingApi.startMLTraining()
      setMessage({ type: 'success', text: 'Training started! Check training page for progress.' })
    } catch (error: any) {
      setMessage({ type: 'error', text: error.response?.data?.detail || 'Failed to start training' })
    }
  }

  const renderParamInput = (
    tab: TabType,
    param: string,
    value: number | string | boolean,
    desc: ParamDescription
  ) => {
    if (desc.options) {
      return (
        <select
          value={String(value)}
          onChange={(e) => handleParamChange(tab, param, e.target.value)}
          className="w-full px-3 py-2 border rounded-lg bg-background focus:ring-2 focus:ring-primary focus:outline-none"
        >
          {desc.options.map((opt) => (
            <option key={opt} value={opt}>
              {opt}
            </option>
          ))}
        </select>
      )
    }

    if (typeof value === 'boolean') {
      return (
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={value}
            onChange={(e) => handleParamChange(tab, param, e.target.checked)}
            className="w-5 h-5 rounded border-gray-300 text-primary focus:ring-primary"
          />
          <span className="text-sm">{value ? 'Enabled' : 'Disabled'}</span>
        </label>
      )
    }

    if (typeof value === 'number' && desc.range) {
      const [min, max] = desc.range
      const step = max - min > 10 ? 1 : 0.01
      return (
        <div className="space-y-2">
          <input
            type="range"
            min={min}
            max={max}
            step={step}
            value={value}
            onChange={(e) => handleParamChange(tab, param, parseFloat(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-primary"
          />
          <div className="flex justify-between items-center">
            <input
              type="number"
              min={min}
              max={max}
              step={step}
              value={value}
              onChange={(e) => handleParamChange(tab, param, parseFloat(e.target.value) || min)}
              className="w-24 px-2 py-1 border rounded text-sm"
            />
            <span className="text-xs text-muted-foreground">
              Range: {min} - {max}
            </span>
          </div>
        </div>
      )
    }

    return (
      <input
        type="number"
        value={value as number}
        onChange={(e) => handleParamChange(tab, param, parseFloat(e.target.value) || 0)}
        className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-primary focus:outline-none"
      />
    )
  }

  const renderParamCard = (tab: TabType, param: string, value: number | string | boolean) => {
    const desc = descriptions?.[tab]?.[param]
    if (!desc) return null

    return (
      <div key={param} className="border rounded-lg p-4 hover:border-primary/50 transition-colors">
        <div className="flex justify-between items-start mb-2">
          <div>
            <h4 className="font-medium">{desc.name}</h4>
            <span className="text-xs px-2 py-0.5 bg-gray-100 text-gray-600 rounded">
              {desc.category}
            </span>
          </div>
          <span className="text-xs text-muted-foreground">
            Default: {String(desc.default)}
          </span>
        </div>
        <p className="text-sm text-muted-foreground mb-3">{desc.description}</p>
        {renderParamInput(tab, param, value, desc)}
      </div>
    )
  }

  const renderModelTab = (tab: TabType) => {
    if (!config || !config[tab]) return null

    const tabConfig = config[tab] as ModelConfig
    const params = Object.entries(tabConfig).filter(([key]) => key !== 'random_seed' && key !== 'random_state')
    const seedParam = Object.entries(tabConfig).find(([key]) => key === 'random_seed' || key === 'random_state')

    // Group by category
    const categories: Record<string, Array<[string, number | string | boolean]>> = {}
    params.forEach(([key, value]) => {
      const desc = descriptions?.[tab]?.[key]
      const category = desc?.category || 'Other'
      if (!categories[category]) categories[category] = []
      categories[category].push([key, value])
    })

    return (
      <div className="space-y-6">
        {Object.entries(categories).map(([category, categoryParams]) => (
          <div key={category}>
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              {category === 'Training' && <span>🎯</span>}
              {category === 'Tree Structure' && <span>🌳</span>}
              {category === 'Regularization' && <span>🛡️</span>}
              {category === 'Model' && <span>🔧</span>}
              {category === 'Ensemble' && <span>⚖️</span>}
              {category}
            </h3>
            <div className="grid md:grid-cols-2 gap-4">
              {categoryParams.map(([key, value]) => renderParamCard(tab, key, value))}
            </div>
          </div>
        ))}

        {seedParam && (
          <div className="border-t pt-4">
            <h3 className="text-sm font-medium text-muted-foreground mb-2">Reproducibility</h3>
            {renderParamCard(tab, seedParam[0], seedParam[1])}
          </div>
        )}
      </div>
    )
  }

  const renderEnsembleTab = () => {
    if (!config?.ensemble) return null

    const { catboost_weight, xgboost_weight, lightgbm_weight, voting_method, threshold } = config.ensemble
    const totalWeight = catboost_weight + xgboost_weight + lightgbm_weight

    return (
      <div className="space-y-6">
        {/* Weight Distribution */}
        <div className="border rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Model Weight Distribution</h3>
          <p className="text-sm text-muted-foreground mb-4">
            Adjust how much each model contributes to the final ensemble prediction.
            Weights are automatically normalized.
          </p>

          {/* Visual weight bar */}
          <div className="h-8 flex rounded-lg overflow-hidden mb-4">
            <div
              className="bg-blue-500 flex items-center justify-center text-white text-xs font-medium"
              style={{ width: `${(catboost_weight / totalWeight) * 100}%` }}
            >
              CatBoost
            </div>
            <div
              className="bg-green-500 flex items-center justify-center text-white text-xs font-medium"
              style={{ width: `${(xgboost_weight / totalWeight) * 100}%` }}
            >
              XGBoost
            </div>
            <div
              className="bg-purple-500 flex items-center justify-center text-white text-xs font-medium"
              style={{ width: `${(lightgbm_weight / totalWeight) * 100}%` }}
            >
              LightGBM
            </div>
          </div>

          <div className="grid md:grid-cols-3 gap-4">
            {/* CatBoost weight */}
            <div className="space-y-2">
              <label className="text-sm font-medium flex items-center gap-2">
                <div className="w-3 h-3 bg-blue-500 rounded" />
                CatBoost Weight
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={catboost_weight}
                onChange={(e) => handleParamChange('ensemble', 'catboost_weight', parseFloat(e.target.value))}
                className="w-full accent-blue-500"
              />
              <div className="text-center font-mono text-sm">
                {(catboost_weight * 100).toFixed(0)}%
              </div>
            </div>

            {/* XGBoost weight */}
            <div className="space-y-2">
              <label className="text-sm font-medium flex items-center gap-2">
                <div className="w-3 h-3 bg-green-500 rounded" />
                XGBoost Weight
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={xgboost_weight}
                onChange={(e) => handleParamChange('ensemble', 'xgboost_weight', parseFloat(e.target.value))}
                className="w-full accent-green-500"
              />
              <div className="text-center font-mono text-sm">
                {(xgboost_weight * 100).toFixed(0)}%
              </div>
            </div>

            {/* LightGBM weight */}
            <div className="space-y-2">
              <label className="text-sm font-medium flex items-center gap-2">
                <div className="w-3 h-3 bg-purple-500 rounded" />
                LightGBM Weight
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={lightgbm_weight}
                onChange={(e) => handleParamChange('ensemble', 'lightgbm_weight', parseFloat(e.target.value))}
                className="w-full accent-purple-500"
              />
              <div className="text-center font-mono text-sm">
                {(lightgbm_weight * 100).toFixed(0)}%
              </div>
            </div>
          </div>
        </div>

        {/* Voting Method */}
        <div className="border rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-2">Voting Method</h3>
          <p className="text-sm text-muted-foreground mb-4">
            Choose how model predictions are combined.
          </p>
          <div className="flex gap-4">
            <label className={`flex-1 p-4 border rounded-lg cursor-pointer transition-colors ${
              voting_method === 'soft' ? 'border-primary bg-primary/5' : 'hover:border-gray-400'
            }`}>
              <input
                type="radio"
                name="voting_method"
                value="soft"
                checked={voting_method === 'soft'}
                onChange={() => handleParamChange('ensemble', 'voting_method', 'soft')}
                className="sr-only"
              />
              <div className="font-medium mb-1">Soft Voting</div>
              <div className="text-sm text-muted-foreground">
                Weighted average of predicted probabilities. Best for calibrated models.
              </div>
            </label>
            <label className={`flex-1 p-4 border rounded-lg cursor-pointer transition-colors ${
              voting_method === 'hard' ? 'border-primary bg-primary/5' : 'hover:border-gray-400'
            }`}>
              <input
                type="radio"
                name="voting_method"
                value="hard"
                checked={voting_method === 'hard'}
                onChange={() => handleParamChange('ensemble', 'voting_method', 'hard')}
                className="sr-only"
              />
              <div className="font-medium mb-1">Hard Voting</div>
              <div className="text-sm text-muted-foreground">
                Majority voting of predicted classes. More robust to outliers.
              </div>
            </label>
          </div>
        </div>

        {/* Classification Threshold */}
        <div className="border rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-2">Classification Threshold</h3>
          <p className="text-sm text-muted-foreground mb-4">
            Probability threshold for classifying as "Lame". Lower = more sensitive (catches more),
            Higher = more specific (fewer false positives).
          </p>
          <div className="space-y-4">
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={threshold}
              onChange={(e) => handleParamChange('ensemble', 'threshold', parseFloat(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-sm">
              <span className="text-green-600">More Sensitive</span>
              <span className="font-mono font-medium">{threshold}</span>
              <span className="text-blue-600">More Specific</span>
            </div>
            <div className="h-2 bg-gradient-to-r from-green-500 via-yellow-500 to-blue-500 rounded-full relative">
              <div
                className="absolute w-4 h-4 bg-white border-2 border-gray-800 rounded-full -top-1"
                style={{ left: `calc(${threshold * 100}% - 8px)` }}
              />
            </div>
          </div>
        </div>
      </div>
    )
  }

  const renderTrainingTab = () => {
    if (!config?.training) return null

    return (
      <div className="space-y-6">
        <div className="grid md:grid-cols-2 gap-4">
          {Object.entries(config.training).map(([key, value]) =>
            renderParamCard('training', key, value as number | string | boolean)
          )}
        </div>
      </div>
    )
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4" />
          <div className="text-muted-foreground">Loading ML configuration...</div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <h2 className="text-3xl font-bold">ML Configuration</h2>
          <p className="text-muted-foreground mt-1">
            Configure CatBoost, XGBoost, LightGBM parameters and ensemble settings
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={handleStartTraining}
            className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 font-medium"
          >
            Start Training
          </button>
          <button
            onClick={handleReset}
            disabled={saving}
            className="px-4 py-2 border rounded-lg hover:bg-gray-50 disabled:opacity-50"
          >
            Reset to Defaults
          </button>
        </div>
      </div>

      {/* Message */}
      {message && (
        <div className={`p-4 rounded-lg ${
          message.type === 'success' ? 'bg-green-50 text-green-800 border border-green-200' : 'bg-red-50 text-red-800 border border-red-200'
        }`}>
          {message.text}
        </div>
      )}

      {/* Model Status Cards */}
      {modelsStatus && (
        <div className="grid grid-cols-4 gap-4">
          {['catboost', 'xgboost', 'lightgbm', 'ensemble'].map((model) => {
            const status = modelsStatus.models?.[model]
            return (
              <div key={model} className="border rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <div className={`w-3 h-3 rounded-full ${status?.trained ? 'bg-green-500' : 'bg-gray-300'}`} />
                  <span className="font-medium capitalize">{model}</span>
                </div>
                <div className="text-sm text-muted-foreground">
                  {status?.trained ? (
                    <>
                      {status.size && <span>{(status.size / 1024).toFixed(1)} KB</span>}
                      {status.weights && (
                        <div className="text-xs mt-1">
                          Weights: {Object.entries(status.weights).map(([k, v]) =>
                            `${k}: ${((v as number) * 100).toFixed(0)}%`
                          ).join(', ')}
                        </div>
                      )}
                    </>
                  ) : (
                    'Not trained'
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Tabs */}
      <div className="border-b">
        <div className="flex gap-1">
          {[
            { id: 'catboost', label: 'CatBoost', icon: '🐱' },
            { id: 'xgboost', label: 'XGBoost', icon: '⚡' },
            { id: 'lightgbm', label: 'LightGBM', icon: '💡' },
            { id: 'ensemble', label: 'Ensemble', icon: '🎯' },
            { id: 'training', label: 'Training', icon: '🏋️' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as TabType)}
              className={`px-4 py-3 font-medium text-sm border-b-2 transition-colors ${
                activeTab === tab.id
                  ? 'border-primary text-primary'
                  : 'border-transparent text-muted-foreground hover:text-foreground'
              }`}
            >
              <span className="mr-2">{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Tab Content */}
      <div className="min-h-[400px]">
        {activeTab === 'catboost' && renderModelTab('catboost')}
        {activeTab === 'xgboost' && renderModelTab('xgboost')}
        {activeTab === 'lightgbm' && renderModelTab('lightgbm')}
        {activeTab === 'ensemble' && renderEnsembleTab()}
        {activeTab === 'training' && renderTrainingTab()}
      </div>

      {/* Save Button */}
      <div className="flex justify-end gap-3 pt-4 border-t">
        {hasChanges && (
          <span className="text-sm text-amber-600 self-center">Unsaved changes</span>
        )}
        <button
          onClick={() => handleSave(activeTab)}
          disabled={saving || !hasChanges}
          className="px-6 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50 font-medium"
        >
          {saving ? 'Saving...' : `Save ${activeTab.charAt(0).toUpperCase() + activeTab.slice(1)} Config`}
        </button>
      </div>

      {/* Training Status */}
      {modelsStatus?.training_status && (
        <div className="border rounded-lg p-6 bg-gray-50">
          <h3 className="font-semibold mb-3">Last Training</h3>
          <div className="grid grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-muted-foreground">Status:</span>
              <span className={`ml-2 font-medium ${
                modelsStatus.training_status.status === 'completed' ? 'text-green-600' :
                modelsStatus.training_status.status === 'training' ? 'text-blue-600' : 'text-gray-600'
              }`}>
                {modelsStatus.training_status.status}
              </span>
            </div>
            <div>
              <span className="text-muted-foreground">Samples:</span>
              <span className="ml-2 font-medium">{modelsStatus.training_status.samples_used}</span>
            </div>
            {modelsStatus.training_status.last_trained && (
              <div>
                <span className="text-muted-foreground">Date:</span>
                <span className="ml-2 font-medium">
                  {new Date(modelsStatus.training_status.last_trained).toLocaleString()}
                </span>
              </div>
            )}
            {modelsStatus.training_status.metrics?.ensemble && (
              <div>
                <span className="text-muted-foreground">Accuracy:</span>
                <span className="ml-2 font-medium">
                  {(modelsStatus.training_status.metrics.ensemble.train_accuracy * 100).toFixed(1)}%
                </span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
