import { useEffect, useMemo, useState } from 'react'
import { storageApi } from '@/api/client'
import { useAuth } from '@/contexts/AuthContext'
import { CheckSquare, RefreshCw, Scissors, Square } from 'lucide-react'

interface LongVideoItem {
  key: string
  filename: string
  /** User's original upload name; show on first line */
  original_filename?: string | null
  /** Video UUID for admin; show on second line */
  video_id?: string | null
  tenant_id?: string | null
  tenant_name?: string | null
  size: number
  last_modified?: string | null
  etag?: string | null
  status_info?: {
    status: string
    reason?: string
    good_count?: number
    bad_count?: number
    updated_at?: string | null
    percent?: number
    detail?: string
    upload_failed_count?: number
    upload_error?: string
  }
}

interface Tenant {
  id: string
  name: string
}

export default function LongVideos() {
  const { user } = useAuth()
  const isViewer = user?.role === 'viewer'
  const isAdmin = user?.role === 'admin'
  const [items, setItems] = useState<LongVideoItem[]>([])
  const [tenants, setTenants] = useState<Tenant[]>([])
  const [selectedTenant, setSelectedTenant] = useState<string>('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [processing, setProcessing] = useState(false)
  const [lastResult, setLastResult] = useState<string | null>(null)
  const [processErrors, setProcessErrors] = useState<Array<{ key: string; error: string }>>([])
  const [autoRefresh, setAutoRefresh] = useState(true)

  const loadItems = async () => {
    setLoading(true)
    setError(null)
    setProcessErrors([])
    try {
      const data = await storageApi.listRawVideos(false, selectedTenant || undefined)
      setItems(data.items || [])
      if (data.tenants && isAdmin) {
        setTenants(data.tenants)
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load raw videos')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadItems()
  }, [selectedTenant])
  
  // Also load on mount
  useEffect(() => {
    loadItems()
  }, [])

  const hasQueued = items.some((item) => item.status_info?.status === 'queued')
  useEffect(() => {
    if (!autoRefresh) return
    const intervalMs = hasQueued ? 5000 : 20000
    const interval = setInterval(() => {
      loadItems()
    }, intervalMs)
    return () => clearInterval(interval)
  }, [autoRefresh, hasQueued])

  const allSelected = items.length > 0 && selected.size === items.length

  const toggleAll = () => {
    if (allSelected) {
      setSelected(new Set())
      return
    }
    setSelected(new Set(items.map(item => item.key)))
  }

  const clearSelection = () => setSelected(new Set())

  const invertSelection = () => {
    setSelected(prev => {
      const next = new Set<string>()
      items.forEach(item => {
        if (!prev.has(item.key)) {
          next.add(item.key)
        }
      })
      return next
    })
  }

  const toggleOne = (key: string) => {
    setSelected(prev => {
      const next = new Set(prev)
      if (next.has(key)) {
        next.delete(key)
      } else {
        next.add(key)
      }
      return next
    })
  }

  const formatSize = (bytes: number) => {
    const mb = bytes / 1024 / 1024
    return `${mb.toFixed(2)} MB`
  }

  const formatDate = (iso?: string | null) => {
    if (!iso) return '—'
    const date = new Date(iso)
    if (Number.isNaN(date.getTime())) return '—'
    return date.toLocaleString()
  }

  const formatStatus = (item: LongVideoItem) => {
    const info = item.status_info
    if (!info) return 'unknown'
    if (info.status === 'upload_failed') return 'Upload failed'
    if (info.status === 'not_started') return 'Not processed'
    return info.status || 'unknown'
  }

  const formatStatusDetail = (item: LongVideoItem) => {
    const info = item.status_info
    if (!info) return ''
    if (info.status === 'completed') {
      return `good: ${info.good_count ?? 0}, bad: ${info.bad_count ?? 0}`
    }
    if (info.status === 'upload_failed') {
      const err = info.upload_error || info.detail || info.reason || 'Short clips not uploaded to S3'
      const count = info.upload_failed_count
      return count != null && count > 0 ? `${err} (${count} failed)` : err
    }
    if (info.status === 'queued') {
      const at = info.updated_at ? new Date(info.updated_at).toLocaleTimeString() : ''
      return at ? `Queued at ${at}` : (info.detail || info.reason || 'Queued')
    }
    return info.detail || info.reason || ''
  }

  const renderProgress = (item: LongVideoItem) => {
    const info = item.status_info
    const percent = info?.percent
    if (percent === undefined) return null
    const clamped = Math.max(0, Math.min(100, percent))
    const isQueued = info?.status === 'queued'
    return (
      <div className="mt-2">
        <div className="h-2 w-full bg-muted rounded-full overflow-hidden">
          <div
            className={`h-full transition-all ${isQueued ? 'bg-amber-500 animate-pulse' : 'bg-primary'}`}
            style={{ width: `${clamped}%` }}
          />
        </div>
        <div className="text-xs text-muted-foreground mt-1">{clamped}%</div>
      </div>
    )
  }

  const selectedCount = selected.size

  const handleProcess = async () => {
    if (selectedCount === 0 || processing) return
    setProcessing(true)
    setLastResult(null)
    setError(null)
    try {
      const result = await storageApi.processRawVideos(Array.from(selected)) as { count?: number; processed?: unknown[]; errors?: Array<{ key: string; error: string }> }
      const count = result.count ?? 0
      const errList = result.errors ?? []
      if (count > 0) {
        setLastResult(`Queued ${count} video(s) for preprocessing. Status will update automatically.`)
        setSelected(new Set())
      }
      if (count === 0) {
        if (errList.length > 0) {
          setError(`No videos queued (0 accepted). Reasons below. Fix these and try again.`)
          setProcessErrors(errList)
        } else {
          setError('No videos were queued. Keys may be invalid or S3 not configured. Check backend logs.')
          setProcessErrors([])
        }
      } else if (errList.length > 0) {
        setError(`Queued ${count} video(s). ${errList.length} other(s) failed: ${errList.map(e => e.error).join('; ')}`)
        setProcessErrors(errList)
      } else {
        setProcessErrors([])
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to queue preprocessing')
      setProcessErrors([])
    } finally {
      setProcessing(false)
    }
  }

  const summary = useMemo(() => {
    if (loading) return 'Loading...'
    return `${items.length} raw video(s)`
  }, [items.length, loading])

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold">Raw Videos</h1>
          <p className="text-sm text-muted-foreground">
            Select raw videos (check the box) then click Cut &amp; Filter to queue preprocessing. Status updates every 5s when queued.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {isAdmin && tenants.length > 0 && (
            <select
              value={selectedTenant}
              onChange={(e) => setSelectedTenant(e.target.value)}
              className="px-3 py-2 rounded-md border bg-background text-sm"
            >
              <option value="">All Tenants</option>
              {tenants.map((tenant) => (
                <option key={tenant.id} value={tenant.id}>
                  {tenant.name}
                </option>
              ))}
            </select>
          )}
          <button
            onClick={loadItems}
            className="inline-flex items-center gap-2 px-3 py-2 rounded-md bg-secondary text-secondary-foreground hover:bg-secondary/80"
          >
            <RefreshCw className="h-4 w-4" />
            Refresh
          </button>
          <button
            onClick={() => setAutoRefresh(prev => !prev)}
            className="inline-flex items-center gap-2 px-3 py-2 rounded-md border text-sm"
          >
            Auto refresh: {autoRefresh ? 'On' : 'Off'}
          </button>
          {!isViewer && (
            <button
              onClick={handleProcess}
              disabled={selectedCount === 0 || processing}
              className="inline-flex items-center gap-2 px-3 py-2 rounded-md bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
            >
              <Scissors className="h-4 w-4" />
              {processing ? 'Processing...' : `Cut & Filter (${selectedCount})`}
            </button>
          )}
        </div>
      </div>

      <div className="border rounded-lg bg-card">
        <div className="flex items-center justify-between px-4 py-3 border-b text-sm text-muted-foreground">
          <div className="flex items-center gap-2">
            {!isViewer && (
              <>
                <button
                  onClick={toggleAll}
                  className="inline-flex items-center gap-1 text-foreground"
                >
                  {allSelected ? (
                    <CheckSquare className="h-4 w-4" />
                  ) : (
                    <Square className="h-4 w-4" />
                  )}
                  Select all
                </button>
                <button
                  onClick={clearSelection}
                  className="inline-flex items-center gap-1 text-foreground"
                >
                  Clear
                </button>
                <button
                  onClick={invertSelection}
                  className="inline-flex items-center gap-1 text-foreground"
                >
                  Invert
                </button>
              </>
            )}
            <span className="text-muted-foreground">{summary}</span>
          </div>
          {!isViewer && <span>{selectedCount} selected</span>}
        </div>

        {error && (
          <div className="px-4 py-3 text-sm text-red-600 dark:text-red-400 border-b">
            {error}
          </div>
        )}
        {processErrors.length > 0 && (
          <div className="px-4 py-3 text-sm border-b bg-amber-50 dark:bg-amber-950/30 border-amber-200 dark:border-amber-800">
            <p className="font-medium text-amber-800 dark:text-amber-200 mb-2">Why 0 queued — server reasons:</p>
            <ul className="list-disc list-inside space-y-1 text-amber-700 dark:text-amber-300">
              {processErrors.map((e, i) => (
                <li key={i}>
                  <span className="font-mono text-xs break-all">{e.key}</span>: {e.error}
                </li>
              ))}
            </ul>
          </div>
        )}
        {lastResult && (
          <div className="px-4 py-3 text-sm text-green-700 dark:text-green-400 border-b">
            {lastResult}
          </div>
        )}
        {hasQueued && (
          <div className="px-4 py-3 text-sm bg-amber-500/10 text-amber-700 dark:text-amber-300 border-b flex flex-wrap items-center gap-2">
            <span className="font-medium">Queued:</span>
            <span>Status will update every 5s while any video is queued. If it stays at 0% for a long time, the pipeline worker may not be running — start it with Docker: <code className="px-1 py-0.5 rounded bg-black/10 text-xs">docker compose --profile gpu up -d</code> or <code className="px-1 py-0.5 rounded bg-black/10 text-xs">--profile cpu</code> for <code className="text-xs">video-preprocessing</code>.</span>
          </div>
        )}

        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-muted/40">
              <tr>
                {!isViewer && <th className="text-left px-4 py-3 font-medium">Select</th>}
                {isAdmin && <th className="text-left px-4 py-3 font-medium">Tenant</th>}
                <th className="text-left px-4 py-3 font-medium">Filename</th>
                <th className="text-left px-4 py-3 font-medium">Size</th>
                <th className="text-left px-4 py-3 font-medium">Status</th>
                <th className="text-left px-4 py-3 font-medium">Last Modified</th>
              </tr>
            </thead>
            <tbody>
              {items.map(item => (
                <tr key={item.key} className="border-t">
                  {!isViewer && (
                    <td className="px-4 py-3">
                      <input
                        type="checkbox"
                        checked={selected.has(item.key)}
                        onChange={() => toggleOne(item.key)}
                      />
                    </td>
                  )}
                  {isAdmin && (
                    <td className="px-4 py-3">
                      <span className="text-sm font-medium">{item.tenant_name || '—'}</span>
                    </td>
                  )}
                  <td className="px-4 py-3">
                    <div className="font-medium" title={item.original_filename || item.filename}>
                      {item.original_filename || item.filename}
                    </div>
                    <div className="text-xs text-muted-foreground font-mono" title={item.video_id || item.key}>
                      {item.video_id || item.key}
                    </div>
                  </td>
                  <td className="px-4 py-3">{formatSize(item.size)}</td>
                  <td className="px-4 py-3">
                    <div className="capitalize">{formatStatus(item)}</div>
                    <div className="text-xs text-muted-foreground">
                      {formatStatusDetail(item)}
                    </div>
                    {renderProgress(item)}
                  </td>
                  <td className="px-4 py-3">{formatDate(item.last_modified)}</td>
                </tr>
              ))}
              {!loading && items.length === 0 && (
                <tr>
                  <td colSpan={isViewer ? (isAdmin ? 5 : 4) : (isAdmin ? 6 : 5)} className="px-4 py-8 text-center text-muted-foreground">
                    No raw videos found
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
