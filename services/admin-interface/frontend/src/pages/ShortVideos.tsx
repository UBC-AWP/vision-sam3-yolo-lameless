import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { storageApi } from '@/api/client'
import { useAuth } from '@/contexts/AuthContext'
import { ExternalLink, RefreshCw, PlayCircle } from 'lucide-react'

interface ShortVideoItem {
  key: string
  filename: string
  size: number
  last_modified?: string | null
  url?: string | null
  // Backend still returns label/probability, but this page does not display them.
  label?: string | null
  prob_bad?: number | null
  parent_video_id?: string | null
  parent_video_name?: string | null
  clip_slug?: string | null
  tenant_id?: string | null
  tenant_name?: string | null
}

interface Tenant {
  id: string
  name: string
}

export default function ShortVideos() {
  const navigate = useNavigate()
  const { user } = useAuth()
  const isAdmin = user?.role === 'admin'
  const [items, setItems] = useState<ShortVideoItem[]>([])
  const [tenants, setTenants] = useState<Tenant[]>([])
  const [selectedTenant, setSelectedTenant] = useState<string>('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [autoRefresh, setAutoRefresh] = useState(true)

  const buildDisplayName = (item: ShortVideoItem) => {
    // Prefer the original raw-video filename as the display prefix.
    const base = item.parent_video_name || item.filename
    const dotIndex = base.lastIndexOf('.')
    const stem = dotIndex > 0 ? base.slice(0, dotIndex) : base
    const ext = dotIndex > 0 ? base.slice(dotIndex) : ''
    const suffix = item.clip_slug || 'clip_1'
    return `${stem}_${suffix}${ext}`
  }

  const loadItems = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await storageApi.listShortVideos(true, selectedTenant || undefined)
      console.log('[ShortVideos] API Response:', data)
      console.log('[ShortVideos] Items count:', data.items?.length || 0)
      if (data.items && data.items.length > 0) {
        console.log('[ShortVideos] First item:', data.items[0])
        console.log('[ShortVideos] First item display name:', buildDisplayName(data.items[0]))
      }
      setItems(data.items || [])
      if (data.tenants && isAdmin) {
        setTenants(data.tenants)
      }
    } catch (err: any) {
      console.error('[ShortVideos] Error loading items:', err)
      setError(err.response?.data?.detail || 'Failed to load short videos')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadItems()
  }, [selectedTenant])

  useEffect(() => {
    if (!autoRefresh) return
    const interval = setInterval(() => {
      loadItems()
    }, 20000)
    return () => clearInterval(interval)
  }, [autoRefresh])

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

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold">Short Videos</h1>
          <p className="text-sm text-muted-foreground">
            Filtered short clips generated from raw videos
          </p>
        </div>
        <div className="flex items-center gap-2">
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
        </div>
      </div>

      <div className="border rounded-lg bg-card">
        <div className="px-4 py-3 border-b text-sm text-muted-foreground">
          {loading ? 'Loading...' : `${items.length} short video(s)`}
        </div>

        {error && (
          <div className="px-4 py-3 text-sm text-red-600 border-b">
            {error}
          </div>
        )}

        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-muted/40">
              <tr>
                {isAdmin && <th className="text-left px-4 py-3 font-medium">Tenant</th>}
                <th className="text-left px-4 py-3 font-medium">Filename</th>
                <th className="text-left px-4 py-3 font-medium">Size</th>
                <th className="text-left px-4 py-3 font-medium">Parent Video</th>
                <th className="text-left px-4 py-3 font-medium">Modified</th>
                <th className="text-right px-4 py-3 font-medium">Actions</th>
              </tr>
            </thead>
            <tbody>
              {items.map(item => (
                <tr key={item.key} className="border-t">
                  {isAdmin && (
                    <td className="px-4 py-3">
                      <span className="text-sm font-medium">{item.tenant_name || '—'}</span>
                    </td>
                  )}
                  <td className="px-4 py-3">
                    <div
                      className="font-medium truncate max-w-[200px]"
                      title={item.filename}
                    >
                      {buildDisplayName(item)}
                    </div>
                    <div className="text-xs text-muted-foreground font-mono truncate max-w-[200px]" title={item.key}>
                      {item.key}
                    </div>
                  </td>
                  <td className="px-4 py-3">{formatSize(item.size)}</td>
                  <td className="px-4 py-3">
                    {item.parent_video_name ? (
                      <span className="text-sm" title={item.parent_video_name}>
                        {item.parent_video_name}
                      </span>
                    ) : item.parent_video_id ? (
                      <span className="font-mono text-xs" title={item.parent_video_id}>
                        {item.parent_video_id.slice(0, 8)}…
                      </span>
                    ) : (
                      '—'
                    )}
                  </td>
                  <td className="px-4 py-3">{formatDate(item.last_modified)}</td>
                  <td className="px-4 py-3 text-right">
                    <div className="inline-flex items-center gap-2">
                      {item.url ? (
                        <>
                          <button
                            type="button"
                            onClick={() => navigate(`/video/short?key=${encodeURIComponent(item.key)}`)}
                            className="inline-flex items-center gap-1 px-2 py-1 rounded bg-primary/10 text-primary hover:bg-primary/20"
                          >
                            <PlayCircle className="h-4 w-4" />
                            View
                          </button>
                          <a
                            href={item.url}
                            target="_blank"
                            rel="noreferrer"
                            className="inline-flex items-center gap-1 px-2 py-1 rounded bg-secondary text-secondary-foreground hover:bg-secondary/80"
                          >
                            <ExternalLink className="h-4 w-4" />
                            Open
                          </a>
                        </>
                      ) : (
                        <span className="text-xs text-muted-foreground">No URL</span>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
              {!loading && items.length === 0 && (
                <tr>
                  <td colSpan={isAdmin ? 6 : 5} className="px-4 py-8 text-center text-muted-foreground">
                    No short videos found
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
