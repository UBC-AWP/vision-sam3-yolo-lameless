import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { videosApi } from '@/api/client'
import { RefreshCw, Upload, Search, Film, PlayCircle } from 'lucide-react'

type LabelFilter = 'all' | 'sound' | 'lame' | 'unlabeled'

export default function VideoLibrary() {
  const navigate = useNavigate()
  const [videos, setVideos] = useState<any[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [labelFilter, setLabelFilter] = useState<LabelFilter>('all')
  const [statusFilter, setStatusFilter] = useState('all')
  const [skip, setSkip] = useState(0)
  const limit = 100

  const loadVideos = async () => {
    setLoading(true)
    setError(null)
    try {
      const params: { label?: number; status?: string } = {}
      if (labelFilter === 'sound') params.label = 0
      if (labelFilter === 'lame') params.label = 1
      if (statusFilter !== 'all') params.status = statusFilter
      const data = await videosApi.list(skip, limit, params)
      setVideos(data.videos || [])
      setTotal(data.total || 0)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load videos')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadVideos()
  }, [skip, labelFilter, statusFilter])

  const filteredVideos = useMemo(() => {
    const q = search.trim().toLowerCase()
    return videos.filter(v => {
      if (labelFilter === 'unlabeled' && v.label !== null) return false
      if (!q) return true
      return [
        v.filename,
        v.original_filename,
        v.video_id,
      ].some((field: string | undefined) => (field || '').toLowerCase().includes(q))
    })
  }, [videos, search, labelFilter])

  const totalPages = Math.max(1, Math.ceil(total / limit))
  const currentPage = Math.floor(skip / limit) + 1

  const formatSize = (bytes: number) => {
    if (!bytes && bytes !== 0) return '—'
    const mb = bytes / 1024 / 1024
    return `${mb.toFixed(2)} MB`
  }

  const formatDate = (iso?: string) => {
    if (!iso) return '—'
    const date = new Date(iso)
    if (Number.isNaN(date.getTime())) return '—'
    return date.toLocaleString()
  }

  const labelText = (label: number | null) => {
    if (label === 0) return 'Sound'
    if (label === 1) return 'Lame'
    return 'Unlabeled'
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold">Videos</h1>
          <p className="text-sm text-muted-foreground">
            Manage uploaded videos and review results
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={loadVideos}
            className="inline-flex items-center gap-2 px-3 py-2 rounded-md bg-secondary text-secondary-foreground hover:bg-secondary/80"
          >
            <RefreshCw className="h-4 w-4" />
            Refresh
          </button>
          <Link
            to="/upload"
            className="inline-flex items-center gap-2 px-3 py-2 rounded-md bg-primary text-primary-foreground hover:bg-primary/90"
          >
            <Upload className="h-4 w-4" />
            Upload
          </Link>
        </div>
      </div>

      <div className="grid gap-3 md:grid-cols-3">
        <div className="relative">
          <Search className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search filename or video ID"
            className="w-full pl-9 pr-3 py-2 border rounded-md bg-background"
          />
        </div>
        <select
          value={labelFilter}
          onChange={(e) => {
            setSkip(0)
            setLabelFilter(e.target.value as LabelFilter)
          }}
          className="w-full px-3 py-2 border rounded-md bg-background"
        >
          <option value="all">All labels</option>
          <option value="sound">Sound</option>
          <option value="lame">Lame</option>
          <option value="unlabeled">Unlabeled</option>
        </select>
        <select
          value={statusFilter}
          onChange={(e) => {
            setSkip(0)
            setStatusFilter(e.target.value)
          }}
          className="w-full px-3 py-2 border rounded-md bg-background"
        >
          <option value="all">All status</option>
          <option value="uploaded">Uploaded</option>
          <option value="processing">Processing</option>
          <option value="analyzed">Analyzed</option>
          <option value="failed">Failed</option>
        </select>
      </div>

      <div className="border rounded-lg bg-card">
        <div className="flex items-center justify-between px-4 py-3 border-b">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Film className="h-4 w-4" />
            {loading ? 'Loading...' : `${filteredVideos.length} / ${total} videos`}
          </div>
          <div className="flex items-center gap-2 text-sm">
            <button
              onClick={() => setSkip(Math.max(0, skip - limit))}
              disabled={currentPage <= 1}
              className="px-2 py-1 rounded border disabled:opacity-50"
            >
              Prev
            </button>
            <span>
              Page {currentPage} / {totalPages}
            </span>
            <button
              onClick={() => setSkip(Math.min((totalPages - 1) * limit, skip + limit))}
              disabled={currentPage >= totalPages}
              className="px-2 py-1 rounded border disabled:opacity-50"
            >
              Next
            </button>
          </div>
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
                <th className="text-left px-4 py-3 font-medium">Filename</th>
                <th className="text-left px-4 py-3 font-medium">Size</th>
                <th className="text-left px-4 py-3 font-medium">Status</th>
                <th className="text-left px-4 py-3 font-medium">Label</th>
                <th className="text-left px-4 py-3 font-medium">Storage</th>
                <th className="text-left px-4 py-3 font-medium">Uploaded</th>
                <th className="text-right px-4 py-3 font-medium">Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredVideos.map((video) => (
                <tr key={video.video_id} className="border-t">
                  <td className="px-4 py-3">
                    <div className="font-medium truncate max-w-[320px]" title={video.original_filename || video.filename}>
                      {video.original_filename || video.filename}
                    </div>
                    <div className="text-xs text-muted-foreground font-mono">
                      {video.video_id}
                    </div>
                  </td>
                  <td className="px-4 py-3">{formatSize(video.file_size)}</td>
                  <td className="px-4 py-3 capitalize">{video.status || 'uploaded'}</td>
                  <td className="px-4 py-3">{labelText(video.label)}</td>
                  <td className="px-4 py-3 uppercase">{video.storage}</td>
                  <td className="px-4 py-3">{formatDate(video.uploaded_at)}</td>
                  <td className="px-4 py-3 text-right">
                    <div className="inline-flex items-center gap-2">
                      <button
                        onClick={() => navigate(`/video/${video.video_id}`)}
                        className="inline-flex items-center gap-1 px-2 py-1 rounded bg-primary/10 text-primary hover:bg-primary/20"
                      >
                        <PlayCircle className="h-4 w-4" />
                        View
                      </button>
                      <button
                        onClick={() => navigate(`/results/${video.video_id}`)}
                        className="inline-flex items-center gap-1 px-2 py-1 rounded bg-secondary text-secondary-foreground hover:bg-secondary/80"
                      >
                        Results
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
              {!loading && filteredVideos.length === 0 && (
                <tr>
                  <td colSpan={7} className="px-4 py-8 text-center text-muted-foreground">
                    No videos found
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
