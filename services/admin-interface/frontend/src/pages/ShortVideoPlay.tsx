import { useEffect, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { storageApi } from '@/api/client'

/**
 * Same layout as Video Analysis page: play a short video by S3 key.
 * Route: /video/short?key=...
 */
export default function ShortVideoPlay() {
  const [searchParams] = useSearchParams()
  const key = searchParams.get('key')
  const navigate = useNavigate()
  const [url, setUrl] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const filename = key ? key.split('/').pop() || key : ''

  useEffect(() => {
    if (!key) {
      setError('Missing video key')
      setLoading(false)
      return
    }
    let cancelled = false
    storageApi
      .getShortVideoUrl(key)
      .then((data) => {
        if (!cancelled) {
          setUrl(data.url)
          setError(null)
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err.response?.data?.detail || 'Failed to load video')
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [key])

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary" />
      </div>
    )
  }

  if (error || !url) {
    return (
      <div className="space-y-6">
        <div className="flex justify-between items-start">
          <h2 className="text-3xl font-bold">Short Video</h2>
          <button
            onClick={() => navigate('/short-videos')}
            className="px-3 py-1 text-sm text-muted-foreground hover:text-foreground"
          >
            ← Back
          </button>
        </div>
        <p className="text-destructive">{error || 'Video not found'}</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-start">
        <div>
          <h2 className="text-3xl font-bold">Video Analysis</h2>
          <p className="text-muted-foreground mt-1 truncate max-w-2xl" title={filename}>
            {filename}
          </p>
        </div>
        <button
          onClick={() => navigate('/short-videos')}
          className="px-3 py-1 text-sm text-muted-foreground hover:text-foreground"
        >
          ← Back
        </button>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2 space-y-4">
          <div className="rounded-lg overflow-hidden bg-black border border-border">
            <video
              src={url}
              controls
              autoPlay
              className="w-full"
              style={{ maxHeight: '70vh' }}
            >
              Your browser does not support the video tag.
            </video>
          </div>
        </div>
      </div>
    </div>
  )
}
