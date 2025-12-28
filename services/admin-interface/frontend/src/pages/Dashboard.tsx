import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { videosApi, trainingApi } from '@/api/client'

export default function Dashboard() {
  const [videos, setVideos] = useState<any[]>([])
  const [stats, setStats] = useState<any>(null)
  const [trainingStatus, setTrainingStatus] = useState<any>(null)
  const [pairwiseStats, setPairwiseStats] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    try {
      const [videoData, statsData, statusData, pairwiseData] = await Promise.all([
        videosApi.list(),
        trainingApi.getStats().catch(() => null),
        trainingApi.getStatus().catch(() => null),
        trainingApi.getPairwiseStats().catch(() => null)
      ])
      setVideos(videoData.videos || [])
      setStats(statsData)
      setTrainingStatus(statusData)
      setPairwiseStats(pairwiseData)
    } catch (error) {
      console.error('Failed to load data:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <div className="text-muted-foreground">Loading dashboard...</div>
        </div>
      </div>
    )
  }

  const labeledCount = videos.filter(v => v.has_label).length
  const soundCount = videos.filter(v => v.label === 0).length
  const lameCount = videos.filter(v => v.label === 1).length
  const analyzedCount = videos.filter(v => v.has_analysis).length
  const annotatedCount = videos.filter(v => v.has_annotated).length

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <h2 className="text-3xl font-bold">Dashboard</h2>
          <p className="text-muted-foreground mt-1">
            Overview of your lameness detection research
          </p>
        </div>
        <div className="flex gap-2">
          <Link
            to="/upload"
            className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 font-medium"
          >
            + Upload Videos
          </Link>
        </div>
      </div>

      {/* Quick Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
        <StatCard label="Total Videos" value={videos.length} />
        <StatCard label="Analyzed" value={analyzedCount} icon="📊" />
        <StatCard label="Annotated" value={annotatedCount} icon="🎯" />
        <StatCard label="Labeled" value={labeledCount} icon="🏷️" />
        <StatCard 
          label="Sound" 
          value={soundCount} 
          className="bg-green-50 border-green-200"
          valueClassName="text-green-700"
        />
        <StatCard 
          label="Lame" 
          value={lameCount}
          className="bg-red-50 border-red-200"
          valueClassName="text-red-700"
        />
      </div>

      {/* Quick Actions */}
      <div className="grid md:grid-cols-3 gap-6">
        {/* Pairwise Comparison Progress */}
        <div className="border rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Pairwise Comparisons</h3>
          {pairwiseStats ? (
            <>
              <div className="mb-4">
                <div className="flex justify-between text-sm mb-1">
                  <span>Progress</span>
                  <span>{pairwiseStats.pairs_compared} / {pairwiseStats.total_possible_pairs}</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-500 h-2 rounded-full transition-all"
                    style={{ width: `${pairwiseStats.completion_rate * 100}%` }}
                  />
                </div>
              </div>
              <p className="text-sm text-muted-foreground mb-4">
                {pairwiseStats.total_comparisons} total comparisons made
              </p>
              <Link
                to="/pairwise"
                className="block text-center py-2 border rounded-lg hover:bg-accent transition-colors"
              >
                Continue Comparing →
              </Link>
            </>
          ) : (
            <p className="text-muted-foreground text-sm">
              Upload at least 2 videos to start pairwise comparisons
            </p>
          )}
        </div>

        {/* Training Status */}
        <div className="border rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Training Status</h3>
          {trainingStatus ? (
            <>
              <div className="space-y-2 mb-4">
                <div className="flex justify-between text-sm">
                  <span>Status</span>
                  <span className={`font-medium capitalize ${
                    trainingStatus.status === 'completed' ? 'text-green-600' :
                    trainingStatus.status === 'training' ? 'text-blue-600' : 'text-gray-600'
                  }`}>
                    {trainingStatus.status}
                  </span>
                </div>
                {trainingStatus.last_trained && (
                  <div className="flex justify-between text-sm">
                    <span>Last Trained</span>
                    <span>{new Date(trainingStatus.last_trained).toLocaleDateString()}</span>
                  </div>
                )}
                <div className="flex justify-between text-sm">
                  <span>Samples Used</span>
                  <span>{trainingStatus.samples_used}</span>
                </div>
              </div>
              <Link
                to="/training"
                className="block text-center py-2 border rounded-lg hover:bg-accent transition-colors"
              >
                Manage Training →
              </Link>
            </>
          ) : (
            <p className="text-muted-foreground text-sm">
              Training status unavailable
            </p>
          )}
        </div>

        {/* Videos Needing Labels */}
        <div className="border rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Needs Attention</h3>
          <div className="space-y-2">
            {videos.filter(v => !v.has_label).length > 0 ? (
              <>
                <p className="text-sm text-muted-foreground mb-4">
                  {videos.filter(v => !v.has_label).length} videos need labeling
                </p>
                {videos.filter(v => !v.has_label).slice(0, 3).map((video) => (
                  <Link
                    key={video.video_id}
                    to={`/video/${video.video_id}`}
                    className="block p-2 bg-gray-50 rounded hover:bg-gray-100 text-sm truncate"
                  >
                    {video.filename}
                  </Link>
                ))}
                {videos.filter(v => !v.has_label).length > 3 && (
                  <p className="text-xs text-muted-foreground text-center">
                    +{videos.filter(v => !v.has_label).length - 3} more
                  </p>
                )}
              </>
            ) : (
              <p className="text-green-600 text-sm">
                ✓ All videos are labeled!
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Recent Videos */}
      <div>
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-xl font-semibold">Recent Videos</h3>
          <span className="text-sm text-muted-foreground">
            Showing {Math.min(10, videos.length)} of {videos.length}
          </span>
        </div>
        
        {videos.length === 0 ? (
          <div className="text-center py-12 border rounded-lg bg-gray-50">
            <div className="text-4xl mb-4">📹</div>
            <h4 className="text-lg font-semibold mb-2">No videos yet</h4>
            <p className="text-muted-foreground mb-4">
              Upload cow walking videos to start your lameness analysis
            </p>
            <Link
              to="/upload"
              className="inline-block px-6 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90"
            >
              Upload Videos
            </Link>
          </div>
        ) : (
          <div className="border rounded-lg overflow-hidden">
            <table className="w-full">
              <thead className="bg-gray-50 border-b">
                <tr>
                  <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Video</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Status</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Label</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Size</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground"></th>
                </tr>
              </thead>
              <tbody className="divide-y">
                {videos.slice(0, 10).map((video) => (
                  <tr key={video.video_id} className="hover:bg-gray-50">
                    <td className="py-3 px-4">
                      <div className="font-medium truncate max-w-xs">{video.filename}</div>
                      <div className="text-xs text-muted-foreground font-mono">
                        {video.video_id.slice(0, 8)}...
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex gap-1">
                        {video.has_analysis && (
                          <span className="px-2 py-0.5 text-xs bg-blue-100 text-blue-700 rounded">
                            Analyzed
                          </span>
                        )}
                        {video.has_annotated && (
                          <span className="px-2 py-0.5 text-xs bg-purple-100 text-purple-700 rounded">
                            Annotated
                          </span>
                        )}
                        {!video.has_analysis && !video.has_annotated && (
                          <span className="px-2 py-0.5 text-xs bg-gray-100 text-gray-600 rounded">
                            Pending
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      {video.has_label ? (
                        <span className={`px-2 py-0.5 text-xs font-medium rounded ${
                          video.label === 0 
                            ? 'bg-green-100 text-green-700' 
                            : 'bg-red-100 text-red-700'
                        }`}>
                          {video.label === 0 ? 'Sound' : 'Lame'}
                        </span>
                      ) : (
                        <span className="text-muted-foreground text-xs">Unlabeled</span>
                      )}
                    </td>
                    <td className="py-3 px-4 text-sm text-muted-foreground">
                      {(video.file_size / 1024 / 1024).toFixed(1)} MB
                    </td>
                    <td className="py-3 px-4 text-right">
                      <div className="flex gap-2 justify-end">
                        <Link
                          to={`/video/${video.video_id}`}
                          className="px-2 py-1 text-xs bg-muted hover:bg-muted/80 rounded"
                        >
                          View
                        </Link>
                        {video.has_analysis && (
                          <Link
                            to={`/results/${video.video_id}`}
                            className="px-2 py-1 text-xs bg-primary text-primary-foreground hover:bg-primary/90 rounded"
                          >
                            Results
                          </Link>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

interface StatCardProps {
  label: string
  value: number
  icon?: string
  className?: string
  valueClassName?: string
}

function StatCard({ label, value, icon, className = '', valueClassName = '' }: StatCardProps) {
  return (
    <div className={`p-4 border rounded-lg bg-background ${className}`}>
      <div className="flex items-center gap-2 mb-1">
        {icon && <span>{icon}</span>}
        <p className="text-sm text-muted-foreground">{label}</p>
      </div>
      <p className={`text-2xl font-bold ${valueClassName}`}>{value}</p>
    </div>
  )
}
