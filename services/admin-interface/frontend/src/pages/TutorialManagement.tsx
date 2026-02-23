import { useEffect, useState } from 'react'
import { tutorialApi, videosApi, GoldTask, CreateGoldTaskData } from '@/api/client'

interface VideoOption {
  id: string
  name: string
}

export default function TutorialManagement() {
  const [tasks, setTasks] = useState<GoldTask[]>([])
  const [loading, setLoading] = useState(true)
  const [stats, setStats] = useState<any>(null)
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [showEditModal, setShowEditModal] = useState(false)
  const [editingTask, setEditingTask] = useState<GoldTask | null>(null)
  const [videos, setVideos] = useState<VideoOption[]>([])
  const [filter, setFilter] = useState<'all' | 'tutorial' | 'validation'>('all')
  const [autoGenerating, setAutoGenerating] = useState(false)

  // Form state for create/edit
  const [formData, setFormData] = useState<CreateGoldTaskData>({
    video_id_1: '',
    video_id_2: '',
    correct_winner: 1,
    correct_degree: 2,
    difficulty: 'medium',
    description: '',
    hint: '',
    is_tutorial: true,
    tutorial_order: 0,
  })

  useEffect(() => {
    loadData()
  }, [filter])

  const loadData = async () => {
    setLoading(true)
    try {
      // Load videos separately to ensure they load even if other APIs fail
      const videosData = await videosApi.list()
      console.log('Videos API response:', videosData)
      const videosList = videosData.videos?.map((v: any) => ({
        id: v.video_id,
        name: v.original_filename || v.filename || v.video_id
      })) || []
      console.log('Mapped videos:', videosList)
      setVideos(videosList)

      // Load tasks and stats
      const [tasksData, statsData] = await Promise.all([
        tutorialApi.listTasks({
          is_tutorial: filter === 'tutorial' ? true : filter === 'validation' ? false : undefined,
          is_active: true,
        }),
        tutorialApi.getStats(),
      ])
      setTasks(tasksData.tasks || [])
      setStats(statsData)
    } catch (error) {
      console.error('Failed to load data:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleCreate = async () => {
    try {
      await tutorialApi.createTask(formData)
      setShowCreateModal(false)
      resetForm()
      await loadData()
    } catch (error) {
      console.error('Failed to create task:', error)
      alert('Failed to create task')
    }
  }

  const handleUpdate = async () => {
    if (!editingTask) return
    try {
      await tutorialApi.updateTask(editingTask.id, formData)
      setShowEditModal(false)
      setEditingTask(null)
      resetForm()
      await loadData()
    } catch (error) {
      console.error('Failed to update task:', error)
      alert('Failed to update task')
    }
  }

  const handleDelete = async (taskId: string) => {
    if (!confirm('Are you sure you want to delete this task?')) return
    try {
      await tutorialApi.deleteTask(taskId)
      await loadData()
    } catch (error) {
      console.error('Failed to delete task:', error)
      alert('Failed to delete task')
    }
  }

  const handleAutoGenerate = async () => {
    setAutoGenerating(true)
    try {
      const result = await tutorialApi.autoGenerate(3)
      alert(`Generated ${result.generated} tutorial examples`)
      await loadData()
    } catch (error) {
      console.error('Failed to auto-generate:', error)
      alert('Failed to auto-generate tutorials')
    } finally {
      setAutoGenerating(false)
    }
  }

  const openEditModal = (task: GoldTask) => {
    setEditingTask(task)
    setFormData({
      video_id_1: task.video_id_1,
      video_id_2: task.video_id_2,
      correct_winner: task.correct_winner,
      correct_degree: task.correct_degree || 2,
      difficulty: task.difficulty,
      description: task.description || '',
      hint: task.hint || '',
      is_tutorial: task.is_tutorial,
      tutorial_order: task.tutorial_order || 0,
    })
    setShowEditModal(true)
  }

  const resetForm = () => {
    setFormData({
      video_id_1: '',
      video_id_2: '',
      correct_winner: 1,
      correct_degree: 2,
      difficulty: 'medium',
      description: '',
      hint: '',
      is_tutorial: true,
      tutorial_order: 0,
    })
  }

  const getCorrectAnswerLabel = (winner: number, degree: number) => {
    if (winner === 0) return 'Equal / Cannot Decide'
    const direction = winner === 1 ? 'A' : 'B'
    const intensity = degree === 3 ? 'Much More' : degree === 2 ? 'More' : 'Slightly More'
    return `${direction} ${intensity} Lame`
  }

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'easy': return 'bg-success/20 text-success'
      case 'medium': return 'bg-warning/20 text-warning'
      case 'hard': return 'bg-destructive/20 text-destructive'
      default: return 'bg-muted text-muted-foreground'
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <div className="text-muted-foreground">Loading tutorials...</div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold">Tutorial Management</h2>
          <p className="text-muted-foreground mt-1">
            Manage tutorial examples and validation gold tasks for rater training
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={handleAutoGenerate}
            disabled={autoGenerating}
            className="px-4 py-2 border rounded-lg hover:bg-accent disabled:opacity-50"
          >
            {autoGenerating ? 'Generating...' : 'Auto-Generate Tutorials'}
          </button>
          <button
            onClick={() => {
              resetForm()
              setShowCreateModal(true)
            }}
            className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90"
          >
            + Create Task
          </button>
        </div>
      </div>

      {/* Stats Cards */}
      {stats && (
        <div className="grid grid-cols-4 gap-4">
          <div className="border rounded-lg p-4">
            <div className="text-2xl font-bold">{stats.total_tasks}</div>
            <div className="text-sm text-muted-foreground">Total Tasks</div>
          </div>
          <div className="border rounded-lg p-4">
            <div className="text-2xl font-bold text-blue-600">{stats.tutorial_tasks}</div>
            <div className="text-sm text-muted-foreground">Tutorial Examples</div>
          </div>
          <div className="border rounded-lg p-4">
            <div className="text-2xl font-bold text-purple-600">{stats.validation_tasks}</div>
            <div className="text-sm text-muted-foreground">Validation Tasks</div>
          </div>
          <div className="border rounded-lg p-4">
            <div className="text-2xl font-bold text-green-600">{stats.active_tasks}</div>
            <div className="text-sm text-muted-foreground">Active Tasks</div>
          </div>
        </div>
      )}

      {/* Filter Tabs */}
      <div className="flex gap-2 border-b">
        {(['all', 'tutorial', 'validation'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setFilter(tab)}
            className={`px-4 py-2 text-sm font-medium transition-colors ${
              filter === tab
                ? 'text-primary border-b-2 border-primary'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            {tab === 'all' ? 'All Tasks' : tab === 'tutorial' ? 'Tutorials' : 'Validation'}
          </button>
        ))}
      </div>

      {/* Tasks Table */}
      <div className="border border-border rounded-lg overflow-hidden">
        <table className="w-full">
          <thead className="bg-muted/50">
            <tr>
              <th className="px-4 py-3 text-left text-sm font-medium text-foreground">Order</th>
              <th className="px-4 py-3 text-left text-sm font-medium text-foreground">Type</th>
              <th className="px-4 py-3 text-left text-sm font-medium text-foreground">Videos</th>
              <th className="px-4 py-3 text-left text-sm font-medium text-foreground">Correct Answer</th>
              <th className="px-4 py-3 text-left text-sm font-medium text-foreground">Difficulty</th>
              <th className="px-4 py-3 text-left text-sm font-medium text-foreground">Description</th>
              <th className="px-4 py-3 text-left text-sm font-medium text-foreground">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {tasks.map((task) => (
              <tr key={task.id} className="hover:bg-accent/50 transition-colors">
                <td className="px-4 py-3 text-sm">
                  {task.is_tutorial ? task.tutorial_order || '-' : '-'}
                </td>
                <td className="px-4 py-3">
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                    task.is_tutorial ? 'bg-info/20 text-info' : 'bg-primary/20 text-primary'
                  }`}>
                    {task.is_tutorial ? 'Tutorial' : 'Validation'}
                  </span>
                </td>
                <td className="px-4 py-3">
                  <div className="flex gap-2 items-center">
                    <div className="w-24 h-14 bg-muted rounded overflow-hidden">
                      <video
                        src={videosApi.getStreamUrl(task.video_id_1)}
                        className="w-full h-full object-cover"
                        muted
                      />
                    </div>
                    <span className="text-muted-foreground">vs</span>
                    <div className="w-24 h-14 bg-muted rounded overflow-hidden">
                      <video
                        src={videosApi.getStreamUrl(task.video_id_2)}
                        className="w-full h-full object-cover"
                        muted
                      />
                    </div>
                  </div>
                </td>
                <td className="px-4 py-3 text-sm text-foreground">
                  {getCorrectAnswerLabel(task.correct_winner, task.correct_degree || 2)}
                </td>
                <td className="px-4 py-3">
                  <span className={`px-2 py-1 rounded-full text-xs font-medium capitalize ${getDifficultyColor(task.difficulty)}`}>
                    {task.difficulty}
                  </span>
                </td>
                <td className="px-4 py-3 text-sm text-muted-foreground max-w-xs truncate">
                  {task.description || '-'}
                </td>
                <td className="px-4 py-3">
                  <div className="flex gap-2">
                    <button
                      onClick={() => openEditModal(task)}
                      className="px-3 py-1 text-sm border border-border rounded hover:bg-accent transition-colors"
                    >
                      Edit
                    </button>
                    <button
                      onClick={() => handleDelete(task.id)}
                      className="px-3 py-1 text-sm text-destructive border border-destructive/30 rounded hover:bg-destructive/10 transition-colors"
                    >
                      Delete
                    </button>
                  </div>
                </td>
              </tr>
            ))}
            {tasks.length === 0 && (
              <tr>
                <td colSpan={7} className="px-4 py-8 text-center text-muted-foreground">
                  No tasks found. Create one or use auto-generate.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Create/Edit Modal */}
      {(showCreateModal || showEditModal) && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-card border border-border rounded-lg p-6 max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto shadow-xl">
            <h3 className="text-lg font-semibold mb-4 text-foreground">
              {showCreateModal ? 'Create New Task' : 'Edit Task'}
            </h3>

            <div className="space-y-4">
              {/* Task Type */}
              <div>
                <label className="block text-sm font-medium mb-1 text-foreground">Task Type</label>
                <div className="flex gap-4">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="radio"
                      checked={formData.is_tutorial}
                      onChange={() => setFormData({ ...formData, is_tutorial: true })}
                      className="accent-primary"
                    />
                    <span className="text-foreground">Tutorial (shown to new raters)</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="radio"
                      checked={!formData.is_tutorial}
                      onChange={() => setFormData({ ...formData, is_tutorial: false })}
                      className="accent-primary"
                    />
                    <span className="text-foreground">Validation (hidden gold task)</span>
                  </label>
                </div>
              </div>

              {/* Video Selection - Drag & Drop */}
              <div className="grid grid-cols-2 gap-4">
                {/* Video A Drop Zone */}
                <div>
                  <label className="block text-sm font-medium mb-2 text-foreground">
                    Video A (Cow A)
                    {formData.video_id_1 && (
                      <button
                        onClick={() => setFormData({ ...formData, video_id_1: '' })}
                        className="ml-2 text-xs text-destructive hover:text-destructive/80"
                      >
                        Clear
                      </button>
                    )}
                  </label>
                  <div
                    onDragOver={(e) => { e.preventDefault(); e.currentTarget.classList.add('border-info', 'bg-info/10') }}
                    onDragLeave={(e) => { e.currentTarget.classList.remove('border-info', 'bg-info/10') }}
                    onDrop={(e) => {
                      e.preventDefault()
                      e.currentTarget.classList.remove('border-info', 'bg-info/10')
                      const videoId = e.dataTransfer.getData('video_id')
                      if (videoId && videoId !== formData.video_id_2) {
                        setFormData({ ...formData, video_id_1: videoId })
                      }
                    }}
                    className={`border-2 border-dashed rounded-lg overflow-hidden transition-all ${
                      formData.video_id_1 ? 'border-info bg-info/10' : 'border-border'
                    }`}
                  >
                    {formData.video_id_1 ? (
                      <div className="relative">
                        <video
                          src={videosApi.getStreamUrl(formData.video_id_1)}
                          className="w-full aspect-video object-cover"
                          controls
                          muted
                        />
                        <div className="absolute top-2 left-2 px-2 py-1 bg-info text-white text-xs font-bold rounded">
                          A
                        </div>
                      </div>
                    ) : (
                      <div className="aspect-video flex flex-col items-center justify-center text-muted-foreground bg-muted/30">
                        <span className="text-3xl mb-2">📥</span>
                        <span className="text-sm">Drop Video A here</span>
                      </div>
                    )}
                  </div>
                </div>

                {/* Video B Drop Zone */}
                <div>
                  <label className="block text-sm font-medium mb-2 text-foreground">
                    Video B (Cow B)
                    {formData.video_id_2 && (
                      <button
                        onClick={() => setFormData({ ...formData, video_id_2: '' })}
                        className="ml-2 text-xs text-destructive hover:text-destructive/80"
                      >
                        Clear
                      </button>
                    )}
                  </label>
                  <div
                    onDragOver={(e) => { e.preventDefault(); e.currentTarget.classList.add('border-success', 'bg-success/10') }}
                    onDragLeave={(e) => { e.currentTarget.classList.remove('border-success', 'bg-success/10') }}
                    onDrop={(e) => {
                      e.preventDefault()
                      e.currentTarget.classList.remove('border-success', 'bg-success/10')
                      const videoId = e.dataTransfer.getData('video_id')
                      if (videoId && videoId !== formData.video_id_1) {
                        setFormData({ ...formData, video_id_2: videoId })
                      }
                    }}
                    className={`border-2 border-dashed rounded-lg overflow-hidden transition-all ${
                      formData.video_id_2 ? 'border-success bg-success/10' : 'border-border'
                    }`}
                  >
                    {formData.video_id_2 ? (
                      <div className="relative">
                        <video
                          src={videosApi.getStreamUrl(formData.video_id_2)}
                          className="w-full aspect-video object-cover"
                          controls
                          muted
                        />
                        <div className="absolute top-2 left-2 px-2 py-1 bg-success text-white text-xs font-bold rounded">
                          B
                        </div>
                      </div>
                    ) : (
                      <div className="aspect-video flex flex-col items-center justify-center text-muted-foreground bg-muted/30">
                        <span className="text-3xl mb-2">📥</span>
                        <span className="text-sm">Drop Video B here</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Available Videos - Draggable Grid */}
              {videos.length > 0 && (
                <div className="border-t border-border pt-4">
                  <h4 className="font-medium mb-2 text-sm text-foreground">Available Videos ({videos.length}) - Drag to drop zones above</h4>
                  <div className="grid grid-cols-5 gap-2 max-h-48 overflow-y-auto p-2 bg-muted/30 rounded-lg">
                    {videos.map((video) => {
                      const isSelectedA = formData.video_id_1 === video.id
                      const isSelectedB = formData.video_id_2 === video.id
                      return (
                        <div
                          key={video.id}
                          draggable
                          onDragStart={(e) => {
                            e.dataTransfer.setData('video_id', video.id)
                            e.dataTransfer.effectAllowed = 'copy'
                          }}
                          className={`relative border-2 rounded-lg overflow-hidden cursor-grab active:cursor-grabbing transition-all ${
                            isSelectedA ? 'border-info ring-2 ring-info/30' :
                            isSelectedB ? 'border-success ring-2 ring-success/30' :
                            'border-border hover:border-muted-foreground hover:shadow-md bg-card'
                          }`}
                        >
                          <div className="relative">
                            <video
                              src={videosApi.getStreamUrl(video.id)}
                              className="w-full aspect-video object-cover bg-muted"
                              muted
                              onMouseEnter={(e) => e.currentTarget.play()}
                              onMouseLeave={(e) => { e.currentTarget.pause(); e.currentTarget.currentTime = 0 }}
                            />
                            {(isSelectedA || isSelectedB) && (
                              <div className={`absolute top-1 right-1 px-1.5 py-0.5 rounded text-xs font-bold text-white ${
                                isSelectedA ? 'bg-info' : 'bg-success'
                              }`}>
                                {isSelectedA ? 'A' : 'B'}
                              </div>
                            )}
                          </div>
                          <div className="p-1 text-[10px] truncate bg-card border-t border-border text-center text-foreground">
                            {(video.name || video.id).slice(0, 8)}...
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}

              {/* Correct Answer */}
              <div>
                <label className="block text-sm font-medium mb-1 text-foreground">Correct Answer</label>
                <select
                  value={`${formData.correct_winner}:${formData.correct_degree}`}
                  onChange={(e) => {
                    const [winner, degree] = e.target.value.split(':').map(Number)
                    setFormData({ ...formData, correct_winner: winner, correct_degree: degree })
                  }}
                  className="w-full p-2 border border-border rounded-lg bg-background text-foreground focus:ring-2 focus:ring-primary focus:outline-none"
                >
                  <option value="1:3">A Much More Lame (-3)</option>
                  <option value="1:2">A More Lame (-2)</option>
                  <option value="1:1">A Slightly More Lame (-1)</option>
                  <option value="0:1">Equal / Cannot Decide (0)</option>
                  <option value="2:1">B Slightly More Lame (+1)</option>
                  <option value="2:2">B More Lame (+2)</option>
                  <option value="2:3">B Much More Lame (+3)</option>
                </select>
              </div>

              {/* Difficulty */}
              <div>
                <label className="block text-sm font-medium mb-1 text-foreground">Difficulty</label>
                <select
                  value={formData.difficulty}
                  onChange={(e) => setFormData({ ...formData, difficulty: e.target.value })}
                  className="w-full p-2 border border-border rounded-lg bg-background text-foreground focus:ring-2 focus:ring-primary focus:outline-none"
                >
                  <option value="easy">Easy - Obvious difference</option>
                  <option value="medium">Medium - Moderate difference</option>
                  <option value="hard">Hard - Subtle difference</option>
                </select>
              </div>

              {/* Tutorial Order */}
              {formData.is_tutorial && (
                <div>
                  <label className="block text-sm font-medium mb-1 text-foreground">Tutorial Order</label>
                  <input
                    type="number"
                    value={formData.tutorial_order}
                    onChange={(e) => setFormData({ ...formData, tutorial_order: parseInt(e.target.value) || 0 })}
                    className="w-full p-2 border border-border rounded-lg bg-background text-foreground focus:ring-2 focus:ring-primary focus:outline-none"
                    min="0"
                  />
                  <p className="text-xs text-muted-foreground mt-1">
                    Lower numbers appear first in the tutorial sequence
                  </p>
                </div>
              )}

              {/* Description */}
              <div>
                <label className="block text-sm font-medium mb-1 text-foreground">Description (what to look for)</label>
                <textarea
                  value={formData.description}
                  onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                  className="w-full p-2 border border-border rounded-lg bg-background text-foreground focus:ring-2 focus:ring-primary focus:outline-none"
                  rows={2}
                  placeholder="Describe what the rater should observe..."
                />
              </div>

              {/* Hint */}
              <div>
                <label className="block text-sm font-medium mb-1 text-foreground">Hint (explanation of correct answer)</label>
                <textarea
                  value={formData.hint}
                  onChange={(e) => setFormData({ ...formData, hint: e.target.value })}
                  className="w-full p-2 border border-border rounded-lg bg-background text-foreground focus:ring-2 focus:ring-primary focus:outline-none"
                  rows={3}
                  placeholder="Explain why this is the correct answer..."
                />
              </div>
            </div>

            <div className="flex justify-end gap-2 mt-6 pt-4 border-t border-border">
              <button
                onClick={() => {
                  setShowCreateModal(false)
                  setShowEditModal(false)
                  setEditingTask(null)
                  resetForm()
                }}
                className="px-4 py-2 border border-border rounded-lg hover:bg-accent text-foreground transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={showCreateModal ? handleCreate : handleUpdate}
                disabled={!formData.video_id_1 || !formData.video_id_2}
                className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50 transition-colors"
              >
                {showCreateModal ? 'Create' : 'Save Changes'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
