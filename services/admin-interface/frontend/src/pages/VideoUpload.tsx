import { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { videosApi, trainingApi } from '@/api/client'

interface FileItem {
  id: string
  file: File
  label: number | null // null = not set, 0 = sound, 1 = lame
  status: 'pending' | 'uploading' | 'success' | 'error'
  progress: number
  videoId?: string
  error?: string
}

export default function VideoUpload() {
  const [files, setFiles] = useState<FileItem[]>([])
  const [uploading, setUploading] = useState(false)
  const [dragActive, setDragActive] = useState(false)
  const [urlInput, setUrlInput] = useState('')
  const [urlStatus, setUrlStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle')
  const [urlVideoId, setUrlVideoId] = useState<string | null>(null)
  const [urlError, setUrlError] = useState<string | null>(null)
  const navigate = useNavigate()

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      addFiles(Array.from(e.target.files))
    }
    e.target.value = ''
  }

  const handleFolderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      addFiles(Array.from(e.target.files))
    }
    e.target.value = ''
  }

  const VIDEO_EXT = ['mp4', 'avi', 'mov', 'mkv']

  const addFiles = (newFiles: File[]) => {
    const fileItems: FileItem[] = newFiles
      .filter(f => {
        const ext = f.name.split('.').pop()?.toLowerCase()
        return VIDEO_EXT.includes(ext || '')
      })
      .map(file => ({
        id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        file,
        label: null,
        status: 'pending' as const,
        progress: 0,
      }))
    setFiles(prev => [...prev, ...fileItems])
  }

  // Recursively collect video files from a directory (used when user drops a folder)
  const collectFilesFromDirectory = useCallback(
    (entry: FileSystemDirectoryEntry): Promise<File[]> =>
      new Promise((resolve, reject) => {
        const files: File[] = []
        const reader = entry.createReader()

        const readBatch = () => {
          reader.readEntries(
            (entries: FileSystemEntry[]) => {
              if (entries.length === 0) {
                resolve(files)
                return
              }
              let pending = entries.length
              entries.forEach((e) => {
                if (e.isFile) {
                  (e as FileSystemFileEntry).file(
                    (file: File) => {
                      const ext = file.name.split('.').pop()?.toLowerCase()
                      if (VIDEO_EXT.includes(ext || '')) files.push(file)
                      if (--pending === 0) readBatch()
                    },
                    () => {
                      if (--pending === 0) readBatch()
                    }
                  )
                } else {
                  collectFilesFromDirectory(e as FileSystemDirectoryEntry).then((sub) => {
                    files.push(...sub)
                    if (--pending === 0) readBatch()
                  }).catch(() => {
                    if (--pending === 0) readBatch()
                  })
                }
              })
            },
            (err) => reject(err)
          )
        }
        readBatch()
      }),
    []
  )

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }, [])

  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault()
      e.stopPropagation()
      setDragActive(false)

      const items = e.dataTransfer?.items
      if (!items || items.length === 0) return

      const collected: File[] = []
      const dirPromises: Promise<File[]>[] = []

      for (let i = 0; i < items.length; i++) {
        const item = items[i]
        if (item.kind !== 'file') continue
        const entry = (item as DataTransferItem & { webkitGetAsEntry?: () => FileSystemEntry | null }).webkitGetAsEntry?.()
        if (!entry) {
          const file = item.getAsFile()
          if (file) collected.push(file)
          continue
        }
        if (entry.isDirectory) {
          dirPromises.push(collectFilesFromDirectory(entry as FileSystemDirectoryEntry))
        } else if (entry.isFile) {
          const file = await new Promise<File | null>((res) =>
            (entry as FileSystemFileEntry).file(res, () => res(null))
          )
          if (file) collected.push(file)
        }
      }

      const dirFiles = await Promise.all(dirPromises)
      dirFiles.forEach((f) => collected.push(...f))
      if (collected.length > 0) addFiles(collected)
    },
    [collectFilesFromDirectory]
  )

  const setLabel = (fileId: string, label: number) => {
    setFiles(prev => prev.map(f => 
      f.id === fileId ? { ...f, label } : f
    ))
  }

  const removeFile = (fileId: string) => {
    setFiles(prev => prev.filter(f => f.id !== fileId))
  }

  const labelAllAs = (label: number) => {
    setFiles(prev => prev.map(f => 
      f.status === 'pending' ? { ...f, label } : f
    ))
  }

  const uploadFile = async (fileItem: FileItem): Promise<FileItem> => {
    try {
      // Upload the video
      const result = await videosApi.upload(fileItem.file)
      
      // If label is set, save it
      if (fileItem.label !== null) {
        await trainingApi.label(result.video_id, fileItem.label)
      }
      
      return {
        ...fileItem,
        status: 'success',
        progress: 100,
        videoId: result.video_id,
      }
    } catch (err: any) {
      return {
        ...fileItem,
        status: 'error',
        error: err.response?.data?.detail || 'Upload failed',
      }
    }
  }

  const handleUploadAll = async () => {
    const pendingFiles = files.filter(f => f.status === 'pending')
    if (pendingFiles.length === 0) return

    setUploading(true)

    // Upload files sequentially to avoid overwhelming the server
    for (const fileItem of pendingFiles) {
      setFiles(prev => prev.map(f => 
        f.id === fileItem.id ? { ...f, status: 'uploading', progress: 50 } : f
      ))

      const result = await uploadFile(fileItem)
      
      setFiles(prev => prev.map(f => 
        f.id === fileItem.id ? result : f
      ))
    }

    setUploading(false)
  }

  const handleImportUrl = async () => {
    if (!urlInput.trim() || urlStatus === 'loading') return
    setUrlStatus('loading')
    setUrlError(null)
    setUrlVideoId(null)
    try {
      const result = await videosApi.importFromUrl(urlInput.trim())
      setUrlStatus('success')
      setUrlVideoId(result.video_id)
    } catch (err: any) {
      setUrlStatus('error')
      setUrlError(err.response?.data?.detail || 'Import failed')
    }
  }

  const successCount = files.filter(f => f.status === 'success').length
  const errorCount = files.filter(f => f.status === 'error').length
  const pendingCount = files.filter(f => f.status === 'pending').length
  const labeledCount = files.filter(f => f.label !== null && f.status === 'pending').length

  return (
    <div className="max-w-4xl mx-auto">
      <h2 className="text-3xl font-bold mb-6">Upload Videos</h2>

      {/* Drop zone */}
      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
          dragActive ? 'border-primary bg-primary/5' : 'border-muted-foreground/25'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <div className="mb-4">
          <svg className="mx-auto h-12 w-12 text-muted-foreground" stroke="currentColor" fill="none" viewBox="0 0 48 48">
            <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
        <p className="text-muted-foreground mb-2">
          Drag and drop video files or a folder of videos here, or click to select
        </p>
        <div className="flex flex-wrap items-center justify-center gap-2">
          <input
            type="file"
            accept="video/*"
            multiple
            onChange={handleFileChange}
            className="hidden"
            id="file-input"
            disabled={uploading}
          />
          <label
            htmlFor="file-input"
            className="inline-block px-4 py-2 bg-secondary text-secondary-foreground rounded-md cursor-pointer hover:bg-secondary/80"
          >
            Select Files
          </label>
          <input
            type="file"
            id="folder-input"
            className="hidden"
            disabled={uploading}
            multiple
            onChange={handleFolderChange}
            ref={(el) => {
              if (el) (el as HTMLInputElement & { webkitdirectory?: boolean }).webkitdirectory = true
            }}
          />
          <label
            htmlFor="folder-input"
            className="inline-block px-4 py-2 bg-secondary text-secondary-foreground rounded-md cursor-pointer hover:bg-secondary/80"
          >
            Select Folder
          </label>
        </div>
        <p className="text-xs text-muted-foreground mt-2">
          Supported formats: MP4, AVI, MOV, MKV. Folder: all videos inside (including subfolders) are added.
        </p>
      </div>

      {/* URL import */}
      <div className="mt-8 border rounded-lg p-6 bg-background">
        <h3 className="text-xl font-semibold mb-3">Import Video From URL</h3>
        <div className="flex flex-col gap-3">
          <input
            type="url"
            placeholder="https://.../your_video.mp4"
            value={urlInput}
            onChange={(e) => setUrlInput(e.target.value)}
            className="w-full px-3 py-2 border rounded-md"
            disabled={urlStatus === 'loading'}
          />
          <div className="flex items-center gap-3">
            <button
              onClick={handleImportUrl}
              className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 disabled:opacity-50"
              disabled={!urlInput.trim() || urlStatus === 'loading'}
            >
              {urlStatus === 'loading' ? 'Importing...' : 'Import URL'}
            </button>
            {urlStatus === 'success' && urlVideoId && (
              <button
                onClick={() => navigate(`/video/${urlVideoId}`)}
                className="px-3 py-2 text-sm bg-primary text-primary-foreground rounded hover:bg-primary/90"
              >
                View
              </button>
            )}
            {urlStatus === 'error' && (
              <span className="text-sm text-red-600">{urlError}</span>
            )}
            {urlStatus === 'success' && urlVideoId && (
              <span className="text-sm text-green-600">Imported</span>
            )}
          </div>
        </div>
      </div>

      {/* File list */}
      {files.length > 0 && (
        <div className="mt-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-xl font-semibold">
              Selected Files ({files.length})
            </h3>
            <div className="flex gap-2">
              <button
                onClick={() => labelAllAs(0)}
                disabled={uploading || pendingCount === 0}
                className="px-3 py-1 text-sm bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
              >
                Label All Sound
              </button>
              <button
                onClick={() => labelAllAs(1)}
                disabled={uploading || pendingCount === 0}
                className="px-3 py-1 text-sm bg-red-600 text-white rounded hover:bg-red-700 disabled:opacity-50"
              >
                Label All Lame
              </button>
              <button
                onClick={() => setFiles([])}
                disabled={uploading}
                className="px-3 py-1 text-sm bg-secondary text-secondary-foreground rounded hover:bg-secondary/80 disabled:opacity-50"
              >
                Clear All
              </button>
            </div>
          </div>

          <div className="space-y-2 max-h-96 overflow-y-auto">
            {files.map((fileItem) => (
              <div
                key={fileItem.id}
                className={`flex items-center gap-4 p-3 border rounded-lg ${
                  fileItem.status === 'success' ? 'bg-green-50 border-green-200' :
                  fileItem.status === 'error' ? 'bg-red-50 border-red-200' :
                  fileItem.status === 'uploading' ? 'bg-blue-50 border-blue-200' :
                  'bg-background'
                }`}
              >
                {/* File info */}
                <div className="flex-1 min-w-0">
                  <p className="font-medium truncate">{fileItem.file.name}</p>
                  <p className="text-sm text-muted-foreground">
                    {(fileItem.file.size / 1024 / 1024).toFixed(2)} MB
                    {fileItem.status === 'success' && fileItem.videoId && (
                      <span className="ml-2 text-green-600">✓ Uploaded</span>
                    )}
                    {fileItem.status === 'error' && (
                      <span className="ml-2 text-red-600">✗ {fileItem.error}</span>
                    )}
                    {fileItem.status === 'uploading' && (
                      <span className="ml-2 text-blue-600">Uploading...</span>
                    )}
                  </p>
                </div>

                {/* Label buttons */}
                {fileItem.status === 'pending' && (
                  <div className="flex gap-1">
                    <button
                      onClick={() => setLabel(fileItem.id, 0)}
                      className={`px-3 py-1 text-sm rounded transition-colors ${
                        fileItem.label === 0
                          ? 'bg-green-600 text-white'
                          : 'bg-green-100 text-green-700 hover:bg-green-200'
                      }`}
                    >
                      Sound
                    </button>
                    <button
                      onClick={() => setLabel(fileItem.id, 1)}
                      className={`px-3 py-1 text-sm rounded transition-colors ${
                        fileItem.label === 1
                          ? 'bg-red-600 text-white'
                          : 'bg-red-100 text-red-700 hover:bg-red-200'
                      }`}
                    >
                      Lame
                    </button>
                  </div>
                )}

                {/* Status indicator */}
                {fileItem.status === 'success' && (
                  <button
                    onClick={() => navigate(`/video/${fileItem.videoId}`)}
                    className="px-3 py-1 text-sm bg-primary text-primary-foreground rounded hover:bg-primary/90"
                  >
                    View
                  </button>
                )}

                {/* Remove button */}
                {fileItem.status === 'pending' && (
                  <button
                    onClick={() => removeFile(fileItem.id)}
                    className="p-1 text-muted-foreground hover:text-foreground"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                )}
              </div>
            ))}
          </div>

          {/* Summary and upload button */}
          <div className="mt-4 flex justify-between items-center">
            <div className="text-sm text-muted-foreground">
              {pendingCount > 0 && (
                <span>
                  {labeledCount}/{pendingCount} labeled
                  {labeledCount < pendingCount && ' (unlabeled files will be uploaded without labels)'}
                </span>
              )}
              {successCount > 0 && <span className="ml-4 text-green-600">{successCount} uploaded</span>}
              {errorCount > 0 && <span className="ml-4 text-red-600">{errorCount} failed</span>}
            </div>
            <button
              onClick={handleUploadAll}
              disabled={pendingCount === 0 || uploading}
              className="px-6 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 disabled:opacity-50"
            >
              {uploading ? 'Uploading...' : `Upload ${pendingCount} File${pendingCount !== 1 ? 's' : ''}`}
            </button>
          </div>
        </div>
      )}

      {/* Quick stats */}
      {successCount > 0 && (
        <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
          <p className="text-green-800">
            Successfully uploaded {successCount} video{successCount !== 1 ? 's' : ''}.{' '}
            <button
              onClick={() => navigate('/')}
              className="underline hover:no-underline"
            >
              Go to Dashboard
            </button>
          </p>
        </div>
      )}
    </div>
  )
}
