/**
 * WebSocket Hook
 * Provides real-time connection to backend WebSocket channels
 */
import { useState, useEffect, useRef, useCallback } from 'react'

type WebSocketStatus = 'connecting' | 'connected' | 'disconnected' | 'error'

interface WebSocketMessage {
  type: string
  timestamp?: string
  [key: string]: unknown
}

interface UseWebSocketOptions {
  onMessage?: (message: WebSocketMessage) => void
  onConnect?: () => void
  onDisconnect?: () => void
  onError?: (error: Event) => void
  reconnectAttempts?: number
  reconnectInterval?: number
  autoConnect?: boolean
}

interface UseWebSocketReturn {
  status: WebSocketStatus
  lastMessage: WebSocketMessage | null
  connect: () => void
  disconnect: () => void
  send: (message: object) => void
  isConnected: boolean
}

const WS_BASE_URL = import.meta.env.VITE_WS_URL ||
  (window.location.protocol === 'https:' ? 'wss://' : 'ws://') +
  (import.meta.env.VITE_API_URL?.replace(/^https?:\/\//, '') || 'localhost:8000')

export function useWebSocket(
  channel: 'pipeline' | 'health' | 'queue' | 'rater',
  options: UseWebSocketOptions = {}
): UseWebSocketReturn {
  const {
    onMessage,
    onConnect,
    onDisconnect,
    onError,
    reconnectAttempts = 3,
    reconnectInterval = 5000,
    autoConnect = true
  } = options

  const [status, setStatus] = useState<WebSocketStatus>('disconnected')
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null)

  const wsRef = useRef<WebSocket | null>(null)
  const reconnectCountRef = useRef(0)
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const isConnectingRef = useRef(false)
  const isMountedRef = useRef(true)

  // Store callbacks in refs to avoid useEffect re-runs
  const onMessageRef = useRef(onMessage)
  const onConnectRef = useRef(onConnect)
  const onDisconnectRef = useRef(onDisconnect)
  const onErrorRef = useRef(onError)

  // Update refs when callbacks change
  useEffect(() => {
    onMessageRef.current = onMessage
    onConnectRef.current = onConnect
    onDisconnectRef.current = onDisconnect
    onErrorRef.current = onError
  }, [onMessage, onConnect, onDisconnect, onError])

  const clearTimers = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current)
      pingIntervalRef.current = null
    }
  }, [])

  const disconnect = useCallback(() => {
    clearTimers()
    isConnectingRef.current = false
    if (wsRef.current) {
      wsRef.current.onclose = null // Prevent reconnection on intentional close
      wsRef.current.close()
      wsRef.current = null
    }
    if (isMountedRef.current) {
      setStatus('disconnected')
    }
  }, [clearTimers])

  const connect = useCallback(() => {
    // Prevent multiple simultaneous connection attempts
    if (isConnectingRef.current || wsRef.current?.readyState === WebSocket.OPEN || wsRef.current?.readyState === WebSocket.CONNECTING) {
      return
    }

    isConnectingRef.current = true
    clearTimers()
    setStatus('connecting')

    try {
      const wsUrl = `${WS_BASE_URL}/api/ws/${channel}`
      wsRef.current = new WebSocket(wsUrl)

      wsRef.current.onopen = () => {
        isConnectingRef.current = false
        if (isMountedRef.current) {
          setStatus('connected')
          reconnectCountRef.current = 0
          onConnectRef.current?.()

          // Set up ping interval
          pingIntervalRef.current = setInterval(() => {
            if (wsRef.current?.readyState === WebSocket.OPEN) {
              wsRef.current.send('ping')
            }
          }, 25000)
        }
      }

      wsRef.current.onmessage = (event) => {
        try {
          // Handle pong response
          if (event.data === 'pong') {
            return
          }

          const message: WebSocketMessage = JSON.parse(event.data)

          // Handle ping from server
          if (message.type === 'ping') {
            wsRef.current?.send('pong')
            return
          }

          if (isMountedRef.current) {
            setLastMessage(message)
            onMessageRef.current?.(message)
          }
        } catch {
          // Silently ignore parse errors for non-JSON messages
        }
      }

      wsRef.current.onclose = () => {
        isConnectingRef.current = false
        if (isMountedRef.current) {
          setStatus('disconnected')
          onDisconnectRef.current?.()
        }
        clearTimers()

        // Attempt reconnection with exponential backoff
        if (isMountedRef.current && reconnectCountRef.current < reconnectAttempts) {
          reconnectCountRef.current += 1
          const backoff = reconnectInterval * Math.pow(2, reconnectCountRef.current - 1)
          reconnectTimeoutRef.current = setTimeout(() => {
            if (isMountedRef.current) {
              connect()
            }
          }, Math.min(backoff, 30000)) // Cap at 30 seconds
        }
      }

      wsRef.current.onerror = () => {
        isConnectingRef.current = false
        if (isMountedRef.current) {
          setStatus('error')
          onErrorRef.current?.(new Event('error'))
        }
      }
    } catch {
      isConnectingRef.current = false
      if (isMountedRef.current) {
        setStatus('error')
      }
    }
  }, [channel, reconnectAttempts, reconnectInterval, clearTimers])

  const send = useCallback((message: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message))
    }
  }, [])

  // Auto-connect on mount - only depends on channel and autoConnect
  useEffect(() => {
    isMountedRef.current = true
    
    if (autoConnect) {
      // Small delay to prevent rapid reconnection on hot reload
      const timeoutId = setTimeout(() => {
        if (isMountedRef.current) {
          connect()
        }
      }, 100)
      
      return () => {
        clearTimeout(timeoutId)
        isMountedRef.current = false
        disconnect()
      }
    }

    return () => {
      isMountedRef.current = false
      disconnect()
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [channel, autoConnect])

  return {
    status,
    lastMessage,
    connect,
    disconnect,
    send,
    isConnected: status === 'connected'
  }
}

// Convenience hooks for specific channels
export function usePipelineWebSocket(options?: UseWebSocketOptions) {
  return useWebSocket('pipeline', options)
}

export function useHealthWebSocket(options?: UseWebSocketOptions) {
  return useWebSocket('health', options)
}

export function useQueueWebSocket(options?: UseWebSocketOptions) {
  return useWebSocket('queue', options)
}

export function useRaterWebSocket(options?: UseWebSocketOptions) {
  return useWebSocket('rater', options)
}
