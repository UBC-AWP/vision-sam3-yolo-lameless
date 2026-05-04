/**
 * Authentication Context
 * Provides authentication state and methods throughout the app
 */
import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react'
import { jwtDecode } from 'jwt-decode'
import axios from 'axios'

// Types
interface User {
  id: string
  email: string
  username: string
  role: 'admin' | 'researcher' | 'rater'
  is_active: boolean
  rater_tier?: string | null
  created_at: string
  last_login?: string | null
}

interface TokenPayload {
  sub: string
  email: string
  username: string
  role: string
  exp: number
  type: string
}

interface AuthContextType {
  user: User | null
  isAuthenticated: boolean
  isLoading: boolean
  login: (email: string, password: string) => Promise<void>
  loginAsGuest: () => void
  logout: () => Promise<void>
  register: (email: string, username: string, password: string) => Promise<void>
  refreshToken: () => Promise<boolean>
  hasRole: (roles: string[]) => boolean
  getAccessToken: () => string | null
}

interface AuthProviderProps {
  children: ReactNode
}

// API base URL - use relative URLs in production (ALB routes), localhost in development
const API_URL = import.meta.env.VITE_API_URL || (import.meta.env.PROD ? '' : 'http://localhost:8000')

// Token storage keys
const ACCESS_TOKEN_KEY = 'access_token'
const REFRESH_TOKEN_KEY = 'refresh_token'

// Create context
const AuthContext = createContext<AuthContextType | undefined>(undefined)

// Token utilities
const getStoredToken = (key: string): string | null => {
  return localStorage.getItem(key)
}

const setStoredToken = (key: string, token: string): void => {
  localStorage.setItem(key, token)
}

const removeStoredTokens = (): void => {
  localStorage.removeItem(ACCESS_TOKEN_KEY)
  localStorage.removeItem(REFRESH_TOKEN_KEY)
}

const isTokenExpired = (token: string): boolean => {
  try {
    const decoded = jwtDecode<TokenPayload>(token)
    return decoded.exp * 1000 < Date.now()
  } catch {
    return true
  }
}

// Configure axios defaults
const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json'
  }
})

// Auth Provider Component
export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  // Fetch current user
  const fetchUser = useCallback(async (token: string) => {
    try {
      const response = await api.get<User>('/api/auth/me', {
        headers: { Authorization: `Bearer ${token}` }
      })
      setUser(response.data)
      return true
    } catch (error) {
      console.error('Failed to fetch user:', error)
      return false
    }
  }, [])

  // Refresh token
  const refreshToken = useCallback(async (): Promise<boolean> => {
    const refresh = getStoredToken(REFRESH_TOKEN_KEY)
    if (!refresh) return false

    try {
      const response = await api.post<{
        access_token: string
        refresh_token: string
      }>('/api/auth/refresh', { refresh_token: refresh })

      setStoredToken(ACCESS_TOKEN_KEY, response.data.access_token)
      setStoredToken(REFRESH_TOKEN_KEY, response.data.refresh_token)

      await fetchUser(response.data.access_token)
      return true
    } catch (error) {
      console.error('Token refresh failed:', error)
      removeStoredTokens()
      setUser(null)
      return false
    }
  }, [fetchUser])

  // Initialize auth state
  useEffect(() => {
    const initAuth = async () => {
      const token = getStoredToken(ACCESS_TOKEN_KEY)

      if (token && !isTokenExpired(token)) {
        await fetchUser(token)
      } else if (getStoredToken(REFRESH_TOKEN_KEY)) {
        await refreshToken()
      }

      setIsLoading(false)
    }

    initAuth()
  }, [fetchUser, refreshToken])

  // Set up axios interceptor for token refresh
  useEffect(() => {
    const interceptor = api.interceptors.response.use(
      response => response,
      async error => {
        const originalRequest = error.config

        if (error.response?.status === 401 && !originalRequest._retry) {
          originalRequest._retry = true

          const refreshed = await refreshToken()
          if (refreshed) {
            const newToken = getStoredToken(ACCESS_TOKEN_KEY)
            originalRequest.headers.Authorization = `Bearer ${newToken}`
            return api(originalRequest)
          }
        }

        return Promise.reject(error)
      }
    )

    return () => {
      api.interceptors.response.eject(interceptor)
    }
  }, [refreshToken])

  // Login
  const login = async (email: string, password: string): Promise<void> => {
    const response = await api.post<{
      access_token: string
      refresh_token: string
    }>('/api/auth/login', { email, password })

    setStoredToken(ACCESS_TOKEN_KEY, response.data.access_token)
    setStoredToken(REFRESH_TOKEN_KEY, response.data.refresh_token)

    await fetchUser(response.data.access_token)
  }

  // Logout
  const logout = async (): Promise<void> => {
    const token = getStoredToken(ACCESS_TOKEN_KEY)

    try {
      if (token) {
        await api.post('/api/auth/logout', {}, {
          headers: { Authorization: `Bearer ${token}` }
        })
      }
    } catch (error) {
      console.error('Logout request failed:', error)
    } finally {
      removeStoredTokens()
      setUser(null)
    }
  }

  // Guest login – no backend required, purely client-side
  const loginAsGuest = (): void => {
    setUser({
      id: 'guest',
      email: 'guest@demo.local',
      username: 'Guest',
      role: 'rater',
      is_active: true,
      rater_tier: null,
      created_at: new Date().toISOString(),
      last_login: null,
    })
  }

  // Register
  const register = async (email: string, username: string, password: string): Promise<void> => {
    await api.post('/api/auth/register', { email, username, password })
  }

  // Check if user has one of the required roles
  const hasRole = (roles: string[]): boolean => {
    if (!user) return false
    return roles.includes(user.role)
  }

  // Get current access token
  const getAccessToken = (): string | null => {
    return getStoredToken(ACCESS_TOKEN_KEY)
  }

  const value: AuthContextType = {
    user,
    isAuthenticated: !!user,
    isLoading,
    login,
    loginAsGuest,
    logout,
    register,
    refreshToken,
    hasRole,
    getAccessToken
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}

// Hook to use auth context
export function useAuth(): AuthContextType {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

// Export API instance for use in other modules
export { api }
