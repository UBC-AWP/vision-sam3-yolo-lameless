/**
 * Theme Context
 * Provides theme management with multiple beautiful themes
 */
import { createContext, useContext, useEffect, useState, ReactNode } from 'react'

export type ThemeMode = 'light' | 'dark' | 'midnight' | 'ocean' | 'forest' | 'sunset' | 'system'

interface ThemeContextType {
  theme: ThemeMode
  resolvedTheme: Exclude<ThemeMode, 'system'>
  setTheme: (theme: ThemeMode) => void
  themes: { value: ThemeMode; label: string; icon: string; preview: string[] }[]
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined)

const THEME_STORAGE_KEY = 'lameness-theme'

// Theme definitions with preview colors
export const themes: ThemeContextType['themes'] = [
  { value: 'light', label: 'Light', icon: '☀️', preview: ['#ffffff', '#f8fafc', '#0f172a'] },
  { value: 'dark', label: 'Dark', icon: '🌙', preview: ['#0f172a', '#1e293b', '#f8fafc'] },
  { value: 'midnight', label: 'Midnight', icon: '🌌', preview: ['#020617', '#0f172a', '#a855f7'] },
  { value: 'ocean', label: 'Ocean', icon: '🌊', preview: ['#0c1929', '#0f2744', '#22d3ee'] },
  { value: 'forest', label: 'Forest', icon: '🌲', preview: ['#0d1f0d', '#1a3a1a', '#22c55e'] },
  { value: 'sunset', label: 'Sunset', icon: '🌅', preview: ['#1c1117', '#2d1b24', '#f97316'] },
  { value: 'system', label: 'System', icon: '💻', preview: ['#94a3b8', '#64748b', '#475569'] },
]

function getSystemTheme(): 'light' | 'dark' {
  if (typeof window === 'undefined') return 'dark'
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<ThemeMode>(() => {
    if (typeof window === 'undefined') return 'dark'
    return (localStorage.getItem(THEME_STORAGE_KEY) as ThemeMode) || 'dark'
  })

  const [resolvedTheme, setResolvedTheme] = useState<Exclude<ThemeMode, 'system'>>(() => {
    const stored = typeof window !== 'undefined' 
      ? localStorage.getItem(THEME_STORAGE_KEY) as ThemeMode 
      : 'dark'
    return stored === 'system' ? getSystemTheme() : (stored || 'dark') as Exclude<ThemeMode, 'system'>
  })

  const setTheme = (newTheme: ThemeMode) => {
    setThemeState(newTheme)
    localStorage.setItem(THEME_STORAGE_KEY, newTheme)
  }

  useEffect(() => {
    const resolved = theme === 'system' ? getSystemTheme() : theme
    setResolvedTheme(resolved as Exclude<ThemeMode, 'system'>)

    // Remove all theme classes
    const root = document.documentElement
    root.classList.remove('light', 'dark', 'midnight', 'ocean', 'forest', 'sunset')
    
    // Add the resolved theme class
    root.classList.add(resolved)
    
    // Also set the color-scheme for browser native elements
    root.style.colorScheme = ['light'].includes(resolved) ? 'light' : 'dark'
  }, [theme])

  // Listen for system theme changes
  useEffect(() => {
    if (theme !== 'system') return

    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
    const handleChange = () => {
      setResolvedTheme(getSystemTheme())
      document.documentElement.classList.remove('light', 'dark')
      document.documentElement.classList.add(getSystemTheme())
    }

    mediaQuery.addEventListener('change', handleChange)
    return () => mediaQuery.removeEventListener('change', handleChange)
  }, [theme])

  return (
    <ThemeContext.Provider value={{ theme, resolvedTheme, setTheme, themes }}>
      {children}
    </ThemeContext.Provider>
  )
}

export function useTheme() {
  const context = useContext(ThemeContext)
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider')
  }
  return context
}

