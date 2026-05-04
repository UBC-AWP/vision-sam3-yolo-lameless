/**
 * Theme Context
 * Provides theme management with multiple beautiful themes
 */
import { createContext, useContext, useEffect, useState, ReactNode } from 'react'

export type ThemeMode = 'light' | 'dark'

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
]


export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<ThemeMode>(() => {
    if (typeof window === 'undefined') return 'dark'
    return (localStorage.getItem(THEME_STORAGE_KEY) as ThemeMode) || 'dark'
  })

  const [resolvedTheme, setResolvedTheme] = useState<ThemeMode>(() => {
    const stored = typeof window !== 'undefined' 
      ? localStorage.getItem(THEME_STORAGE_KEY) as ThemeMode 
      : 'dark'
    return (stored || 'dark') as ThemeMode
  })

  const setTheme = (newTheme: ThemeMode) => {
    setThemeState(newTheme)
    localStorage.setItem(THEME_STORAGE_KEY, newTheme)
  }

  useEffect(() => {
    setResolvedTheme(theme)

    // Remove all theme classes
    const root = document.documentElement
    root.classList.remove('light', 'dark')
    
    // Add the theme class
    root.classList.add(theme)
    
    // Also set the color-scheme for browser native elements
    root.style.colorScheme = theme === 'light' ? 'light' : 'dark'
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

