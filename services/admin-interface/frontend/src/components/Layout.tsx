/**
 * Premium Layout Component
 * Modern glassmorphism design with animated sidebar
 */
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { useState, useEffect, useRef } from 'react'
import { cn } from '@/lib/utils'
import { useAuth } from '../contexts/AuthContext'
import { useTheme } from '../contexts/ThemeContext'
import {
  User,
  LogOut,
  Settings,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Shield,
  FlaskConical,
  UserCircle,
  Loader2,
  Home,
  Upload,
  GitCompare,
  Triangle,
  Network,
  Map,
  GraduationCap,
  Activity,
  Heart,
  ListTodo,
  Cpu,
  Users,
  BookOpen,
  Menu,
  X,
  Sparkles,
  Beef,
  Sun,
  Moon,
  Palette,
  Bell,
  Search,
  Command
} from 'lucide-react'

interface LayoutProps {
  children: React.ReactNode
}

export default function Layout({ children }: LayoutProps) {
  const location = useLocation()
  const navigate = useNavigate()
  const { user, isAuthenticated, isLoading, logout, hasRole } = useAuth()
  const { theme, setTheme, themes, resolvedTheme } = useTheme()
  const [isUserMenuOpen, setIsUserMenuOpen] = useState(false)
  const [isThemeMenuOpen, setIsThemeMenuOpen] = useState(false)
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(() => {
    const saved = localStorage.getItem('sidebar_collapsed')
    return saved === 'true'
  })
  const [searchQuery, setSearchQuery] = useState('')
  const [isSearchOpen, setIsSearchOpen] = useState(false)
  const sidebarRef = useRef<HTMLElement>(null)

  useEffect(() => {
    localStorage.setItem('sidebar_collapsed', String(isSidebarCollapsed))
  }, [isSidebarCollapsed])

  // Close mobile menu on route change
  useEffect(() => {
    setIsMobileMenuOpen(false)
  }, [location.pathname])

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Cmd/Ctrl + K for search
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        setIsSearchOpen(true)
      }
      // Escape to close modals
      if (e.key === 'Escape') {
        setIsSearchOpen(false)
        setIsUserMenuOpen(false)
        setIsThemeMenuOpen(false)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  const navItems = [
    { path: '/', label: 'Dashboard', icon: Home, roles: ['admin', 'researcher', 'rater'] },
    { path: '/upload', label: 'Upload', icon: Upload, roles: ['admin', 'researcher'] },
    { path: '/cows', label: 'Cow Registry', icon: Beef, roles: ['admin', 'researcher', 'rater'] },
    { path: '/pairwise', label: 'Pairwise', icon: GitCompare, roles: ['admin', 'researcher', 'rater'] },
    { path: '/triplet', label: 'Triplet', icon: Triangle, roles: ['admin', 'researcher', 'rater'] },
    { path: '/hierarchy', label: 'Hierarchy', icon: Network, roles: ['admin', 'researcher', 'rater'] },
    { path: '/similarity', label: 'Similarity', icon: Map, roles: ['admin', 'researcher', 'rater'] },
    { path: '/learn', label: 'Learn', icon: GraduationCap, roles: ['admin', 'researcher', 'rater'] },
    { path: '/pipelines', label: 'Pipelines', icon: Activity, roles: ['admin', 'researcher'] },
    { path: '/health', label: 'Health', icon: Heart, roles: ['admin', 'researcher'] },
    { path: '/training', label: 'Queue', icon: ListTodo, roles: ['admin', 'researcher'] },
    { path: '/config', label: 'ML Config', icon: Cpu, roles: ['admin', 'researcher'] },
    { path: '/users', label: 'Users', icon: Users, roles: ['admin'] },
    { path: '/tutorials', label: 'Tutorials', icon: BookOpen, roles: ['admin'] },
  ]

  const handleLogout = async () => {
    await logout()
    navigate('/login')
  }

  const getRoleIcon = (role: string) => {
    switch (role) {
      case 'admin':
        return <Shield className="h-4 w-4 text-rose-500" />
      case 'researcher':
        return <FlaskConical className="h-4 w-4 text-blue-500" />
      default:
        return <UserCircle className="h-4 w-4 text-emerald-500" />
    }
  }

  const getRoleBadgeStyles = (role: string) => {
    switch (role) {
      case 'admin':
        return 'bg-rose-500/15 text-rose-500 border-rose-500/30'
      case 'researcher':
        return 'bg-blue-500/15 text-blue-500 border-blue-500/30'
      default:
        return 'bg-emerald-500/15 text-emerald-500 border-emerald-500/30'
    }
  }

  const getThemeIcon = () => {
    if (theme === 'system') return <Palette className="h-4 w-4" />
    if (resolvedTheme === 'light') return <Sun className="h-4 w-4" />
    return <Moon className="h-4 w-4" />
  }

  // Show loading state
  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-4 animate-fade-in">
          <div className="relative">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary to-primary/60 flex items-center justify-center animate-pulse-soft">
              <Sparkles className="h-8 w-8 text-primary-foreground" />
            </div>
            <div className="absolute -inset-2 bg-primary/20 rounded-3xl blur-xl animate-pulse-soft" />
          </div>
          <div className="text-center">
            <p className="text-lg font-medium">Loading...</p>
            <p className="text-sm text-muted-foreground">Preparing your workspace</p>
          </div>
        </div>
      </div>
    )
  }

  // Redirect to login if not authenticated
  if (!isAuthenticated) {
    navigate('/login')
    return null
  }

  return (
    <div className="min-h-screen bg-background flex">
      {/* Ambient background effects */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-1/2 -left-1/2 w-full h-full bg-gradient-radial from-primary/5 to-transparent opacity-50" />
        <div className="absolute -bottom-1/2 -right-1/2 w-full h-full bg-gradient-radial from-accent/5 to-transparent opacity-50" />
      </div>

      {/* Mobile menu backdrop */}
      {isMobileMenuOpen && (
        <div
          className="fixed inset-0 bg-background/80 backdrop-blur-sm z-40 lg:hidden animate-fade-in"
          onClick={() => setIsMobileMenuOpen(false)}
        />
      )}

      {/* Sidebar Navigation */}
      <aside
        ref={sidebarRef}
        className={cn(
          "fixed left-0 top-0 h-full z-50 transition-all duration-300 ease-out flex flex-col",
          "glass-card border-r border-border/50",
          isSidebarCollapsed ? "w-[72px]" : "w-64",
          // Mobile styles
          "lg:translate-x-0",
          isMobileMenuOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"
        )}
      >
        {/* Logo/Title */}
        <div className="p-4 border-b border-border/50">
          <Link 
            to="/" 
            className={cn(
              "flex items-center gap-3 group",
              isSidebarCollapsed && "justify-center"
            )}
          >
            <div className="relative">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-primary/60 flex items-center justify-center flex-shrink-0 shadow-lg shadow-primary/20 group-hover:shadow-primary/40 transition-shadow">
                <Sparkles className="h-5 w-5 text-primary-foreground" />
              </div>
              <div className="absolute inset-0 rounded-xl bg-primary/20 blur-lg opacity-0 group-hover:opacity-100 transition-opacity" />
            </div>
            {!isSidebarCollapsed && (
              <div className="overflow-hidden animate-fade-in">
                <h1 className="text-base font-bold leading-tight truncate gradient-text">
                  CowHealth AI
                </h1>
                <p className="text-xs text-muted-foreground truncate">
                  Research Pipeline
                </p>
              </div>
            )}
          </Link>
        </div>

        {/* Navigation Items */}
        <nav className="flex-1 overflow-y-auto p-3 space-y-1 scrollbar-hide">
          {navItems
            .filter(item => hasRole(item.roles))
            .map((item, index) => {
              const Icon = item.icon
              const isActive = location.pathname === item.path
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={cn(
                    'nav-item',
                    isActive && 'active',
                    isSidebarCollapsed && 'justify-center px-2',
                    `stagger-${Math.min(index + 1, 6)}`
                  )}
                  style={{ animationFillMode: 'backwards' }}
                  title={isSidebarCollapsed ? item.label : undefined}
                >
                  <Icon className="h-[18px] w-[18px] flex-shrink-0" />
                  {!isSidebarCollapsed && (
                    <span className="truncate">{item.label}</span>
                  )}
                </Link>
              )
            })}
        </nav>

        {/* Collapse Toggle */}
        <div className="p-3 border-t border-border/50 hidden lg:block">
          <button
            onClick={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
            className={cn(
              "w-full flex items-center justify-center gap-2 px-3 py-2.5 rounded-xl",
              "text-sm text-muted-foreground hover:text-foreground",
              "hover:bg-accent/50 transition-all duration-200"
            )}
          >
            {isSidebarCollapsed ? (
              <ChevronRight className="h-4 w-4" />
            ) : (
              <>
                <ChevronLeft className="h-4 w-4" />
                <span>Collapse</span>
              </>
            )}
          </button>
        </div>

        {/* User Section */}
        <div className="p-3 border-t border-border/50">
          <div className="relative">
            <button
              onClick={() => setIsUserMenuOpen(!isUserMenuOpen)}
              className={cn(
                "w-full flex items-center gap-3 px-3 py-2.5 rounded-xl",
                "hover:bg-accent/50 transition-all duration-200",
                isSidebarCollapsed && "justify-center px-2"
              )}
            >
              <div className="relative">
                <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center flex-shrink-0 border border-border/50">
                  <User className="h-4 w-4 text-primary" />
                </div>
                {/* Online indicator */}
                <div className="absolute -bottom-0.5 -right-0.5 w-3 h-3 bg-emerald-500 rounded-full border-2 border-card" />
              </div>
              {!isSidebarCollapsed && (
                <>
                  <div className="text-left flex-1 overflow-hidden">
                    <p className="text-sm font-medium truncate">{user?.username}</p>
                    <p className="text-xs text-muted-foreground capitalize truncate">{user?.role}</p>
                  </div>
                  <ChevronDown className={cn(
                    "h-4 w-4 text-muted-foreground transition-transform duration-200 flex-shrink-0",
                    isUserMenuOpen && "rotate-180"
                  )} />
                </>
              )}
            </button>

            {/* User Dropdown Menu */}
            {isUserMenuOpen && (
              <>
                <div
                  className="fixed inset-0 z-[60]"
                  onClick={() => setIsUserMenuOpen(false)}
                />
                <div className={cn(
                  "absolute z-[61] bg-card rounded-xl shadow-xl border border-border/50 animate-scale-in overflow-hidden",
                  isSidebarCollapsed 
                    ? "left-full ml-2 bottom-0 w-72" 
                    : "left-0 right-0 bottom-full mb-2"
                )}>
                  {/* User info header */}
                  <div className="p-4 bg-gradient-to-br from-primary/10 to-accent/10">
                    <div className="flex items-center gap-3">
                      <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center border border-border/50">
                        <User className="h-6 w-6 text-primary" />
                      </div>
                      <div className="overflow-hidden flex-1">
                        <p className="font-semibold truncate">{user?.username}</p>
                        <p className="text-xs text-muted-foreground truncate">{user?.email}</p>
                      </div>
                    </div>
                    <div className="mt-3 flex items-center gap-2 flex-wrap">
                      {user && getRoleIcon(user.role)}
                      <span className={cn(
                        "text-xs px-2 py-1 rounded-lg font-medium capitalize border",
                        user && getRoleBadgeStyles(user.role)
                      )}>
                        {user?.role}
                      </span>
                      {user?.rater_tier && (
                        <span className="text-xs px-2 py-1 rounded-lg font-medium capitalize bg-amber-500/15 text-amber-500 border border-amber-500/30">
                          {user.rater_tier} Tier
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Menu items */}
                  <div className="p-2">
                    <button
                      onClick={() => {
                        setIsUserMenuOpen(false)
                        navigate('/settings')
                      }}
                      className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-accent/50 transition-colors text-sm"
                    >
                      <Settings className="h-4 w-4 text-muted-foreground" />
                      <span>Settings</span>
                    </button>
                    <button
                      onClick={() => {
                        setIsUserMenuOpen(false)
                        handleLogout()
                      }}
                      className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-destructive/10 text-destructive transition-colors text-sm"
                    >
                      <LogOut className="h-4 w-4" />
                      <span>Sign out</span>
                    </button>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className={cn(
        "flex-1 flex flex-col min-h-screen transition-all duration-300",
        isSidebarCollapsed ? "lg:ml-[72px]" : "lg:ml-64"
      )}>
        {/* Top Header Bar */}
        <header className="h-16 border-b border-border/50 glass-card flex items-center px-4 lg:px-6 sticky top-0 z-30">
          {/* Mobile menu button */}
          <button
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            className="p-2 rounded-xl hover:bg-accent/50 transition-colors lg:hidden mr-2"
          >
            {isMobileMenuOpen ? (
              <X className="h-5 w-5" />
            ) : (
              <Menu className="h-5 w-5" />
            )}
          </button>

          {/* Search bar */}
          <div className="flex-1 max-w-md hidden sm:block">
            <button
              onClick={() => setIsSearchOpen(true)}
              className="w-full flex items-center gap-3 px-4 py-2 rounded-xl bg-muted/50 hover:bg-muted text-muted-foreground text-sm transition-colors"
            >
              <Search className="h-4 w-4" />
              <span>Search...</span>
              <kbd className="ml-auto hidden md:inline-flex h-5 select-none items-center gap-1 rounded border border-border bg-background px-1.5 font-mono text-[10px] font-medium text-muted-foreground">
                <span className="text-xs">⌘</span>K
              </kbd>
            </button>
          </div>

          <div className="flex-1 sm:hidden" />

          {/* Right side actions */}
          <div className="flex items-center gap-2">
            {/* Theme toggle */}
            <div className="relative">
              <button
                onClick={() => setIsThemeMenuOpen(!isThemeMenuOpen)}
                className="p-2.5 rounded-xl hover:bg-accent/50 transition-colors"
                title="Change theme"
              >
                {getThemeIcon()}
              </button>

              {isThemeMenuOpen && (
                <>
                  {/* Backdrop */}
                  <div
                    className="fixed inset-0 z-[100]"
                    onClick={() => setIsThemeMenuOpen(false)}
                  />
                  {/* Dropdown menu */}
                  <div className="absolute right-0 top-full mt-2 w-52 bg-card rounded-xl shadow-2xl border border-border z-[101] animate-scale-in overflow-hidden">
                    <div className="p-2">
                      <p className="px-3 py-2 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                        Choose Theme
                      </p>
                      {themes.map((t) => (
                        <button
                          key={t.value}
                          onClick={() => {
                            setTheme(t.value)
                            setIsThemeMenuOpen(false)
                          }}
                          className={cn(
                            "w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all text-sm",
                            theme === t.value 
                              ? "bg-primary text-primary-foreground shadow-sm" 
                              : "hover:bg-accent"
                          )}
                        >
                          <span className="text-base">{t.icon}</span>
                          <span className="font-medium">{t.label}</span>
                          {theme === t.value && (
                            <div className="ml-auto">
                              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                              </svg>
                            </div>
                          )}
                        </button>
                      ))}
                    </div>
                  </div>
                </>
              )}
            </div>

            {/* Notifications */}
            <button className="p-2.5 rounded-xl hover:bg-accent/50 transition-colors relative">
              <Bell className="h-4 w-4" />
              <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-primary rounded-full" />
            </button>

            {/* Date display */}
            <div className="hidden md:block text-sm text-muted-foreground px-3 py-1.5 rounded-lg bg-muted/50">
              {new Date().toLocaleDateString('en-US', { 
                weekday: 'short', 
                month: 'short', 
                day: 'numeric' 
              })}
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="flex-1 p-4 lg:p-6 animate-fade-in">
          {children}
        </main>

        {/* Footer */}
        <footer className="border-t border-border/50 py-4 glass-card">
          <div className="px-4 lg:px-6 flex items-center justify-between text-xs text-muted-foreground">
            <span>CowHealth AI Research Platform</span>
            <span>v2.0.0</span>
          </div>
        </footer>
      </div>

      {/* Search Modal */}
      {isSearchOpen && (
        <>
          <div
            className="fixed inset-0 bg-background/80 backdrop-blur-sm z-50 animate-fade-in"
            onClick={() => setIsSearchOpen(false)}
          />
          <div className="fixed left-1/2 top-1/4 -translate-x-1/2 w-full max-w-lg z-50 animate-scale-in">
            <div className="bg-card rounded-2xl shadow-2xl border border-border/50 overflow-hidden mx-4">
              <div className="flex items-center gap-3 p-4 border-b border-border/50">
                <Search className="h-5 w-5 text-muted-foreground" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search pages, cows, videos..."
                  className="flex-1 bg-transparent border-none outline-none text-foreground placeholder:text-muted-foreground"
                  autoFocus
                />
                <kbd className="hidden sm:inline-flex h-5 select-none items-center gap-1 rounded border border-border bg-muted px-1.5 font-mono text-[10px] font-medium text-muted-foreground">
                  ESC
                </kbd>
              </div>
              <div className="p-2 max-h-80 overflow-y-auto">
                {/* Quick links */}
                <p className="px-3 py-2 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                  Quick Links
                </p>
                {navItems
                  .filter(item => hasRole(item.roles))
                  .filter(item => 
                    searchQuery === '' || 
                    item.label.toLowerCase().includes(searchQuery.toLowerCase())
                  )
                  .slice(0, 5)
                  .map((item) => {
                    const Icon = item.icon
                    return (
                      <button
                        key={item.path}
                        onClick={() => {
                          navigate(item.path)
                          setIsSearchOpen(false)
                          setSearchQuery('')
                        }}
                        className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-accent/50 transition-colors text-sm"
                      >
                        <Icon className="h-4 w-4 text-muted-foreground" />
                        <span>{item.label}</span>
                        <ChevronRight className="h-4 w-4 text-muted-foreground ml-auto" />
                      </button>
                    )
                  })}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
