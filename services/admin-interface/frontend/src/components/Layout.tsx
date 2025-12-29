import { Link, useLocation, useNavigate } from 'react-router-dom'
import { useState, useEffect } from 'react'
import { cn } from '@/lib/utils'
import { useAuth } from '../contexts/AuthContext'
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
  Menu
} from 'lucide-react'

interface LayoutProps {
  children: React.ReactNode
}

export default function Layout({ children }: LayoutProps) {
  const location = useLocation()
  const navigate = useNavigate()
  const { user, isAuthenticated, isLoading, logout, hasRole } = useAuth()
  const [isUserMenuOpen, setIsUserMenuOpen] = useState(false)
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(() => {
    const saved = localStorage.getItem('sidebar_collapsed')
    return saved === 'true'
  })

  useEffect(() => {
    localStorage.setItem('sidebar_collapsed', String(isSidebarCollapsed))
  }, [isSidebarCollapsed])

  const navItems = [
    { path: '/', label: 'Dashboard', icon: Home, roles: ['admin', 'researcher', 'rater'] },
    { path: '/upload', label: 'Upload', icon: Upload, roles: ['admin', 'researcher'] },
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
        return <Shield className="h-4 w-4 text-red-500" />
      case 'researcher':
        return <FlaskConical className="h-4 w-4 text-blue-500" />
      default:
        return <UserCircle className="h-4 w-4 text-green-500" />
    }
  }

  const getRoleBadgeColor = (role: string) => {
    switch (role) {
      case 'admin':
        return 'bg-red-100 text-red-700'
      case 'researcher':
        return 'bg-blue-100 text-blue-700'
      default:
        return 'bg-green-100 text-green-700'
    }
  }

  // Show loading state
  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          <p className="text-muted-foreground">Loading...</p>
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
      {/* Sidebar Navigation */}
      <aside
        className={cn(
          "fixed left-0 top-0 h-full bg-card border-r shadow-sm z-50 transition-all duration-300 flex flex-col",
          isSidebarCollapsed ? "w-16" : "w-56"
        )}
      >
        {/* Logo/Title */}
        <div className="p-4 border-b flex items-center justify-between">
          <Link to="/" className={cn("flex items-center gap-3", isSidebarCollapsed && "justify-center")}>
            <div className="w-8 h-8 bg-gradient-to-br from-primary to-primary/60 rounded-lg flex items-center justify-center flex-shrink-0">
              <span className="text-primary-foreground font-bold text-sm">C</span>
            </div>
            {!isSidebarCollapsed && (
              <div className="overflow-hidden">
                <h1 className="text-sm font-bold leading-tight truncate">Lameness Detection</h1>
                <p className="text-xs text-muted-foreground truncate">Research Pipeline</p>
              </div>
            )}
          </Link>
        </div>

        {/* Navigation Items */}
        <nav className="flex-1 overflow-y-auto p-2">
          <ul className="space-y-1">
            {navItems
              .filter(item => hasRole(item.roles))
              .map((item) => {
                const Icon = item.icon
                const isActive = location.pathname === item.path
                return (
                  <li key={item.path}>
                    <Link
                      to={item.path}
                      className={cn(
                        'flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200',
                        isActive
                          ? 'bg-primary text-primary-foreground shadow-sm'
                          : 'text-muted-foreground hover:text-foreground hover:bg-accent',
                        isSidebarCollapsed && 'justify-center px-2'
                      )}
                      title={isSidebarCollapsed ? item.label : undefined}
                    >
                      <Icon className="h-4 w-4 flex-shrink-0" />
                      {!isSidebarCollapsed && <span className="truncate">{item.label}</span>}
                    </Link>
                  </li>
                )
              })}
          </ul>
        </nav>

        {/* Collapse Toggle */}
        <div className="p-2 border-t">
          <button
            onClick={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-sm text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
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
        <div className="p-2 border-t">
          <div className="relative">
            <button
              onClick={() => setIsUserMenuOpen(!isUserMenuOpen)}
              className={cn(
                "w-full flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-accent transition-colors",
                isSidebarCollapsed && "justify-center px-2"
              )}
            >
              <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
                <User className="h-4 w-4 text-primary" />
              </div>
              {!isSidebarCollapsed && (
                <>
                  <div className="text-left flex-1 overflow-hidden">
                    <p className="text-sm font-medium truncate">{user?.username}</p>
                    <p className="text-xs text-muted-foreground capitalize truncate">{user?.role}</p>
                  </div>
                  <ChevronDown className={cn(
                    "h-4 w-4 text-muted-foreground transition-transform flex-shrink-0",
                    isUserMenuOpen && "transform rotate-180"
                  )} />
                </>
              )}
            </button>

            {/* User Dropdown Menu */}
            {isUserMenuOpen && (
              <div className={cn(
                "absolute bottom-full mb-2 bg-card rounded-lg shadow-lg border z-50",
                isSidebarCollapsed ? "left-full ml-2 w-64" : "left-0 right-0"
              )}>
                <div className="p-4 border-b">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
                      <User className="h-5 w-5 text-primary" />
                    </div>
                    <div className="overflow-hidden">
                      <p className="font-medium truncate">{user?.username}</p>
                      <p className="text-xs text-muted-foreground truncate">{user?.email}</p>
                    </div>
                  </div>
                  <div className="mt-3 flex items-center gap-2 flex-wrap">
                    {user && getRoleIcon(user.role)}
                    <span className={cn(
                      "text-xs px-2 py-1 rounded-full font-medium capitalize",
                      user && getRoleBadgeColor(user.role)
                    )}>
                      {user?.role}
                    </span>
                    {user?.rater_tier && (
                      <span className="text-xs px-2 py-1 rounded-full font-medium bg-yellow-100 text-yellow-700 capitalize">
                        {user.rater_tier} Tier
                      </span>
                    )}
                  </div>
                </div>

                <div className="p-2">
                  <button
                    onClick={() => {
                      setIsUserMenuOpen(false)
                    }}
                    className="w-full flex items-center gap-2 px-3 py-2 rounded-md hover:bg-accent transition-colors text-sm"
                  >
                    <Settings className="h-4 w-4" />
                    Settings
                  </button>
                  <button
                    onClick={() => {
                      setIsUserMenuOpen(false)
                      handleLogout()
                    }}
                    className="w-full flex items-center gap-2 px-3 py-2 rounded-md hover:bg-red-50 text-red-600 transition-colors text-sm"
                  >
                    <LogOut className="h-4 w-4" />
                    Sign out
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </aside>

      {/* Click outside to close menu */}
      {isUserMenuOpen && (
        <div
          className="fixed inset-0 z-40"
          onClick={() => setIsUserMenuOpen(false)}
        />
      )}

      {/* Main Content Area */}
      <div className={cn(
        "flex-1 flex flex-col min-h-screen transition-all duration-300",
        isSidebarCollapsed ? "ml-16" : "ml-56"
      )}>
        {/* Top Header Bar */}
        <header className="h-14 border-b bg-card flex items-center px-6 sticky top-0 z-30">
          <button
            onClick={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
            className="p-2 rounded-lg hover:bg-accent transition-colors mr-4 lg:hidden"
          >
            <Menu className="h-5 w-5" />
          </button>
          <div className="flex-1" />
          <div className="text-sm text-muted-foreground">
            {new Date().toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}
          </div>
        </header>

        {/* Main Content */}
        <main className="flex-1 p-6">
          {children}
        </main>

        {/* Footer */}
        <footer className="border-t py-4">
          <div className="px-6 text-center text-xs text-muted-foreground">
            Cow Lameness Detection Research Platform
          </div>
        </footer>
      </div>
    </div>
  )
}
