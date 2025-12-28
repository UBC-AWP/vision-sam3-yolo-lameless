import { Link, useLocation, useNavigate } from 'react-router-dom'
import { useState } from 'react'
import { cn } from '@/lib/utils'
import { useAuth } from '../contexts/AuthContext'
import {
  User,
  LogOut,
  Settings,
  ChevronDown,
  Shield,
  FlaskConical,
  UserCircle,
  Loader2
} from 'lucide-react'

interface LayoutProps {
  children: React.ReactNode
}

export default function Layout({ children }: LayoutProps) {
  const location = useLocation()
  const navigate = useNavigate()
  const { user, isAuthenticated, isLoading, logout, hasRole } = useAuth()
  const [isUserMenuOpen, setIsUserMenuOpen] = useState(false)

  const navItems = [
    { path: '/', label: 'Dashboard', roles: ['admin', 'researcher', 'rater'] },
    { path: '/upload', label: 'Upload', roles: ['admin', 'researcher'] },
    { path: '/pairwise', label: 'Pairwise', roles: ['admin', 'researcher', 'rater'] },
    { path: '/triplet', label: 'Triplet', roles: ['admin', 'researcher', 'rater'] },
    { path: '/hierarchy', label: 'Hierarchy', roles: ['admin', 'researcher', 'rater'] },
    { path: '/similarity', label: 'Similarity', roles: ['admin', 'researcher', 'rater'] },
    { path: '/learn', label: 'Learn', roles: ['admin', 'researcher', 'rater'] },
    { path: '/pipelines', label: 'Pipelines', roles: ['admin', 'researcher'] },
    { path: '/health', label: 'Health', roles: ['admin', 'researcher'] },
    { path: '/training', label: 'Queue', roles: ['admin', 'researcher'] },
    { path: '/models', label: 'Config', roles: ['admin', 'researcher'] },
    { path: '/ml-config', label: 'ML Config', roles: ['admin', 'researcher'] },
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
    <div className="min-h-screen bg-background">
      {/* Header Navigation */}
      <nav className="border-b bg-card shadow-sm">
        <div className="container mx-auto px-4">
          <div className="flex items-center justify-between h-16">
            {/* Logo/Title */}
            <Link to="/" className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-primary to-primary/60 rounded-lg flex items-center justify-center">
                <span className="text-primary-foreground font-bold text-lg">C</span>
              </div>
              <div>
                <h1 className="text-lg font-bold leading-tight">Lameness Detection</h1>
                <p className="text-xs text-muted-foreground">Research Pipeline</p>
              </div>
            </Link>

            {/* Navigation Items */}
            <div className="flex items-center gap-1">
              {navItems
                .filter(item => hasRole(item.roles))
                .map((item) => (
                  <Link
                    key={item.path}
                    to={item.path}
                    className={cn(
                      'px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200',
                      location.pathname === item.path
                        ? 'bg-primary text-primary-foreground shadow-sm'
                        : 'text-muted-foreground hover:text-foreground hover:bg-accent'
                    )}
                  >
                    {item.label}
                  </Link>
                ))}
            </div>

            {/* User Menu */}
            <div className="relative">
              <button
                onClick={() => setIsUserMenuOpen(!isUserMenuOpen)}
                className="flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-accent transition-colors"
              >
                <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
                  <User className="h-4 w-4 text-primary" />
                </div>
                <div className="text-left hidden sm:block">
                  <p className="text-sm font-medium">{user?.username}</p>
                  <p className="text-xs text-muted-foreground capitalize">{user?.role}</p>
                </div>
                <ChevronDown className={cn(
                  "h-4 w-4 text-muted-foreground transition-transform",
                  isUserMenuOpen && "transform rotate-180"
                )} />
              </button>

              {/* Dropdown Menu */}
              {isUserMenuOpen && (
                <div className="absolute right-0 mt-2 w-64 bg-card rounded-lg shadow-lg border z-50">
                  <div className="p-4 border-b">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
                        <User className="h-5 w-5 text-primary" />
                      </div>
                      <div>
                        <p className="font-medium">{user?.username}</p>
                        <p className="text-xs text-muted-foreground">{user?.email}</p>
                      </div>
                    </div>
                    <div className="mt-3 flex items-center gap-2">
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
                        // TODO: Navigate to settings
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
        </div>
      </nav>

      {/* Click outside to close menu */}
      {isUserMenuOpen && (
        <div
          className="fixed inset-0 z-40"
          onClick={() => setIsUserMenuOpen(false)}
        />
      )}

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        {children}
      </main>

      {/* Footer */}
      <footer className="border-t mt-auto py-4">
        <div className="container mx-auto px-4 text-center text-xs text-muted-foreground">
          Cow Lameness Detection Research Platform • YOLO + SAM3 + DINOv3 + T-LEAP + TCN + Transformer + GraphGPS Pipeline
        </div>
      </footer>
    </div>
  )
}
