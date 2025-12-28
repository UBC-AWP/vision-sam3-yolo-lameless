import { Link, useLocation } from 'react-router-dom'
import { cn } from '@/lib/utils'

interface LayoutProps {
  children: React.ReactNode
}

export default function Layout({ children }: LayoutProps) {
  const location = useLocation()

  const navItems = [
    { path: '/', label: 'Dashboard' },
    { path: '/upload', label: 'Upload' },
    { path: '/pairwise', label: 'Pairwise' },
    { path: '/triplet', label: 'Triplet' },
    { path: '/hierarchy', label: 'Hierarchy' },
    { path: '/similarity', label: 'Similarity' },
    { path: '/learn', label: '🎓 Learn' },
    { path: '/training', label: 'Queue' },
    { path: '/models', label: 'Config' },
  ]

  return (
    <div className="min-h-screen bg-background">
      {/* Header Navigation */}
      <nav className="border-b bg-card shadow-sm">
        <div className="container mx-auto px-4">
          <div className="flex items-center justify-between h-16">
            {/* Logo/Title */}
            <Link to="/" className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-primary to-primary/60 rounded-lg flex items-center justify-center">
                <span className="text-primary-foreground font-bold text-lg">🐄</span>
              </div>
              <div>
                <h1 className="text-lg font-bold leading-tight">Lameness Detection</h1>
                <p className="text-xs text-muted-foreground">Research Pipeline</p>
              </div>
            </Link>

            {/* Navigation Items */}
            <div className="flex items-center gap-1">
              {navItems.map((item) => (
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

            {/* Status Indicators */}
            <div className="flex items-center gap-3 text-xs">
              <div className="flex items-center gap-1.5">
                <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                <span className="text-muted-foreground">System Active</span>
              </div>
            </div>
          </div>
        </div>
      </nav>

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
