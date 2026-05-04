import { Navigate } from 'react-router-dom'
import { useAuth } from '@/contexts/AuthContext'
import Dashboard from './Dashboard'

export default function HomeRedirect() {
  const { user } = useAuth()

  if (user?.role === 'rater') {
    return <Navigate to="/pairwise" replace />
  }

  return <Dashboard />
}
