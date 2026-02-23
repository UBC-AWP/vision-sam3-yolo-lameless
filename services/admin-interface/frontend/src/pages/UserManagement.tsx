/**
 * User Management Page
 * Admin-only page for managing users, roles, and tiers
 */
import { useState, useEffect } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { usersApi, User, CreateUserData } from '../api/client'
import { cn } from '@/lib/utils'
import {
  Users,
  UserPlus,
  Shield,
  ShieldCheck,
  Crown,
  Loader2,
  AlertCircle,
  CheckCircle,
  Trash2,
  Edit,
  Eye,
  EyeOff,
  X,
  Search,
  RefreshCw,
  Sparkles,
} from 'lucide-react'

type RoleFilter = 'all' | 'admin' | 'tenant_admin' | 'researcher' | 'viewer' | 'rater'

export default function UserManagement() {
  const { user: currentUser } = useAuth()
  const [users, setUsers] = useState<User[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [roleFilter, setRoleFilter] = useState<RoleFilter>('all')

  // Modal states
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [showEditModal, setShowEditModal] = useState(false)
  const [selectedUser, setSelectedUser] = useState<User | null>(null)
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const [userToDelete, setUserToDelete] = useState<User | null>(null)

  // Form states
  const [formData, setFormData] = useState<CreateUserData>({
    email: '',
    username: '',
    password: '',
    role: 'rater',
    rater_tier: 'bronze',
    tenant_name: null,
  })
  const [showPassword, setShowPassword] = useState(false)
  const [formLoading, setFormLoading] = useState(false)

  useEffect(() => {
    loadUsers()
  }, [])

  const loadUsers = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await usersApi.list()
      setUsers(data)
    } catch (err: unknown) {
      const error = err as { response?: { data?: { detail?: string } } }
      setError(error.response?.data?.detail || 'Failed to load users')
    } finally {
      setLoading(false)
    }
  }

  const handleCreateUser = async (e: React.FormEvent) => {
    e.preventDefault()
    setFormLoading(true)
    setError(null)

    try {
      await usersApi.create(formData)
      setSuccess('User created successfully')
      setShowCreateModal(false)
      resetForm()
      loadUsers()
    } catch (err: unknown) {
      const error = err as { response?: { data?: { detail?: string } } }
      setError(error.response?.data?.detail || 'Failed to create user')
    } finally {
      setFormLoading(false)
    }
  }

  const handleUpdateRole = async (userId: string, newRole: string) => {
    try {
      await usersApi.updateRole(userId, newRole)
      setSuccess('User role updated')
      loadUsers()
    } catch (err: unknown) {
      const error = err as { response?: { data?: { detail?: string } } }
      setError(error.response?.data?.detail || 'Failed to update role')
    }
  }

  const handleUpdateTier = async (userId: string, newTier: string) => {
    try {
      await usersApi.updateTier(userId, newTier)
      setSuccess('User tier updated')
      loadUsers()
    } catch (err: unknown) {
      const error = err as { response?: { data?: { detail?: string } } }
      setError(error.response?.data?.detail || 'Failed to update tier')
    }
  }

  const handleToggleStatus = async (userId: string, currentStatus: boolean) => {
    try {
      await usersApi.updateStatus(userId, !currentStatus)
      setSuccess(`User ${currentStatus ? 'disabled' : 'enabled'}`)
      loadUsers()
    } catch (err: unknown) {
      const error = err as { response?: { data?: { detail?: string } } }
      setError(error.response?.data?.detail || 'Failed to update status')
    }
  }

  const handleDeleteUser = async () => {
    if (!userToDelete) return

    try {
      await usersApi.delete(userToDelete.id)
      setSuccess('User deleted successfully')
      setShowDeleteConfirm(false)
      setUserToDelete(null)
      loadUsers()
    } catch (err: unknown) {
      const error = err as { response?: { data?: { detail?: string } } }
      setError(error.response?.data?.detail || 'Failed to delete user')
    }
  }

  const resetForm = () => {
    setFormData({
      email: '',
      username: '',
      password: '',
      role: 'rater',
      rater_tier: 'bronze',
      tenant_name: null,
    })
    setShowPassword(false)
  }

  const filteredUsers = users.filter(user => {
    const matchesSearch =
      user.username.toLowerCase().includes(searchTerm.toLowerCase()) ||
      user.email.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesRole = roleFilter === 'all' || user.role === roleFilter
    return matchesSearch && matchesRole
  })

  const getRoleIcon = (role: string) => {
    switch (role) {
      case 'admin':
        return <Crown className="h-4 w-4 text-amber-500" />
      case 'tenant_admin':
        return <ShieldCheck className="h-4 w-4 text-teal-500" />
      case 'researcher':
        return <ShieldCheck className="h-4 w-4 text-blue-500" />
      case 'viewer':
        return <Eye className="h-4 w-4 text-slate-500" />
      case 'rater':
        return <Shield className="h-4 w-4 text-muted-foreground" />
      default:
        return <Users className="h-4 w-4" />
    }
  }

  const getTierBadge = (tier: string | null | undefined) => {
    if (!tier) return null
    const styles = {
      gold: 'bg-amber-500/15 text-amber-500 border-amber-500/30',
      silver: 'bg-slate-400/15 text-slate-400 border-slate-400/30',
      bronze: 'bg-orange-500/15 text-orange-500 border-orange-500/30',
    }
    return (
      <span className={cn(
        "px-2 py-0.5 text-xs rounded-lg border font-medium capitalize",
        styles[tier as keyof typeof styles] || ''
      )}>
        {tier}
      </span>
    )
  }

  const roleStats = {
    admin: users.filter(u => u.role === 'admin').length,
    tenant_admin: users.filter(u => u.role === 'tenant_admin').length,
    researcher: users.filter(u => u.role === 'researcher').length,
    viewer: users.filter(u => u.role === 'viewer').length,
    rater: users.filter(u => u.role === 'rater').length,
  }

  if (currentUser?.role !== 'admin' && currentUser?.role !== 'tenant_admin') {
    return (
      <div className="min-h-[60vh] flex items-center justify-center">
        <div className="text-center animate-fade-in">
          <div className="w-16 h-16 rounded-2xl bg-muted flex items-center justify-center mx-auto mb-4">
            <Shield className="h-8 w-8 text-muted-foreground" />
          </div>
          <h2 className="text-xl font-semibold">Access Denied</h2>
          <p className="text-muted-foreground mt-2">Only administrators or tenant admins can access this page.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4 animate-slide-in-up">
        <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-primary to-primary/60 flex items-center justify-center shadow-lg shadow-primary/20">
          <Users className="h-6 w-6 text-primary-foreground" />
        </div>
        <div>
          <h1 className="text-2xl font-bold">User Management</h1>
          <p className="text-muted-foreground">Manage users, roles, and permissions</p>
        </div>
      </div>

      {/* Alerts */}
      {error && (
        <div className="p-4 rounded-xl bg-destructive/10 border border-destructive/30 flex items-center gap-3 text-destructive animate-scale-in">
          <AlertCircle className="h-5 w-5 flex-shrink-0" />
          <span className="text-sm font-medium flex-1">{error}</span>
          <button onClick={() => setError(null)} className="hover:bg-destructive/20 p-1 rounded-lg transition-colors">
            <X className="h-4 w-4" />
          </button>
        </div>
      )}

      {success && (
        <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-center gap-3 text-emerald-500 animate-scale-in">
          <CheckCircle className="h-5 w-5 flex-shrink-0" />
          <span className="text-sm font-medium flex-1">{success}</span>
          <button onClick={() => setSuccess(null)} className="hover:bg-emerald-500/20 p-1 rounded-lg transition-colors">
            <X className="h-4 w-4" />
          </button>
        </div>
      )}

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {[
          { label: 'Total Users', value: users.length, icon: Users, color: 'from-blue-500 to-blue-600', textColor: 'text-blue-500', bgColor: 'bg-blue-500/10' },
          { label: 'Admins', value: roleStats.admin, icon: Crown, color: 'from-amber-500 to-amber-600', textColor: 'text-amber-500', bgColor: 'bg-amber-500/10' },
          { label: 'Researchers', value: roleStats.researcher, icon: ShieldCheck, color: 'from-violet-500 to-violet-600', textColor: 'text-violet-500', bgColor: 'bg-violet-500/10' },
          { label: 'Viewers', value: roleStats.viewer, icon: Eye, color: 'from-slate-500 to-slate-600', textColor: 'text-slate-400', bgColor: 'bg-slate-500/10' },
          { label: 'Raters', value: roleStats.rater, icon: Shield, color: 'from-slate-500 to-slate-600', textColor: 'text-slate-400', bgColor: 'bg-slate-500/10' },
        ].slice(0, 5).map((stat, i) => (
          <div
            key={stat.label}
            className="premium-card animate-slide-in-up"
            style={{ animationDelay: `${i * 0.05}s`, animationFillMode: 'backwards' }}
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">{stat.label}</p>
                <p className="text-2xl font-bold gradient-text">{stat.value}</p>
              </div>
              <div className={cn("w-12 h-12 rounded-xl flex items-center justify-center", stat.bgColor)}>
                <stat.icon className={cn("h-6 w-6", stat.textColor)} />
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Actions Bar */}
      <div className="premium-card animate-slide-in-up" style={{ animationDelay: '0.2s', animationFillMode: 'backwards' }}>
        <div className="flex flex-col md:flex-row gap-4 items-center justify-between">
          {/* Search */}
          <div className="relative flex-1 w-full md:max-w-md">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search by username or email..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="input-premium pl-10"
            />
          </div>

          {/* Filters */}
          <div className="flex items-center gap-2">
            <select
              value={roleFilter}
              onChange={(e) => setRoleFilter(e.target.value as RoleFilter)}
              className="input-premium w-auto"
            >
              <option value="all">All Roles</option>
              <option value="admin">Admins</option>
              <option value="tenant_admin">Tenant Admins</option>
              <option value="researcher">Researchers</option>
              <option value="viewer">Viewers</option>
              <option value="rater">Raters</option>
            </select>

            <button
              onClick={loadUsers}
              className="p-2.5 rounded-xl hover:bg-accent/50 text-muted-foreground hover:text-foreground transition-colors"
              title="Refresh"
            >
              <RefreshCw className={cn("h-5 w-5", loading && 'animate-spin')} />
            </button>

            {currentUser?.role === 'admin' && (
              <button
                onClick={() => {
                  resetForm()
                  setShowCreateModal(true)
                }}
                className="btn-premium flex items-center gap-2"
              >
                <UserPlus className="h-4 w-4" />
                Add User
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Users Table */}
      <div className="premium-card p-0 overflow-hidden animate-slide-in-up" style={{ animationDelay: '0.3s', animationFillMode: 'backwards' }}>
        {loading ? (
          <div className="flex items-center justify-center py-16">
            <div className="text-center">
              <Loader2 className="h-8 w-8 animate-spin text-primary mx-auto mb-2" />
              <p className="text-sm text-muted-foreground">Loading users...</p>
            </div>
          </div>
        ) : filteredUsers.length === 0 ? (
          <div className="text-center py-16">
            <div className="w-16 h-16 rounded-2xl bg-muted flex items-center justify-center mx-auto mb-4">
              <Users className="h-8 w-8 text-muted-foreground" />
            </div>
            <p className="text-muted-foreground">No users found</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="premium-table">
              <thead>
                <tr>
                  <th>User</th>
                  <th>Role</th>
                  <th>Tenant</th>
                  <th>Tier</th>
                  <th>Status</th>
                  <th>Last Login</th>
                  <th className="text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredUsers.map((user, i) => (
                  <tr
                    key={user.id}
                    className="animate-fade-in"
                    style={{ animationDelay: `${i * 0.03}s`, animationFillMode: 'backwards' }}
                  >
                    <td>
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center flex-shrink-0">
                          <span className="text-primary font-semibold">
                            {user.username.charAt(0).toUpperCase()}
                          </span>
                        </div>
                        <div>
                          <div className="font-medium flex items-center gap-2">
                            {user.username}
                            {user.id === currentUser?.id && (
                              <span className="badge badge-primary">You</span>
                            )}
                          </div>
                          <div className="text-sm text-muted-foreground">{user.email}</div>
                        </div>
                      </div>
                    </td>
                    <td>
                      <div className="flex items-center gap-2">
                        {getRoleIcon(user.role)}
                        <select
                          value={user.role}
                          onChange={(e) => handleUpdateRole(user.id, e.target.value)}
                          disabled={user.id === currentUser?.id}
                          className="text-sm bg-transparent border-0 focus:ring-0 cursor-pointer disabled:cursor-not-allowed disabled:opacity-50 capitalize"
                        >
                          {(['admin', 'tenant_admin', 'researcher', 'viewer', 'rater'] as const).filter(r => currentUser?.role !== 'tenant_admin' || ['researcher', 'viewer'].includes(r)).map(r => (
                            <option key={r} value={r}>{r.replace('_', ' ')}</option>
                          ))}
                        </select>
                      </div>
                    </td>
                    <td className="text-muted-foreground text-sm">
                      {['researcher', 'viewer', 'tenant_admin'].includes(user.role) ? (user.tenant_name ?? '—') : '—'}
                    </td>
                    <td>
                      {user.role === 'rater' ? (
                        <select
                          value={user.rater_tier || 'bronze'}
                          onChange={(e) => handleUpdateTier(user.id, e.target.value)}
                          className="input-premium text-sm py-1 px-2 w-auto"
                        >
                          <option value="gold">Gold</option>
                          <option value="silver">Silver</option>
                          <option value="bronze">Bronze</option>
                        </select>
                      ) : (
                        <span className="text-muted-foreground text-sm">N/A</span>
                      )}
                    </td>
                    <td>
                      {currentUser?.role === 'admin' ? (
                        <button
                          onClick={() => handleToggleStatus(user.id, user.is_active)}
                          disabled={user.id === currentUser?.id}
                          className={cn(
                            "badge transition-colors",
                            user.is_active
                              ? 'badge-success hover:bg-emerald-500/25'
                              : 'badge-destructive hover:bg-destructive/25',
                            user.id === currentUser?.id && 'opacity-50 cursor-not-allowed'
                          )}
                        >
                          {user.is_active ? 'Active' : 'Disabled'}
                        </button>
                      ) : (
                        <span className={cn("badge", user.is_active ? 'badge-success' : 'badge-destructive')}>
                          {user.is_active ? 'Active' : 'Disabled'}
                        </span>
                      )}
                    </td>
                    <td className="text-muted-foreground">
                      {user.last_login
                        ? new Date(user.last_login).toLocaleDateString()
                        : 'Never'}
                    </td>
                    <td className="text-right">
                      {currentUser?.role === 'admin' && (
                        <button
                          onClick={() => {
                            setUserToDelete(user)
                            setShowDeleteConfirm(true)
                          }}
                          disabled={user.id === currentUser?.id}
                          className={cn(
                            "p-2 rounded-lg text-destructive hover:bg-destructive/10 transition-colors",
                            user.id === currentUser?.id && 'opacity-50 cursor-not-allowed'
                          )}
                          title="Delete user"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Create User Modal */}
      {showCreateModal && (
        <>
          <div
            className="fixed inset-0 bg-background/80 backdrop-blur-sm z-50 animate-fade-in"
            onClick={() => {
              setShowCreateModal(false)
              resetForm()
            }}
          />
          <div className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-md z-50 animate-scale-in mx-4">
            <div className="bg-card rounded-2xl shadow-2xl border border-border/50 overflow-hidden">
              <div className="flex items-center justify-between p-4 border-b border-border/50">
                <h3 className="text-lg font-semibold">Create New User</h3>
                <button
                  onClick={() => {
                    setShowCreateModal(false)
                    resetForm()
                  }}
                  className="p-2 rounded-lg hover:bg-accent/50 text-muted-foreground hover:text-foreground transition-colors"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>
              <form onSubmit={handleCreateUser} className="p-4 space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Email</label>
                  <input
                    type="email"
                    required
                    value={formData.email}
                    onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                    className="input-premium"
                    placeholder="user@example.com"
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Username</label>
                  <input
                    type="text"
                    required
                    minLength={3}
                    value={formData.username}
                    onChange={(e) => setFormData({ ...formData, username: e.target.value })}
                    className="input-premium"
                    placeholder="username"
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Password</label>
                  <div className="relative">
                    <input
                      type={showPassword ? 'text' : 'password'}
                      required
                      minLength={8}
                      value={formData.password}
                      onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                      className="input-premium pr-10"
                      placeholder="At least 8 characters"
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-3 top-1/2 transform -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                    >
                      {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </button>
                  </div>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Role</label>
                  <select
                    value={formData.role}
                    onChange={(e) => setFormData({
                      ...formData,
                      role: e.target.value as CreateUserData['role'],
                      tenant_name: ['researcher', 'viewer', 'tenant_admin'].includes(e.target.value) ? formData.tenant_name : null
                    })}
                    className="input-premium"
                  >
                    <option value="rater">Rater</option>
                    <option value="viewer">Viewer</option>
                    <option value="researcher">Researcher</option>
                    <option value="tenant_admin">Tenant Admin</option>
                    <option value="admin">Admin</option>
                  </select>
                </div>
                {['researcher', 'viewer', 'tenant_admin'].includes(formData.role) && (
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Tenant</label>
                    <input
                      type="text"
                      value={formData.tenant_name ?? ''}
                      onChange={(e) => setFormData({ ...formData, tenant_name: e.target.value.trim() || null })}
                      className="input-premium"
                      placeholder="e.g. Farm1, Farm2 (leave empty for default)"
                    />
                  </div>
                )}
                {formData.role === 'rater' && (
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Tier</label>
                    <select
                      value={formData.rater_tier}
                      onChange={(e) => setFormData({ ...formData, rater_tier: e.target.value as 'gold' | 'silver' | 'bronze' })}
                      className="input-premium"
                    >
                      <option value="bronze">Bronze</option>
                      <option value="silver">Silver</option>
                      <option value="gold">Gold</option>
                    </select>
                  </div>
                )}
                <div className="flex justify-end gap-3 pt-4">
                  <button
                    type="button"
                    onClick={() => {
                      setShowCreateModal(false)
                      resetForm()
                    }}
                    className="px-4 py-2 rounded-xl border border-border hover:bg-accent/50 transition-colors font-medium"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={formLoading}
                    className="btn-premium flex items-center gap-2"
                  >
                    {formLoading ? (
                      <>
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Creating...
                      </>
                    ) : (
                      <>
                        <UserPlus className="h-4 w-4" />
                        Create User
                      </>
                    )}
                  </button>
                </div>
              </form>
            </div>
          </div>
        </>
      )}

      {/* Delete Confirmation Modal */}
      {showDeleteConfirm && userToDelete && (
        <>
          <div
            className="fixed inset-0 bg-background/80 backdrop-blur-sm z-50 animate-fade-in"
            onClick={() => {
              setShowDeleteConfirm(false)
              setUserToDelete(null)
            }}
          />
          <div className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-md z-50 animate-scale-in mx-4">
            <div className="bg-card rounded-2xl shadow-2xl border border-border/50 p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-12 h-12 rounded-xl bg-destructive/10 flex items-center justify-center">
                  <Trash2 className="h-6 w-6 text-destructive" />
                </div>
                <h3 className="text-lg font-semibold">Delete User</h3>
              </div>
              <p className="text-muted-foreground mb-6">
                Are you sure you want to delete <strong className="text-foreground">{userToDelete.username}</strong>?
                This action cannot be undone.
              </p>
              <div className="flex justify-end gap-3">
                <button
                  onClick={() => {
                    setShowDeleteConfirm(false)
                    setUserToDelete(null)
                  }}
                  className="px-4 py-2 rounded-xl border border-border hover:bg-accent/50 transition-colors font-medium"
                >
                  Cancel
                </button>
                <button
                  onClick={handleDeleteUser}
                  className="px-4 py-2 rounded-xl bg-destructive text-destructive-foreground hover:bg-destructive/90 transition-colors font-medium"
                >
                  Delete
                </button>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
