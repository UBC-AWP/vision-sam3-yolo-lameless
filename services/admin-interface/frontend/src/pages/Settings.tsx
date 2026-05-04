/**
 * Settings Page
 * Premium settings interface with theme selection and profile management
 */
import { useState } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { useTheme, themes as themeOptions } from '../contexts/ThemeContext'
import { authApi } from '../api/client'
import { cn } from '@/lib/utils'
import {
  User,
  Lock,
  Mail,
  Shield,
  FlaskConical,
  UserCircle,
  Eye,
  EyeOff,
  CheckCircle,
  AlertCircle,
  Loader2,
  Palette,
  Sun,
  Moon,
  Monitor,
  Sparkles,
  Check
} from 'lucide-react'

export default function Settings() {
  const { user } = useAuth()
  const { theme, setTheme, resolvedTheme } = useTheme()
  const [activeTab, setActiveTab] = useState<'profile' | 'appearance' | 'password'>('profile')

  // Password change state
  const [currentPassword, setCurrentPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [showCurrentPassword, setShowCurrentPassword] = useState(false)
  const [showNewPassword, setShowNewPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)
  const [passwordLoading, setPasswordLoading] = useState(false)
  const [passwordMessage, setPasswordMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)

  const getRoleIcon = (role: string) => {
    switch (role) {
      case 'admin':
        return <Shield className="h-5 w-5 text-rose-500" />
      case 'researcher':
        return <FlaskConical className="h-5 w-5 text-blue-500" />
      default:
        return <UserCircle className="h-5 w-5 text-emerald-500" />
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

  const getTierBadgeStyles = (tier: string) => {
    switch (tier?.toLowerCase()) {
      case 'gold':
        return 'bg-amber-500/15 text-amber-500 border-amber-500/30'
      case 'silver':
        return 'bg-slate-400/15 text-slate-400 border-slate-400/30'
      default:
        return 'bg-orange-500/15 text-orange-500 border-orange-500/30'
    }
  }

  const handlePasswordChange = async (e: React.FormEvent) => {
    e.preventDefault()
    setPasswordMessage(null)

    // Validation
    if (!currentPassword || !newPassword || !confirmPassword) {
      setPasswordMessage({ type: 'error', text: 'All fields are required' })
      return
    }

    if (newPassword.length < 8) {
      setPasswordMessage({ type: 'error', text: 'New password must be at least 8 characters' })
      return
    }

    if (newPassword !== confirmPassword) {
      setPasswordMessage({ type: 'error', text: 'New passwords do not match' })
      return
    }

    setPasswordLoading(true)
    try {
      await authApi.changePassword(currentPassword, newPassword)
      setPasswordMessage({ type: 'success', text: 'Password changed successfully!' })
      setCurrentPassword('')
      setNewPassword('')
      setConfirmPassword('')
    } catch (error: any) {
      const message = error.response?.data?.detail || 'Failed to change password'
      setPasswordMessage({ type: 'error', text: message })
    } finally {
      setPasswordLoading(false)
    }
  }

  const getPasswordStrength = (password: string): { strength: number; label: string; color: string } => {
    let strength = 0
    if (password.length >= 8) strength++
    if (password.length >= 12) strength++
    if (/[A-Z]/.test(password)) strength++
    if (/[0-9]/.test(password)) strength++
    if (/[^A-Za-z0-9]/.test(password)) strength++

    if (strength <= 1) return { strength: 1, label: 'Weak', color: 'bg-red-500' }
    if (strength <= 2) return { strength: 2, label: 'Fair', color: 'bg-orange-500' }
    if (strength <= 3) return { strength: 3, label: 'Good', color: 'bg-yellow-500' }
    if (strength <= 4) return { strength: 4, label: 'Strong', color: 'bg-emerald-500' }
    return { strength: 5, label: 'Very Strong', color: 'bg-emerald-600' }
  }

  const passwordStrength = getPasswordStrength(newPassword)

  const tabs = [
    { id: 'profile', label: 'Profile', icon: User },
    { id: 'appearance', label: 'Appearance', icon: Palette },
    { id: 'password', label: 'Security', icon: Lock },
  ] as const

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-primary to-primary/60 flex items-center justify-center shadow-lg shadow-primary/20">
          <Sparkles className="h-6 w-6 text-primary-foreground" />
        </div>
        <div>
          <h1 className="text-2xl font-bold">Settings</h1>
          <p className="text-muted-foreground">Manage your account and preferences</p>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 p-1 bg-muted/50 rounded-xl w-fit">
        {tabs.map((tab) => {
          const Icon = tab.icon
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={cn(
                "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200",
                activeTab === tab.id
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              <Icon className="h-4 w-4" />
              {tab.label}
            </button>
          )
        })}
      </div>

      {/* Profile Tab */}
      {activeTab === 'profile' && (
        <div className="premium-card animate-fade-in">
          <h2 className="text-lg font-semibold mb-6">Profile Information</h2>

          <div className="flex flex-col sm:flex-row items-start gap-6">
            {/* Avatar */}
            <div className="relative group">
              <div className="w-24 h-24 rounded-2xl bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center border border-border/50">
                <User className="h-12 w-12 text-primary" />
              </div>
              <div className="absolute inset-0 rounded-2xl bg-primary/10 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                <span className="text-xs font-medium text-primary">Change</span>
              </div>
            </div>

            {/* Info */}
            <div className="flex-1 space-y-6">
              {/* Username */}
              <div className="space-y-2">
                <label className="text-sm font-medium text-muted-foreground">Username</label>
                <div className="flex items-center gap-3 p-3 rounded-xl bg-muted/50">
                  <User className="h-4 w-4 text-muted-foreground" />
                  <span className="font-medium">{user?.username}</span>
                </div>
              </div>

              {/* Email */}
              <div className="space-y-2">
                <label className="text-sm font-medium text-muted-foreground">Email</label>
                <div className="flex items-center gap-3 p-3 rounded-xl bg-muted/50">
                  <Mail className="h-4 w-4 text-muted-foreground" />
                  <span>{user?.email}</span>
                </div>
              </div>

              {/* Role & Tier */}
              <div className="flex flex-wrap gap-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-muted-foreground">Role</label>
                  <div className="flex items-center gap-2">
                    {user && getRoleIcon(user.role)}
                    <span className={cn(
                      "px-3 py-1.5 rounded-lg text-sm font-medium capitalize border",
                      user && getRoleBadgeStyles(user.role)
                    )}>
                      {user?.role}
                    </span>
                  </div>
                </div>

                {user?.rater_tier && (
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-muted-foreground">Rater Tier</label>
                    <span className={cn(
                      "inline-flex px-3 py-1.5 rounded-lg text-sm font-medium capitalize border",
                      getTierBadgeStyles(user.rater_tier)
                    )}>
                      {user.rater_tier} Tier
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="mt-6 pt-6 border-t border-border/50">
            <p className="text-sm text-muted-foreground">
              To change your username or email, please contact an administrator.
            </p>
          </div>
        </div>
      )}

      {/* Appearance Tab */}
      {activeTab === 'appearance' && (
        <div className="space-y-6 animate-fade-in">
          {/* Theme Selection */}
          <div className="premium-card">
            <h2 className="text-lg font-semibold mb-2">Theme</h2>
            <p className="text-sm text-muted-foreground mb-6">
              Choose how the application looks. Select a theme that suits your preference.
            </p>

            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
              {themeOptions.map((t) => (
                <button
                  key={t.value}
                  onClick={() => setTheme(t.value)}
                  className={cn(
                    "relative p-4 rounded-xl border-2 transition-all duration-200 text-left group",
                    theme === t.value
                      ? "border-primary bg-primary/5 shadow-lg shadow-primary/10"
                      : "border-border/50 hover:border-border hover:bg-muted/50"
                  )}
                >
                  {/* Check indicator */}
                  {theme === t.value && (
                    <div className="absolute top-2 right-2 w-5 h-5 rounded-full bg-primary flex items-center justify-center">
                      <Check className="h-3 w-3 text-primary-foreground" />
                    </div>
                  )}

                  {/* Theme preview */}
                  <div className="flex gap-1 mb-3">
                    {t.preview.map((color, i) => (
                      <div
                        key={i}
                        className="w-6 h-6 rounded-lg shadow-sm first:rounded-l-lg last:rounded-r-lg"
                        style={{ backgroundColor: color }}
                      />
                    ))}
                  </div>

                  {/* Theme info */}
                  <div className="flex items-center gap-2">
                    <span className="text-lg">{t.icon}</span>
                    <span className="font-medium text-sm">{t.label}</span>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Current Theme Info */}
          <div className="premium-card bg-gradient-to-br from-primary/5 to-accent/5">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-xl bg-primary/20 flex items-center justify-center">
                {resolvedTheme === 'light' ? (
                  <Sun className="h-6 w-6 text-primary" />
                ) : (
                  <Moon className="h-6 w-6 text-primary" />
                )}
              </div>
              <div>
                <h3 className="font-semibold">
                  Currently using {themeOptions.find(t => t.value === theme)?.label} theme
                </h3>
                <p className="text-sm text-muted-foreground">
                  Theme is applied across the application
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Password Tab */}
      {activeTab === 'password' && (
        <div className="premium-card animate-fade-in">
          <h2 className="text-lg font-semibold mb-6">Change Password</h2>

          {passwordMessage && (
            <div className={cn(
              "mb-6 p-4 rounded-xl flex items-center gap-3 border",
              passwordMessage.type === 'success'
                ? 'bg-emerald-500/10 text-emerald-500 border-emerald-500/30'
                : 'bg-red-500/10 text-red-500 border-red-500/30'
            )}>
              {passwordMessage.type === 'success' ? (
                <CheckCircle className="h-5 w-5 flex-shrink-0" />
              ) : (
                <AlertCircle className="h-5 w-5 flex-shrink-0" />
              )}
              <span className="text-sm font-medium">{passwordMessage.text}</span>
            </div>
          )}

          <form onSubmit={handlePasswordChange} className="space-y-5 max-w-md">
            {/* Current Password */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Current Password</label>
              <div className="relative">
                <input
                  type={showCurrentPassword ? 'text' : 'password'}
                  value={currentPassword}
                  onChange={(e) => setCurrentPassword(e.target.value)}
                  className="input-premium pr-10"
                  placeholder="Enter current password"
                />
                <button
                  type="button"
                  onClick={() => setShowCurrentPassword(!showCurrentPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                >
                  {showCurrentPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
            </div>

            {/* New Password */}
            <div className="space-y-2">
              <label className="text-sm font-medium">New Password</label>
              <div className="relative">
                <input
                  type={showNewPassword ? 'text' : 'password'}
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  className="input-premium pr-10"
                  placeholder="Enter new password"
                />
                <button
                  type="button"
                  onClick={() => setShowNewPassword(!showNewPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                >
                  {showNewPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
              {/* Password Strength Indicator */}
              {newPassword && (
                <div className="space-y-2 pt-1">
                  <div className="flex gap-1">
                    {[1, 2, 3, 4, 5].map((level) => (
                      <div
                        key={level}
                        className={cn(
                          "h-1.5 flex-1 rounded-full transition-colors",
                          level <= passwordStrength.strength ? passwordStrength.color : 'bg-muted'
                        )}
                      />
                    ))}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Strength: <span className="font-medium">{passwordStrength.label}</span>
                  </p>
                </div>
              )}
            </div>

            {/* Confirm Password */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Confirm New Password</label>
              <div className="relative">
                <input
                  type={showConfirmPassword ? 'text' : 'password'}
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  className={cn(
                    "input-premium pr-10",
                    confirmPassword && confirmPassword !== newPassword && 'border-red-500 focus:border-red-500 focus:ring-red-500/20'
                  )}
                  placeholder="Confirm new password"
                />
                <button
                  type="button"
                  onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                >
                  {showConfirmPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
              {confirmPassword && confirmPassword !== newPassword && (
                <p className="text-xs text-red-500">Passwords do not match</p>
              )}
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={passwordLoading || !currentPassword || !newPassword || !confirmPassword || newPassword !== confirmPassword}
              className="btn-premium w-full py-2.5 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {passwordLoading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Changing Password...
                </>
              ) : (
                <>
                  <Lock className="h-4 w-4" />
                  Change Password
                </>
              )}
            </button>
          </form>

          <div className="mt-6 pt-6 border-t border-border/50">
            <h3 className="text-sm font-medium mb-3">Password Requirements</h3>
            <ul className="space-y-2 text-sm">
              {[
                { check: newPassword.length >= 8, text: 'At least 8 characters' },
                { check: /[A-Z]/.test(newPassword), text: 'One uppercase letter (recommended)' },
                { check: /[0-9]/.test(newPassword), text: 'One number (recommended)' },
                { check: /[^A-Za-z0-9]/.test(newPassword), text: 'One special character (recommended)' },
              ].map((req, i) => (
                <li key={i} className={cn(
                  "flex items-center gap-2 transition-colors",
                  req.check ? 'text-emerald-500' : 'text-muted-foreground'
                )}>
                  <div className={cn(
                    "w-4 h-4 rounded-full flex items-center justify-center text-xs",
                    req.check ? 'bg-emerald-500/20' : 'bg-muted'
                  )}>
                    {req.check ? <Check className="h-3 w-3" /> : '•'}
                  </div>
                  {req.text}
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  )
}
