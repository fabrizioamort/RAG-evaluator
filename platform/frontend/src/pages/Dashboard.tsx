import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { api, RecentActivityItem } from '@/api/client'
import {
  FolderOpen,
  Database,
  FileText,
  FlaskConical,
  Plus,
  Upload,
  Play,
  Sparkles,
  Loader2,
  CheckCircle2,
  AlertCircle,
  FolderPlus,
  Activity
} from 'lucide-react'
import { cn } from '@/lib/utils'

interface StatCardProps {
  title: string
  value: string | number
  icon: React.ReactNode
  subtitle?: string
  isLoading?: boolean
}

function StatCard({ title, value, icon, subtitle, isLoading }: StatCardProps) {
  return (
    <div className="rounded-xl border border-border bg-card p-6 transition-all hover:shadow-md">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-muted-foreground">{title}</p>
          {isLoading ? (
            <div className="mt-2 h-9 w-16 animate-pulse rounded bg-muted" />
          ) : (
            <p className="mt-2 text-3xl font-bold">{value}</p>
          )}
          {subtitle && (
            <p className="mt-1 text-xs text-muted-foreground">{subtitle}</p>
          )}
        </div>
        <div className="rounded-full bg-primary/10 p-3 text-primary">
          {icon}
        </div>
      </div>
    </div>
  )
}

interface QuickActionProps {
  title: string
  description: string
  icon: React.ReactNode
  onClick?: () => void
}

function QuickAction({ title, description, icon, onClick }: QuickActionProps) {
  return (
    <button
      onClick={onClick}
      className="flex items-start gap-4 rounded-lg border border-border bg-card p-4 text-left transition-all hover:border-primary/50 hover:shadow-md group"
    >
      <div className="rounded-lg bg-primary/10 p-2.5 text-primary transition-colors group-hover:bg-primary group-hover:text-primary-foreground">
        {icon}
      </div>
      <div>
        <p className="font-semibold">{title}</p>
        <p className="mt-0.5 text-sm text-muted-foreground">{description}</p>
      </div>
    </button>
  )
}

function ActivityIcon({ type, action }: { type: string; action: string }) {
  if (type === 'evaluation') {
    if (action === 'completed') return <CheckCircle2 className="h-4 w-4 text-green-500" />
    if (action === 'failed') return <AlertCircle className="h-4 w-4 text-red-500" />
    if (action === 'running') return <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />
    return <FlaskConical className="h-4 w-4 text-muted-foreground" />
  }
  if (type === 'project') return <FolderOpen className="h-4 w-4 text-violet-500" />
  if (type === 'knowledge_base') return <Database className="h-4 w-4 text-amber-500" />
  if (type === 'test_set') return <FileText className="h-4 w-4 text-cyan-500" />
  return <Activity className="h-4 w-4 text-muted-foreground" />
}

function ActivityItem({ item }: { item: RecentActivityItem }) {
  const navigate = useNavigate()

  const handleClick = () => {
    if (item.type === 'project') {
      navigate(`/projects/${item.id}`)
    } else if (item.type === 'knowledge_base') {
      navigate(`/knowledge-bases/${item.id}`)
    }
  }

  const actionText = {
    created: 'was created',
    completed: 'completed',
    failed: 'failed',
    running: 'is running',
    ready: 'is ready',
    indexing: 'is indexing',
  }[item.action] || item.action

  return (
    <div
      onClick={handleClick}
      className={cn(
        "flex items-center gap-3 rounded-lg px-3 py-2.5 transition-colors",
        (item.type === 'project' || item.type === 'knowledge_base') && "cursor-pointer hover:bg-muted/50"
      )}
    >
      <ActivityIcon type={item.type} action={item.action} />
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium truncate">{item.name}</p>
        <p className="text-xs text-muted-foreground">
          {actionText} • {new Date(item.timestamp).toLocaleString()}
        </p>
      </div>
      {item.type === 'evaluation' && item.metadata?.pass_rate !== undefined && item.metadata?.pass_rate !== null && (
        <span className={cn(
          "text-xs font-bold px-2 py-0.5 rounded-full",
          (item.metadata.pass_rate as number) >= 0.7 ? "bg-green-500/10 text-green-600" : "bg-amber-500/10 text-amber-600"
        )}>
          {((item.metadata.pass_rate as number) * 100).toFixed(0)}%
        </span>
      )}
    </div>
  )
}

export function Dashboard() {
  const navigate = useNavigate()

  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: () => api.health.check(),
  })

  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ['stats'],
    queryFn: () => api.stats.get(),
  })

  const { data: activity, isLoading: activityLoading } = useQuery({
    queryKey: ['recent-activity'],
    queryFn: () => api.stats.recentActivity(10),
  })

  const statsData = stats?.data

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold">Dashboard</h1>
        <p className="mt-2 text-muted-foreground">
          Welcome to the RAG Evaluation Platform
        </p>
      </div>

      {/* Connection Status */}
      <div className="flex items-center gap-2 text-sm">
        <span
          className={`h-2 w-2 rounded-full ${
            health?.data?.status === 'healthy' ? 'bg-green-500' : 'bg-red-500'
          }`}
        />
        <span className="text-muted-foreground">
          API: {health?.data?.status || 'connecting...'}
        </span>
        {health?.data?.version && (
          <span className="text-muted-foreground">
            | v{health.data.version}
          </span>
        )}
      </div>

      {/* Stats Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <StatCard
          title="Projects"
          value={statsData?.projects ?? 0}
          icon={<FolderOpen className="h-5 w-5" />}
          isLoading={statsLoading}
        />
        <StatCard
          title="Knowledge Bases"
          value={statsData?.knowledge_bases ?? 0}
          icon={<Database className="h-5 w-5" />}
          isLoading={statsLoading}
        />
        <StatCard
          title="Test Sets"
          value={statsData?.test_sets ?? 0}
          icon={<FileText className="h-5 w-5" />}
          isLoading={statsLoading}
        />
        <StatCard
          title="Evaluations"
          value={statsData?.evaluations ?? 0}
          icon={<FlaskConical className="h-5 w-5" />}
          subtitle={statsData ? `${statsData.completed_evaluations} completed, ${statsData.running_evaluations} running` : undefined}
          isLoading={statsLoading}
        />
      </div>

      <div className="grid gap-8 lg:grid-cols-2">
        {/* Quick Actions */}
        <div className="rounded-xl border border-border bg-card p-6">
          <h2 className="text-lg font-semibold">Quick Actions</h2>
          <p className="mt-1 text-sm text-muted-foreground">Get started with common tasks</p>
          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            <QuickAction
              title="New Project"
              description="Create a new evaluation project"
              icon={<Plus className="h-4 w-4" />}
              onClick={() => navigate('/projects')}
            />
            <QuickAction
              title="Upload Documents"
              description="Add docs to a knowledge base"
              icon={<Upload className="h-4 w-4" />}
              onClick={() => navigate('/projects')}
            />
            <QuickAction
              title="Run Evaluation"
              description="Start a new RAG evaluation"
              icon={<Play className="h-4 w-4" />}
              onClick={() => navigate('/projects')}
            />
            <QuickAction
              title="Generate Test Set"
              description="Auto-generate test cases"
              icon={<Sparkles className="h-4 w-4" />}
              onClick={() => navigate('/projects')}
            />
          </div>
        </div>

        {/* Recent Activity */}
        <div className="rounded-xl border border-border bg-card p-6">
          <h2 className="text-lg font-semibold">Recent Activity</h2>
          <p className="mt-1 text-sm text-muted-foreground">Latest updates across your projects</p>
          <div className="mt-4 space-y-1">
            {activityLoading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            ) : activity?.data?.items && activity.data.items.length > 0 ? (
              activity.data.items.map((item) => (
                <ActivityItem key={`${item.type}-${item.id}`} item={item} />
              ))
            ) : (
              <div className="flex flex-col items-center justify-center py-8 text-center">
                <div className="rounded-full bg-muted p-3">
                  <FolderPlus className="h-6 w-6 text-muted-foreground" />
                </div>
                <p className="mt-3 text-sm text-muted-foreground">No recent activity</p>
                <p className="text-xs text-muted-foreground">Create your first project to get started</p>
                <button
                  onClick={() => navigate('/projects')}
                  className="mt-4 rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground hover:bg-primary/90"
                >
                  Create Project
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
