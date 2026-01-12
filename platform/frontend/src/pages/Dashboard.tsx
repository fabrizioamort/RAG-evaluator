import { useQuery } from '@tanstack/react-query'
import { api } from '@/api/client'
import {
  FolderOpen,
  Database,
  FileText,
  FlaskConical,
  Plus,
  Upload,
  Play,
  Sparkles
} from 'lucide-react'

interface StatCardProps {
  title: string
  value: string | number
  icon: React.ReactNode
}

function StatCard({ title, value, icon }: StatCardProps) {
  return (
    <div className="rounded-lg border border-border bg-card p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-muted-foreground">{title}</p>
          <p className="mt-2 text-3xl font-bold">{value}</p>
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
  icon: React.ReactNode
  onClick?: () => void
}

function QuickAction({ title, icon, onClick }: QuickActionProps) {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-3 rounded-md border border-border bg-card px-4 py-3 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground"
    >
      {icon}
      {title}
    </button>
  )
}

export function Dashboard() {
  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: () => api.health.check(),
  })

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
          value={0}
          icon={<FolderOpen className="h-5 w-5" />}
        />
        <StatCard
          title="Knowledge Bases"
          value={0}
          icon={<Database className="h-5 w-5" />}
        />
        <StatCard
          title="Test Sets"
          value={0}
          icon={<FileText className="h-5 w-5" />}
        />
        <StatCard
          title="Evaluations"
          value={0}
          icon={<FlaskConical className="h-5 w-5" />}
        />
      </div>

      <div className="grid gap-8 lg:grid-cols-2">
        {/* Quick Actions */}
        <div className="rounded-lg border border-border bg-card p-6">
          <h2 className="text-lg font-semibold">Quick Actions</h2>
          <div className="mt-4 flex flex-col gap-2">
            <QuickAction
              title="New Project"
              icon={<Plus className="h-4 w-4" />}
            />
            <QuickAction
              title="Upload Documents"
              icon={<Upload className="h-4 w-4" />}
            />
            <QuickAction
              title="Run Evaluation"
              icon={<Play className="h-4 w-4" />}
            />
            <QuickAction
              title="Generate Test Set"
              icon={<Sparkles className="h-4 w-4" />}
            />
          </div>
        </div>

        {/* Recent Activity */}
        <div className="rounded-lg border border-border bg-card p-6">
          <h2 className="text-lg font-semibold">Recent Activity</h2>
          <div className="mt-4 flex items-center justify-center py-8 text-muted-foreground">
            <p>No recent activity</p>
          </div>
        </div>
      </div>
    </div>
  )
}
