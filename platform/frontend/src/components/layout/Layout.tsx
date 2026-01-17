import { Outlet, NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  FolderOpen,
  FlaskConical,
  GitCompare,
  TrendingUp,
  Settings,
  Database
} from 'lucide-react'
import { cn } from '@/lib/utils'

const navigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Projects', href: '/projects', icon: FolderOpen },
  { name: 'Indexes', href: '/indexes', icon: Database },
  { name: 'Evaluations', href: '/evaluations', icon: FlaskConical },
  { name: 'Comparisons', href: '/comparisons', icon: GitCompare },
  { name: 'Trends', href: '/trends', icon: TrendingUp },
]

export function Layout() {
  return (
    <div className="flex h-screen bg-background">
      {/* Sidebar */}
      <aside className="w-64 border-r border-border bg-card">
        <div className="flex h-16 items-center border-b border-border px-6">
          <FlaskConical className="h-6 w-6 text-primary" />
          <span className="ml-2 text-lg font-semibold">RAG Evaluator</span>
        </div>
        <nav className="flex flex-col gap-1 p-4">
          {navigation.map((item) => (
            <NavLink
              key={item.name}
              to={item.href}
              className={({ isActive }) =>
                cn(
                  'flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors',
                  isActive
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
                )
              }
            >
              <item.icon className="h-4 w-4" />
              {item.name}
            </NavLink>
          ))}
        </nav>
        <div className="absolute bottom-4 left-4 right-4">
          <NavLink
            to="/settings"
            className={({ isActive }) =>
              cn(
                'flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors',
                isActive
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
              )
            }
          >
            <Settings className="h-4 w-4" />
            Settings
          </NavLink>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto">
        <div className="container py-6">
          <Outlet />
        </div>
      </main>
    </div>
  )
}
