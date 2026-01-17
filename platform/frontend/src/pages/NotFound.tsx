import { Link } from 'react-router-dom'
import { Home } from 'lucide-react'

export function NotFound() {
  return (
    <div className="flex min-h-[50vh] flex-col items-center justify-center">
      <h1 className="text-6xl font-bold text-muted-foreground">404</h1>
      <p className="mt-4 text-xl text-muted-foreground">Page not found</p>
      <p className="mt-2 text-sm text-muted-foreground">
        The page you're looking for doesn't exist or has been moved.
      </p>
      <Link
        to="/"
        className="mt-6 flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90"
      >
        <Home className="h-4 w-4" />
        Back to Dashboard
      </Link>
    </div>
  )
}
