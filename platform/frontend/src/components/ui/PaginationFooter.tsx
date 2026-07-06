import { ChevronLeft, ChevronRight } from 'lucide-react'

interface PaginationFooterProps {
  total: number
  offset: number
  limit: number
  onPageChange: (offset: number) => void
  isLoading?: boolean
}

export function PaginationFooter({
  total,
  offset,
  limit,
  onPageChange,
  isLoading = false,
}: PaginationFooterProps) {
  const start = total === 0 ? 0 : offset + 1
  const end = Math.min(offset + limit, total)
  const hasPrevious = offset > 0
  const hasNext = offset + limit < total

  return (
    <div className="flex flex-col gap-3 border-t border-border px-4 py-3 text-sm text-muted-foreground sm:flex-row sm:items-center sm:justify-between">
      <span>
        Showing {start}-{end} of {total}
      </span>
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => onPageChange(Math.max(0, offset - limit))}
          disabled={!hasPrevious || isLoading}
          className="inline-flex h-9 items-center gap-1 rounded-md border border-border bg-background px-3 text-sm font-medium text-foreground transition-colors hover:bg-accent disabled:cursor-not-allowed disabled:opacity-50"
        >
          <ChevronLeft className="h-4 w-4" />
          Previous
        </button>
        <button
          type="button"
          onClick={() => onPageChange(offset + limit)}
          disabled={!hasNext || isLoading}
          className="inline-flex h-9 items-center gap-1 rounded-md border border-border bg-background px-3 text-sm font-medium text-foreground transition-colors hover:bg-accent disabled:cursor-not-allowed disabled:opacity-50"
        >
          Next
          <ChevronRight className="h-4 w-4" />
        </button>
      </div>
    </div>
  )
}
