import React, { useEffect } from 'react'
import { X } from 'lucide-react'
import { cn } from '@/lib/utils'

interface DialogShellProps {
  isOpen: boolean
  title: React.ReactNode
  description?: React.ReactNode
  icon?: React.ReactNode
  onClose: () => void
  children: React.ReactNode
  footer?: React.ReactNode
  size?: 'sm' | 'md' | 'lg' | 'xl'
  closeDisabled?: boolean
  bodyClassName?: string
  headerExtra?: React.ReactNode
}

const sizeClass = {
  sm: 'max-w-md',
  md: 'max-w-lg',
  lg: 'max-w-2xl',
  xl: 'max-w-3xl',
}

export function DialogShell({
  isOpen,
  title,
  description,
  icon,
  onClose,
  children,
  footer,
  size = 'md',
  closeDisabled = false,
  bodyClassName,
  headerExtra,
}: DialogShellProps) {
  useEffect(() => {
    if (!isOpen || closeDisabled) return

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [closeDisabled, isOpen, onClose])

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div
        className="absolute inset-0 bg-background/80 backdrop-blur-sm animate-in fade-in duration-200"
        onClick={closeDisabled ? undefined : onClose}
      />
      <div
        role="dialog"
        aria-modal="true"
        className={cn(
          'relative flex max-h-[calc(100vh-2rem)] w-full flex-col overflow-hidden rounded-xl border border-border bg-card shadow-2xl animate-in zoom-in-95 duration-200',
          sizeClass[size],
        )}
      >
        <div className="flex items-start justify-between gap-4 border-b border-border p-6">
          <div className="min-w-0">
            <h2 className="flex items-center gap-2 text-xl font-bold">
              {icon}
              {title}
            </h2>
            {description && (
              <p className="mt-1 text-sm text-muted-foreground">{description}</p>
            )}
          </div>
          <div className="flex shrink-0 items-center gap-2">
            {headerExtra}
            <button
              type="button"
              onClick={onClose}
              disabled={closeDisabled}
              aria-label="Close dialog"
              className="rounded-md p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
        </div>

        <div className={cn('flex-1 overflow-y-auto p-6', bodyClassName)}>
          {children}
        </div>

        {footer && (
          <div className="border-t border-border bg-muted/20 p-6">
            {footer}
          </div>
        )}
      </div>
    </div>
  )
}
