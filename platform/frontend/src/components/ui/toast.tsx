import { createContext, useContext, useState, useCallback, ReactNode } from 'react'
import { X, CheckCircle2, AlertCircle, Info, AlertTriangle } from 'lucide-react'
import { cn } from '@/lib/utils'

export type ToastType = 'success' | 'error' | 'info' | 'warning'

export interface Toast {
  id: string
  type: ToastType
  title: string
  description?: string
  duration?: number
}

interface ToastContextType {
  toasts: Toast[]
  addToast: (toast: Omit<Toast, 'id'>) => void
  removeToast: (id: string) => void
}

const ToastContext = createContext<ToastContextType | undefined>(undefined)

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([])

  const addToast = useCallback((toast: Omit<Toast, 'id'>) => {
    const id = crypto.randomUUID()
    const newToast = { ...toast, id }

    setToasts((prev) => [...prev, newToast])

    // Auto-remove after duration (default 5000ms)
    const duration = toast.duration ?? 5000
    if (duration > 0) {
      setTimeout(() => {
        setToasts((prev) => prev.filter((t) => t.id !== id))
      }, duration)
    }
  }, [])

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id))
  }, [])

  return (
    <ToastContext.Provider value={{ toasts, addToast, removeToast }}>
      {children}
      <ToastContainer toasts={toasts} onRemove={removeToast} />
    </ToastContext.Provider>
  )
}

export function useToast() {
  const context = useContext(ToastContext)
  if (!context) {
    throw new Error('useToast must be used within a ToastProvider')
  }

  const { addToast } = context

  return {
    toast: addToast,
    success: (title: string, description?: string) =>
      addToast({ type: 'success', title, description }),
    error: (title: string, description?: string) =>
      addToast({ type: 'error', title, description }),
    info: (title: string, description?: string) =>
      addToast({ type: 'info', title, description }),
    warning: (title: string, description?: string) =>
      addToast({ type: 'warning', title, description }),
  }
}

function ToastContainer({ toasts, onRemove }: { toasts: Toast[]; onRemove: (id: string) => void }) {
  if (toasts.length === 0) return null

  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 max-w-sm">
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} onRemove={onRemove} />
      ))}
    </div>
  )
}

function ToastItem({ toast, onRemove }: { toast: Toast; onRemove: (id: string) => void }) {
  const icons = {
    success: CheckCircle2,
    error: AlertCircle,
    info: Info,
    warning: AlertTriangle,
  }

  const colors = {
    success: 'bg-green-500/10 border-green-500/20 text-green-600',
    error: 'bg-red-500/10 border-red-500/20 text-red-600',
    info: 'bg-blue-500/10 border-blue-500/20 text-blue-600',
    warning: 'bg-amber-500/10 border-amber-500/20 text-amber-600',
  }

  const Icon = icons[toast.type]

  return (
    <div
      className={cn(
        'flex items-start gap-3 rounded-lg border p-4 shadow-lg backdrop-blur-sm animate-in slide-in-from-right-5 fade-in duration-300',
        'bg-card/95',
        colors[toast.type]
      )}
    >
      <Icon className="h-5 w-5 shrink-0 mt-0.5" />
      <div className="flex-1 min-w-0">
        <p className="text-sm font-semibold text-foreground">{toast.title}</p>
        {toast.description && (
          <p className="mt-1 text-xs text-muted-foreground">{toast.description}</p>
        )}
      </div>
      <button
        onClick={() => onRemove(toast.id)}
        className="shrink-0 rounded p-1 hover:bg-muted/50 transition-colors text-muted-foreground hover:text-foreground"
      >
        <X className="h-4 w-4" />
      </button>
    </div>
  )
}
