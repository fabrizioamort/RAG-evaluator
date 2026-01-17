import { createContext, useContext } from 'react'
import { Toast } from './toast-types'

interface ToastContextType {
    toasts: Toast[]
    addToast: (toast: Omit<Toast, 'id'>) => void
    removeToast: (id: string) => void
}

export const ToastContext = createContext<ToastContextType | undefined>(undefined)

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
