import { FileText, Plus, Trash2, Download } from 'lucide-react'
import { TestSet } from '@/api/client'

interface TestSetListProps {
    testSets: TestSet[]
    onCreateClick: () => void
    onViewDetail: (id: string) => void
    onDelete: (id: string) => void
    onExport: (id: string) => void
}

export function TestSetList({
    testSets,
    onCreateClick,
    onViewDetail,
    onDelete,
    onExport
}: TestSetListProps) {
    if (testSets.length === 0) {
        return (
            <div className="flex min-h-[400px] flex-col items-center justify-center rounded-xl border border-dashed border-border bg-card/50 p-8 text-center transition-all hover:bg-card/80">
                <div className="flex h-16 w-16 items-center justify-center rounded-full bg-primary/10 shadow-inner">
                    <FileText className="h-8 w-8 text-primary" />
                </div>
                <h3 className="mt-4 text-xl font-semibold tracking-tight">No test sets yet</h3>
                <p className="mt-2 max-w-sm text-sm text-muted-foreground leading-relaxed">
                    Create your first test set to start evaluating your RAG configurations.
                </p>
                <button
                    onClick={onCreateClick}
                    className="mt-6 flex items-center gap-2 rounded-lg bg-primary px-6 py-2.5 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-all shadow-md active:scale-95"
                >
                    <Plus className="h-4 w-4" />
                    Create Test Set
                </button>
            </div>
        )
    }

    return (
        <div className="space-y-4">
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {testSets.map((ts) => (
                    <div
                        key={ts.id}
                        onClick={() => onViewDetail(ts.id)}
                        className="group relative flex flex-col justify-between overflow-hidden rounded-xl border border-border bg-card p-6 shadow-sm transition-all hover:-translate-y-1 hover:shadow-md cursor-pointer"
                    >
                        <div className="space-y-3">
                            <div className="flex items-start justify-between">
                                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 transition-colors group-hover:bg-primary/20">
                                    <FileText className="h-5 w-5 text-primary" />
                                </div>
                                <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation()
                                            onExport(ts.id)
                                        }}
                                        className="rounded-md p-1.5 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
                                        title="Export CSV"
                                    >
                                        <Download className="h-4 w-4" />
                                    </button>
                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation()
                                            if (confirm('Are you sure you want to delete this test set?')) {
                                                onDelete(ts.id)
                                            }
                                        }}
                                        className="rounded-md p-1.5 text-muted-foreground hover:bg-destructive/10 hover:text-destructive transition-colors"
                                        title="Delete"
                                    >
                                        <Trash2 className="h-4 w-4" />
                                    </button>
                                </div>
                            </div>
                            <div>
                                <h3 className="font-semibold text-lg leading-tight group-hover:text-primary transition-colors">
                                    {ts.name}
                                </h3>
                                <p className="mt-1.5 line-clamp-2 text-sm text-muted-foreground leading-relaxed">
                                    {ts.description || 'No description provided.'}
                                </p>
                            </div>
                        </div>

                        <div className="mt-6 flex items-center justify-between border-t border-border pt-4 text-xs font-medium text-muted-foreground">
                            <div className="flex items-center gap-4">
                                <div className="flex flex-col">
                                    <span className="text-[10px] uppercase tracking-wider opacity-60">Test Cases</span>
                                    <span className="text-sm font-bold text-foreground">{ts.test_case_count}</span>
                                </div>
                            </div>
                            <div className="flex flex-col items-end">
                                <span className="text-[10px] uppercase tracking-wider opacity-60">Created</span>
                                <span>{new Date(ts.created_at).toLocaleDateString()}</span>
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    )
}
