import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
    ArrowLeft,
    Plus,
    Search,
    Edit2,
    Trash2,
    Download,
    Loader2,
    AlertTriangle,
    CheckCircle2,
} from 'lucide-react'
import { api, TestCase, TestCaseCreate } from '@/api/client'
import { cn } from '@/lib/utils'
import { TestCaseDialog } from './TestCaseDialog'

interface TestSetDetailProps {
    testSetId: string
    onBack: () => void
}

export function TestSetDetail({ testSetId, onBack }: TestSetDetailProps) {
    const queryClient = useQueryClient()
    const [searchQuery, setSearchQuery] = useState('')
    const [isCaseDialogOpen, setIsCaseDialogOpen] = useState(false)
    const [editingCase, setEditingCase] = useState<TestCase | undefined>(undefined)

    const { data: testSet, isLoading, isError } = useQuery({
        queryKey: ['test-set', testSetId],
        queryFn: () => api.testSets.get(testSetId),
    })

    const addCaseMutation = useMutation({
        mutationFn: (data: TestCaseCreate) => api.testSets.addCase(testSetId, data),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['test-set', testSetId] })
        },
    })

    const updateCaseMutation = useMutation({
        mutationFn: ({ id, data }: { id: string, data: Partial<TestCaseCreate> }) =>
            api.testSets.updateCase(testSetId, id, data),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['test-set', testSetId] })
        },
    })

    const deleteCaseMutation = useMutation({
        mutationFn: (id: string) => api.testSets.deleteCase(testSetId, id),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['test-set', testSetId] })
        },
    })

    const onExport = async () => {
        try {
            const response = await api.testSets.export(testSetId)
            const url = window.URL.createObjectURL(new Blob([response.data]))
            const link = document.createElement('a')
            link.href = url
            link.setAttribute('download', `${testSet?.data.name || 'test-set'}.json`)
            document.body.appendChild(link)
            link.click()
            link.remove()
        } catch (error) {
            console.error('Failed to export test set:', error)
        }
    }

    if (isLoading) {
        return (
            <div className="flex justify-center py-20">
                <Loader2 className="h-8 w-8 animate-spin text-primary/50" />
            </div>
        )
    }

    if (isError || !testSet) {
        return (
            <div className="flex flex-col items-center justify-center py-20">
                <AlertTriangle className="h-12 w-12 text-destructive mb-4" />
                <p className="text-lg font-semibold">Failed to load test set</p>
                <button onClick={onBack} className="mt-4 text-primary hover:underline flex items-center gap-2">
                    <ArrowLeft className="h-4 w-4" />
                    Back to Test Sets
                </button>
            </div>
        )
    }

    const cases = testSet.data.test_cases || []
    const filteredCases = cases.filter(c =>
        c.question.toLowerCase().includes(searchQuery.toLowerCase()) ||
        c.expected_answer.toLowerCase().includes(searchQuery.toLowerCase())
    )

    return (
        <div className="space-y-6">
            <div className="flex items-center gap-4">
                <button
                    onClick={onBack}
                    className="flex h-10 w-10 items-center justify-center rounded-full border border-border bg-card hover:bg-muted transition-colors"
                >
                    <ArrowLeft className="h-5 w-5" />
                </button>
                <div>
                    <h2 className="text-2xl font-bold tracking-tight">{testSet.data.name}</h2>
                    <p className="text-sm text-muted-foreground">{testSet.data.description || 'No description provided.'}</p>
                </div>
                <div className="ml-auto flex items-center gap-2">
                    <button
                        onClick={onExport}
                        className="flex items-center gap-2 rounded-lg border border-border bg-card px-4 py-2 text-sm font-semibold hover:bg-muted transition-colors"
                    >
                        <Download className="h-4 w-4" />
                        Export
                    </button>
                    <button
                        onClick={() => {
                            setEditingCase(undefined)
                            setIsCaseDialogOpen(true)
                        }}
                        className="flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-all shadow-md active:scale-95"
                    >
                        <Plus className="h-4 w-4" />
                        Add Case
                    </button>
                </div>
            </div>

            {/* Stats / Filters */}
            <div className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-border bg-card/50 p-4">
                <div className="flex items-center gap-6">
                    <div className="flex flex-col">
                        <span className="text-[10px] uppercase font-bold tracking-widest text-muted-foreground">Total Cases</span>
                        <span className="text-lg font-bold">{cases.length}</span>
                    </div>
                </div>

                <div className="relative flex-1 max-w-md">
                    <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                    <input
                        type="text"
                        placeholder="Search questions or answers..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="w-full rounded-lg border border-input bg-background pl-10 pr-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all"
                    />
                </div>
            </div>

            {/* Cases Table */}
            <div className="overflow-hidden rounded-xl border border-border bg-card shadow-sm">
                <div className="overflow-x-auto">
                    <table className="w-full text-left text-sm border-collapse">
                        <thead className="bg-muted/50 border-b border-border">
                            <tr>
                                <th className="px-6 py-4 font-semibold">Question & Expected Answer</th>
                                <th className="px-6 py-4 font-semibold">type</th>
                                <th className="px-6 py-4 font-semibold">Difficulty</th>
                                <th className="px-6 py-4 font-semibold">Status</th>
                                <th className="px-6 py-4 font-semibold text-right">Actions</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-border">
                            {filteredCases.length === 0 ? (
                                <tr>
                                    <td colSpan={5} className="px-6 py-20 text-center text-muted-foreground italic">
                                        No test cases found.
                                    </td>
                                </tr>
                            ) : (
                                filteredCases.map((c) => (
                                    <tr key={c.id} className="hover:bg-muted/30 transition-colors group">
                                        <td className="px-6 py-4">
                                            <div className="space-y-1.5 max-w-xl">
                                                <div className="font-medium text-foreground leading-snug line-clamp-2" title={c.question}>
                                                    {c.question}
                                                </div>
                                                <div className="text-xs text-muted-foreground leading-relaxed line-clamp-2" title={c.expected_answer}>
                                                    {c.expected_answer}
                                                </div>
                                            </div>
                                        </td>
                                        <td className="px-6 py-4">
                                            <span className="inline-flex rounded-full bg-muted px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
                                                {c.question_type}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4">
                                            <span className={cn(
                                                "inline-flex rounded-full px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider",
                                                c.difficulty === 'easy' ? "bg-green-100 text-green-700" :
                                                    c.difficulty === 'medium' ? "bg-amber-100 text-amber-700" :
                                                        "bg-red-100 text-red-700"
                                            )}>
                                                {c.difficulty}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4">
                                            {c.is_reviewed ? (
                                                <div className="flex items-center gap-1.5 text-green-600">
                                                    <CheckCircle2 className="h-4 w-4" />
                                                    <span className="text-xs font-medium">Reviewed</span>
                                                </div>
                                            ) : (
                                                <div className="flex items-center gap-1.5 text-muted-foreground">
                                                    <Loader2 className="h-3.5 w-3.5" />
                                                    <span className="text-xs font-medium">Pending</span>
                                                </div>
                                            )}
                                        </td>
                                        <td className="px-6 py-4 text-right">
                                            <div className="flex justify-end gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                                                <button
                                                    onClick={() => {
                                                        setEditingCase(c)
                                                        setIsCaseDialogOpen(true)
                                                    }}
                                                    className="rounded-md p-1.5 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
                                                >
                                                    <Edit2 className="h-4 w-4" />
                                                </button>
                                                <button
                                                    onClick={() => {
                                                        if (confirm('Are you sure you want to delete this test case?')) {
                                                            deleteCaseMutation.mutate(c.id)
                                                        }
                                                    }}
                                                    className="rounded-md p-1.5 text-muted-foreground hover:bg-destructive/10 hover:text-destructive transition-colors"
                                                >
                                                    <Trash2 className="h-4 w-4" />
                                                </button>
                                            </div>
                                        </td>
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
                </div>
            </div>

            <TestCaseDialog
                isOpen={isCaseDialogOpen}
                onClose={() => setIsCaseDialogOpen(false)}
                testCase={editingCase}
                onSubmit={async (data) => {
                    if (editingCase) {
                        await updateCaseMutation.mutateAsync({ id: editingCase.id, data })
                    } else {
                        await addCaseMutation.mutateAsync(data)
                    }
                }}
            />
        </div>
    )
}
