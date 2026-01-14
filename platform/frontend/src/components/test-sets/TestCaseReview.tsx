import { useState, useMemo } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import {
    CheckCircle2,
    XCircle,
    Loader2,
    Search,
    Filter,
    ChevronDown,
    ChevronUp,
    Edit2,
    CheckCheck,
    X,
    Sparkles,
} from 'lucide-react'
import { api, TestCase, TestCaseCreate, BulkReviewRequest } from '@/api/client'
import { cn } from '@/lib/utils'
import { TestCaseDialog } from './TestCaseDialog'

interface TestCaseReviewProps {
    testSetId: string
    testCases: TestCase[]
    onClose: () => void
}

type FilterType = 'all' | 'pending' | 'reviewed'
type SortField = 'created_at' | 'difficulty' | 'quality_score'
type SortOrder = 'asc' | 'desc'

export function TestCaseReview({ testSetId, testCases, onClose }: TestCaseReviewProps) {
    const queryClient = useQueryClient()

    // Filter and sort state
    const [searchQuery, setSearchQuery] = useState('')
    const [filter, setFilter] = useState<FilterType>('all')
    const [sortField, setSortField] = useState<SortField>('created_at')
    const [sortOrder, setSortOrder] = useState<SortOrder>('desc')

    // Selection state
    const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
    const [expandedId, setExpandedId] = useState<string | null>(null)

    // Edit dialog
    const [editingCase, setEditingCase] = useState<TestCase | null>(null)

    // Only show generated, unreviewed cases by default in review mode
    const generatedCases = useMemo(() =>
        testCases.filter(tc => tc.is_generated),
        [testCases]
    )

    // Filter and sort cases
    const filteredCases = useMemo(() => {
        let cases = [...generatedCases]

        // Apply search filter
        if (searchQuery) {
            const query = searchQuery.toLowerCase()
            cases = cases.filter(tc =>
                tc.question.toLowerCase().includes(query) ||
                tc.expected_answer.toLowerCase().includes(query)
            )
        }

        // Apply status filter
        if (filter === 'pending') {
            cases = cases.filter(tc => !tc.is_reviewed)
        } else if (filter === 'reviewed') {
            cases = cases.filter(tc => tc.is_reviewed)
        }

        // Apply sort
        cases.sort((a, b) => {
            let comparison = 0
            switch (sortField) {
                case 'created_at':
                    comparison = new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
                    break
                case 'difficulty':
                    const difficultyOrder = { easy: 0, medium: 1, hard: 2 }
                    comparison = difficultyOrder[a.difficulty] - difficultyOrder[b.difficulty]
                    break
                case 'quality_score':
                    comparison = (a.quality_score || 0) - (b.quality_score || 0)
                    break
            }
            return sortOrder === 'asc' ? comparison : -comparison
        })

        return cases
    }, [generatedCases, searchQuery, filter, sortField, sortOrder])

    // Bulk review mutation
    const bulkReviewMutation = useMutation({
        mutationFn: (data: BulkReviewRequest) => api.testSets.bulkReview(testSetId, data),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['test-set', testSetId] })
            setSelectedIds(new Set())
        },
    })

    // Update case mutation
    const updateCaseMutation = useMutation({
        mutationFn: ({ id, data }: { id: string; data: Partial<TestCaseCreate> }) =>
            api.testSets.updateCase(testSetId, id, data),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['test-set', testSetId] })
            setEditingCase(null)
        },
    })

    const toggleSelectAll = () => {
        if (selectedIds.size === filteredCases.length) {
            setSelectedIds(new Set())
        } else {
            setSelectedIds(new Set(filteredCases.map(tc => tc.id)))
        }
    }

    const toggleSelect = (id: string) => {
        const newSelection = new Set(selectedIds)
        if (newSelection.has(id)) {
            newSelection.delete(id)
        } else {
            newSelection.add(id)
        }
        setSelectedIds(newSelection)
    }

    const handleBulkApprove = async () => {
        if (selectedIds.size === 0) return
        await bulkReviewMutation.mutateAsync({
            test_case_ids: Array.from(selectedIds),
            action: 'approve',
        })
    }

    const handleBulkReject = async () => {
        if (selectedIds.size === 0) return
        if (!confirm(`Are you sure you want to reject and delete ${selectedIds.size} test case(s)?`)) {
            return
        }
        await bulkReviewMutation.mutateAsync({
            test_case_ids: Array.from(selectedIds),
            action: 'reject',
        })
    }

    const handleApproveAll = async () => {
        const pendingIds = filteredCases.filter(tc => !tc.is_reviewed).map(tc => tc.id)
        if (pendingIds.length === 0) return
        await bulkReviewMutation.mutateAsync({
            test_case_ids: pendingIds,
            action: 'approve',
        })
    }

    const pendingCount = generatedCases.filter(tc => !tc.is_reviewed).length
    const reviewedCount = generatedCases.filter(tc => tc.is_reviewed).length

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <div
                className="absolute inset-0 bg-background/80 backdrop-blur-sm animate-in fade-in duration-200"
                onClick={onClose}
            />
            <div className="relative w-full max-w-5xl max-h-[90vh] rounded-xl border border-border bg-card shadow-2xl animate-in zoom-in-95 duration-200 flex flex-col">
                {/* Header */}
                <div className="flex items-center justify-between border-b border-border p-6 flex-shrink-0">
                    <div>
                        <h2 className="text-xl font-bold flex items-center gap-2">
                            <Sparkles className="h-5 w-5 text-primary" />
                            Review Generated Test Cases
                        </h2>
                        <p className="text-sm text-muted-foreground mt-1">
                            {generatedCases.length} generated • {pendingCount} pending review • {reviewedCount} approved
                        </p>
                    </div>
                    <button
                        onClick={onClose}
                        className="rounded-md p-1 hover:bg-muted transition-colors"
                    >
                        <X className="h-5 w-5" />
                    </button>
                </div>

                {/* Toolbar */}
                <div className="flex flex-wrap items-center justify-between gap-4 border-b border-border p-4 bg-muted/30 flex-shrink-0">
                    {/* Left side - Search and filters */}
                    <div className="flex items-center gap-3 flex-1">
                        <div className="relative flex-1 max-w-xs">
                            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                            <input
                                type="text"
                                placeholder="Search questions..."
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                className="w-full rounded-lg border border-input bg-background pl-10 pr-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
                            />
                        </div>

                        <select
                            value={filter}
                            onChange={(e) => setFilter(e.target.value as FilterType)}
                            className="rounded-lg border border-input bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
                        >
                            <option value="all">All ({generatedCases.length})</option>
                            <option value="pending">Pending ({pendingCount})</option>
                            <option value="reviewed">Approved ({reviewedCount})</option>
                        </select>
                    </div>

                    {/* Right side - Actions */}
                    <div className="flex items-center gap-2">
                        {selectedIds.size > 0 ? (
                            <>
                                <span className="text-sm text-muted-foreground">
                                    {selectedIds.size} selected
                                </span>
                                <button
                                    onClick={handleBulkApprove}
                                    disabled={bulkReviewMutation.isPending}
                                    className="flex items-center gap-2 rounded-lg bg-green-600 px-4 py-2 text-sm font-semibold text-white hover:bg-green-700 transition-colors disabled:opacity-50"
                                >
                                    {bulkReviewMutation.isPending ? (
                                        <Loader2 className="h-4 w-4 animate-spin" />
                                    ) : (
                                        <CheckCircle2 className="h-4 w-4" />
                                    )}
                                    Approve
                                </button>
                                <button
                                    onClick={handleBulkReject}
                                    disabled={bulkReviewMutation.isPending}
                                    className="flex items-center gap-2 rounded-lg bg-destructive px-4 py-2 text-sm font-semibold text-destructive-foreground hover:bg-destructive/90 transition-colors disabled:opacity-50"
                                >
                                    <XCircle className="h-4 w-4" />
                                    Reject
                                </button>
                            </>
                        ) : (
                            pendingCount > 0 && (
                                <button
                                    onClick={handleApproveAll}
                                    disabled={bulkReviewMutation.isPending}
                                    className="flex items-center gap-2 rounded-lg bg-green-600 px-4 py-2 text-sm font-semibold text-white hover:bg-green-700 transition-colors disabled:opacity-50"
                                >
                                    {bulkReviewMutation.isPending ? (
                                        <Loader2 className="h-4 w-4 animate-spin" />
                                    ) : (
                                        <CheckCheck className="h-4 w-4" />
                                    )}
                                    Approve All Pending
                                </button>
                            )
                        )}
                    </div>
                </div>

                {/* Content */}
                <div className="flex-1 overflow-y-auto p-4">
                    {filteredCases.length === 0 ? (
                        <div className="flex flex-col items-center justify-center py-20 text-center">
                            <Sparkles className="h-12 w-12 text-muted-foreground/50 mb-4" />
                            <p className="text-lg font-semibold text-muted-foreground">
                                No generated test cases to review
                            </p>
                            <p className="text-sm text-muted-foreground mt-1">
                                Use the Generate button to create test cases from your knowledge base.
                            </p>
                        </div>
                    ) : (
                        <div className="space-y-2">
                            {/* Select all header */}
                            <div className="flex items-center gap-3 px-4 py-2 bg-muted/50 rounded-lg">
                                <input
                                    type="checkbox"
                                    checked={selectedIds.size === filteredCases.length && filteredCases.length > 0}
                                    onChange={toggleSelectAll}
                                    className="h-4 w-4 rounded border-input text-primary focus:ring-primary/50"
                                />
                                <span className="text-sm font-medium text-muted-foreground">
                                    Select all
                                </span>
                            </div>

                            {/* Test case list */}
                            {filteredCases.map((tc) => (
                                <div
                                    key={tc.id}
                                    className={cn(
                                        'rounded-xl border transition-all',
                                        selectedIds.has(tc.id)
                                            ? 'border-primary bg-primary/5'
                                            : 'border-border bg-card hover:border-primary/30'
                                    )}
                                >
                                    {/* Main row */}
                                    <div className="flex items-start gap-4 p-4">
                                        <input
                                            type="checkbox"
                                            checked={selectedIds.has(tc.id)}
                                            onChange={() => toggleSelect(tc.id)}
                                            className="h-4 w-4 mt-1 rounded border-input text-primary focus:ring-primary/50"
                                        />

                                        <div className="flex-1 min-w-0">
                                            <div className="flex items-start justify-between gap-4">
                                                <div className="flex-1 min-w-0">
                                                    <p className="font-medium leading-snug">
                                                        {tc.question}
                                                    </p>
                                                    <p className="text-sm text-muted-foreground mt-1 line-clamp-2">
                                                        {tc.expected_answer}
                                                    </p>
                                                </div>

                                                <div className="flex items-center gap-2 flex-shrink-0">
                                                    {/* Status badge */}
                                                    {tc.is_reviewed ? (
                                                        <span className="flex items-center gap-1 rounded-full bg-green-100 px-2.5 py-0.5 text-[10px] font-bold uppercase text-green-700">
                                                            <CheckCircle2 className="h-3 w-3" />
                                                            Approved
                                                        </span>
                                                    ) : (
                                                        <span className="rounded-full bg-amber-100 px-2.5 py-0.5 text-[10px] font-bold uppercase text-amber-700">
                                                            Pending
                                                        </span>
                                                    )}

                                                    {/* Difficulty badge */}
                                                    <span className={cn(
                                                        'rounded-full px-2.5 py-0.5 text-[10px] font-bold uppercase',
                                                        tc.difficulty === 'easy' && 'bg-green-100 text-green-700',
                                                        tc.difficulty === 'medium' && 'bg-amber-100 text-amber-700',
                                                        tc.difficulty === 'hard' && 'bg-red-100 text-red-700'
                                                    )}>
                                                        {tc.difficulty}
                                                    </span>

                                                    {/* Quality score */}
                                                    {tc.quality_score !== null && (
                                                        <span className={cn(
                                                            'rounded-full px-2.5 py-0.5 text-[10px] font-bold',
                                                            tc.quality_score >= 0.8 ? 'bg-green-100 text-green-700' :
                                                            tc.quality_score >= 0.5 ? 'bg-amber-100 text-amber-700' :
                                                            'bg-red-100 text-red-700'
                                                        )}>
                                                            {Math.round(tc.quality_score * 100)}%
                                                        </span>
                                                    )}
                                                </div>
                                            </div>

                                            {/* Meta info */}
                                            <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
                                                <span>{tc.question_type}</span>
                                                {tc.category && <span>• {tc.category}</span>}
                                            </div>
                                        </div>

                                        {/* Actions */}
                                        <div className="flex items-center gap-1 flex-shrink-0">
                                            <button
                                                onClick={() => setEditingCase(tc)}
                                                className="rounded-md p-1.5 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
                                                title="Edit"
                                            >
                                                <Edit2 className="h-4 w-4" />
                                            </button>
                                            <button
                                                onClick={() => setExpandedId(expandedId === tc.id ? null : tc.id)}
                                                className="rounded-md p-1.5 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
                                                title="Expand"
                                            >
                                                {expandedId === tc.id ? (
                                                    <ChevronUp className="h-4 w-4" />
                                                ) : (
                                                    <ChevronDown className="h-4 w-4" />
                                                )}
                                            </button>
                                        </div>
                                    </div>

                                    {/* Expanded content */}
                                    {expandedId === tc.id && (
                                        <div className="border-t border-border p-4 bg-muted/20 space-y-3 animate-in fade-in slide-in-from-top-2 duration-200">
                                            <div>
                                                <p className="text-[10px] uppercase font-bold tracking-wider text-muted-foreground mb-1">
                                                    Full Answer
                                                </p>
                                                <p className="text-sm">{tc.expected_answer}</p>
                                            </div>
                                            {tc.ground_truth_context && tc.ground_truth_context.length > 0 && (
                                                <div>
                                                    <p className="text-[10px] uppercase font-bold tracking-wider text-muted-foreground mb-1">
                                                        Ground Truth Context ({tc.ground_truth_context.length} chunks)
                                                    </p>
                                                    <div className="space-y-2 max-h-40 overflow-y-auto">
                                                        {tc.ground_truth_context.map((ctx, i) => (
                                                            <p key={i} className="text-xs text-muted-foreground bg-background rounded p-2 border border-border">
                                                                {ctx}
                                                            </p>
                                                        ))}
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="flex items-center justify-end border-t border-border p-4 bg-muted/20 rounded-b-xl flex-shrink-0">
                    <button
                        onClick={onClose}
                        className="rounded-lg bg-primary px-6 py-2 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-all"
                    >
                        Done
                    </button>
                </div>
            </div>

            {/* Edit Dialog */}
            {editingCase && (
                <TestCaseDialog
                    isOpen={!!editingCase}
                    onClose={() => setEditingCase(null)}
                    testCase={editingCase}
                    onSubmit={async (data) => {
                        await updateCaseMutation.mutateAsync({ id: editingCase.id, data })
                    }}
                />
            )}
        </div>
    )
}
