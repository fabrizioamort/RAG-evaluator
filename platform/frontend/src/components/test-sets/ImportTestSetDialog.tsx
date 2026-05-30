import { useState } from 'react'
import type { ChangeEvent, FormEvent } from 'react'
import { AlertCircle, FileUp, Loader2, X } from 'lucide-react'

type ImportedCase = {
    question?: unknown
    expected_answer?: unknown
    ground_truth_context?: unknown
    difficulty?: unknown
    category?: unknown
    question_type?: unknown
}

type ImportedTestSet = {
    metadata?: {
        dataset?: unknown
        version?: unknown
        description?: unknown
    }
    test_cases?: ImportedCase[]
    name?: unknown
    description?: unknown
    tags?: unknown
}

type PlatformQuestionType = 'factual' | 'reasoning' | 'comparison' | 'multi_hop'
type PlatformDifficulty = 'easy' | 'medium' | 'hard'

interface ImportTestSetPayload {
    name: string
    description?: string
    tags: string[]
    test_cases: {
        question: string
        expected_answer: string
        ground_truth_context: string[]
        difficulty: PlatformDifficulty
        category?: string
        question_type: PlatformQuestionType
    }[]
}

interface ImportTestSetDialogProps {
    isOpen: boolean
    onClose: () => void
    onSubmit: (data: ImportTestSetPayload) => Promise<void>
}

function asString(value: unknown): string {
    return typeof value === 'string' ? value.trim() : ''
}

function normalizeDifficulty(value: unknown): PlatformDifficulty {
    return value === 'easy' || value === 'hard' ? value : 'medium'
}

function normalizeQuestionType(value: unknown): PlatformQuestionType {
    if (value === 'reasoning' || value === 'comparison' || value === 'multi_hop') {
        return value
    }
    return 'factual'
}

function normalizeContext(value: unknown): string[] {
    if (!Array.isArray(value)) {
        return []
    }
    return value.map(asString).filter(Boolean)
}

function deriveName(data: ImportedTestSet): string {
    const explicitName = asString(data.name)
    if (explicitName) {
        return explicitName
    }

    const dataset = asString(data.metadata?.dataset)
    const version = asString(data.metadata?.version)
    if (dataset && version) {
        return `${dataset} - ${version}`
    }
    if (dataset) {
        return dataset
    }
    return 'Imported Test Set'
}

function toImportPayload(data: ImportedTestSet): ImportTestSetPayload {
    const sourceCases = data.test_cases
    if (!Array.isArray(sourceCases) || sourceCases.length === 0) {
        throw new Error('The JSON file must contain a non-empty test_cases array.')
    }

    const testCases = sourceCases.map((testCase, index) => {
        const question = asString(testCase.question)
        const expectedAnswer = asString(testCase.expected_answer)
        if (!question || !expectedAnswer) {
            throw new Error(`Test case ${index + 1} is missing question or expected_answer.`)
        }

        return {
            question,
            expected_answer: expectedAnswer,
            ground_truth_context: normalizeContext(testCase.ground_truth_context),
            difficulty: normalizeDifficulty(testCase.difficulty),
            category: asString(testCase.category) || undefined,
            question_type: normalizeQuestionType(testCase.question_type),
        }
    })

    const explicitTags = Array.isArray(data.tags) ? data.tags.map(asString).filter(Boolean) : []
    const dataset = asString(data.metadata?.dataset)
    const version = asString(data.metadata?.version)

    return {
        name: deriveName(data),
        description:
            asString(data.description) ||
            asString(data.metadata?.description) ||
            (dataset ? `Imported from ${dataset}${version ? ` (${version})` : ''}.` : undefined),
        tags: explicitTags.length > 0 ? explicitTags : [dataset, version].filter(Boolean),
        test_cases: testCases,
    }
}

export function ImportTestSetDialog({ isOpen, onClose, onSubmit }: ImportTestSetDialogProps) {
    const [fileName, setFileName] = useState('')
    const [payload, setPayload] = useState<ImportTestSetPayload | null>(null)
    const [parseError, setParseError] = useState('')
    const [isSubmitting, setIsSubmitting] = useState(false)

    if (!isOpen) return null

    const reset = () => {
        setFileName('')
        setPayload(null)
        setParseError('')
    }

    const handleClose = () => {
        if (isSubmitting) return
        reset()
        onClose()
    }

    const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0]
        setParseError('')
        setPayload(null)
        setFileName(file?.name || '')

        if (!file) {
            return
        }

        try {
            const text = await file.text()
            const parsed = JSON.parse(text) as ImportedTestSet
            setPayload(toImportPayload(parsed))
        } catch (err) {
            setParseError(err instanceof Error ? err.message : 'Could not parse the selected JSON file.')
        }
    }

    const handleSubmit = async (event: FormEvent) => {
        event.preventDefault()
        if (!payload) return

        setIsSubmitting(true)
        try {
            await onSubmit(payload)
            reset()
            onClose()
        } catch (err) {
            console.error('Failed to import test set:', err)
        } finally {
            setIsSubmitting(false)
        }
    }

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <div
                className="absolute inset-0 bg-background/80 backdrop-blur-sm transition-opacity animate-in fade-in"
                onClick={handleClose}
            />

            <div className="relative w-full max-w-xl rounded-xl border border-border bg-card p-8 shadow-2xl animate-in zoom-in-95 duration-200">
                <div className="mb-6 flex items-center justify-between">
                    <div>
                        <h2 className="text-2xl font-bold tracking-tight">Import Test Set</h2>
                        <p className="mt-1 text-sm text-muted-foreground">
                            Upload a RAG Evaluator JSON test set.
                        </p>
                    </div>
                    <button
                        onClick={handleClose}
                        disabled={isSubmitting}
                        className="rounded-full p-2 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors disabled:opacity-50"
                    >
                        <X className="h-5 w-5" />
                    </button>
                </div>

                <form onSubmit={handleSubmit} className="space-y-6">
                    <label className="flex min-h-[180px] cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed border-border bg-background/60 px-6 py-8 text-center transition-colors hover:bg-accent">
                        <FileUp className="h-9 w-9 text-primary" />
                        <span className="mt-4 text-sm font-semibold">
                            {fileName || 'Choose a JSON file'}
                        </span>
                        <span className="mt-1 text-xs text-muted-foreground">
                            Supports files like data/legal_rag_bench/subset/test_set.json
                        </span>
                        <input
                            type="file"
                            accept="application/json,.json"
                            onChange={handleFileChange}
                            className="sr-only"
                            disabled={isSubmitting}
                        />
                    </label>

                    {parseError && (
                        <div className="flex gap-2 rounded-lg border border-destructive/20 bg-destructive/10 p-3 text-sm text-destructive">
                            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                            <span>{parseError}</span>
                        </div>
                    )}

                    {payload && (
                        <div className="rounded-lg border border-border bg-muted/30 p-4">
                            <div className="text-sm font-semibold">{payload.name}</div>
                            <div className="mt-1 text-xs text-muted-foreground">
                                {payload.test_cases.length} test cases
                                {payload.tags.length > 0 ? ` - ${payload.tags.join(', ')}` : ''}
                            </div>
                        </div>
                    )}

                    <div className="flex justify-end gap-3 border-t border-border pt-4">
                        <button
                            type="button"
                            onClick={handleClose}
                            disabled={isSubmitting}
                            className="rounded-lg px-6 py-2.5 text-sm font-semibold hover:bg-muted transition-colors disabled:opacity-50"
                        >
                            Cancel
                        </button>
                        <button
                            type="submit"
                            disabled={!payload || isSubmitting}
                            className="flex items-center gap-2 rounded-lg bg-primary px-8 py-2.5 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-all shadow-md active:scale-95 disabled:pointer-events-none disabled:opacity-50"
                        >
                            {isSubmitting ? (
                                <>
                                    <Loader2 className="h-4 w-4 animate-spin" />
                                    Importing...
                                </>
                            ) : (
                                'Import'
                            )}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    )
}
