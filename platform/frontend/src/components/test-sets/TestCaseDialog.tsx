import { useState, useEffect } from 'react'
import { Loader2, Save } from 'lucide-react'
import { TestCase, TestCaseCreate } from '@/api/client'
import { DialogShell } from '@/components/ui/DialogShell'

interface TestCaseDialogProps {
    isOpen: boolean
    onClose: () => void
    onSubmit: (data: TestCaseCreate) => Promise<void>
    testCase?: TestCase // If provided, we are editing
}

export function TestCaseDialog({ isOpen, onClose, onSubmit, testCase }: TestCaseDialogProps) {
    const [formData, setFormData] = useState<TestCaseCreate>({
        question: '',
        expected_answer: '',
        difficulty: 'medium',
        category: '',
        question_type: 'factual',
    })
    const [isSubmitting, setIsSubmitting] = useState(false)

    useEffect(() => {
        if (testCase) {
            setFormData({
                question: testCase.question,
                expected_answer: testCase.expected_answer,
                difficulty: testCase.difficulty,
                category: testCase.category || '',
                question_type: testCase.question_type,
            })
        } else {
            setFormData({
                question: '',
                expected_answer: '',
                difficulty: 'medium',
                category: '',
                question_type: 'factual',
            })
        }
    }, [testCase, isOpen])

    if (!isOpen) return null

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        if (!formData.question || !formData.expected_answer) return

        setIsSubmitting(true)
        try {
            await onSubmit(formData)
            onClose()
        } catch (error) {
            console.error('Failed to save test case:', error)
        } finally {
            setIsSubmitting(false)
        }
    }

    return (
        <DialogShell
            isOpen={isOpen}
            title={testCase ? 'Edit Test Case' : 'Add Test Case'}
            description={testCase ? 'Update the details of this test case.' : 'Create a new manual test case for evaluation.'}
            onClose={onClose}
            size="lg"
            closeDisabled={isSubmitting}
            footer={(
                <div className="flex justify-end gap-3">
                    <button
                        type="button"
                        onClick={onClose}
                        disabled={isSubmitting}
                        className="rounded-lg px-6 py-2.5 text-sm font-semibold hover:bg-muted transition-colors disabled:opacity-50"
                    >
                        Cancel
                    </button>
                    <button
                        type="submit"
                        form="test-case-form"
                        disabled={isSubmitting}
                        className="flex items-center gap-2 rounded-lg bg-primary px-8 py-2.5 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-all shadow-md active:scale-95 disabled:opacity-50"
                    >
                        {isSubmitting ? (
                            <>
                                <Loader2 className="h-4 w-4 animate-spin" />
                                Saving...
                            </>
                        ) : (
                            <>
                                <Save className="h-4 w-4" />
                                Save Case
                            </>
                        )}
                    </button>
                </div>
            )}
        >
                <form id="test-case-form" onSubmit={handleSubmit} className="space-y-6">
                    <div className="space-y-2">
                        <label className="text-sm font-semibold text-foreground">Question</label>
                        <textarea
                            value={formData.question}
                            onChange={(e) => setFormData({ ...formData, question: e.target.value })}
                            placeholder="What is the main topic of the document?"
                            className="w-full min-h-[100px] rounded-lg border border-input bg-background px-4 py-3 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all resize-none"
                            required
                        />
                    </div>

                    <div className="space-y-2">
                        <label className="text-sm font-semibold text-foreground">Expected Answer</label>
                        <textarea
                            value={formData.expected_answer}
                            onChange={(e) => setFormData({ ...formData, expected_answer: e.target.value })}
                            placeholder="The main topic is..."
                            className="w-full min-h-[100px] rounded-lg border border-input bg-background px-4 py-3 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all resize-none"
                            required
                        />
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="space-y-2">
                            <label className="text-sm font-semibold text-foreground">Difficulty</label>
                            <select
                                value={formData.difficulty}
                                onChange={(e) => setFormData({ ...formData, difficulty: e.target.value as TestCaseCreate['difficulty'] })}
                                className="w-full rounded-lg border border-input bg-background px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all"
                            >
                                <option value="easy">Easy</option>
                                <option value="medium">Medium</option>
                                <option value="hard">Hard</option>
                            </select>
                        </div>

                        <div className="space-y-2">
                            <label className="text-sm font-semibold text-foreground">Question Type</label>
                            <select
                                value={formData.question_type}
                                onChange={(e) => setFormData({ ...formData, question_type: e.target.value as TestCaseCreate['question_type'] })}
                                className="w-full rounded-lg border border-input bg-background px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all"
                            >
                                <option value="factual">Factual</option>
                                <option value="reasoning">Reasoning</option>
                                <option value="comparison">Comparison</option>
                                <option value="multi_hop">Multi-hop</option>
                            </select>
                        </div>

                        <div className="space-y-2">
                            <label className="text-sm font-semibold text-foreground">Category (Optional)</label>
                            <input
                                type="text"
                                value={formData.category}
                                onChange={(e) => setFormData({ ...formData, category: e.target.value })}
                                placeholder="e.g., General Information"
                                className="w-full rounded-lg border border-input bg-background px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all"
                            />
                        </div>
                    </div>
                </form>
        </DialogShell>
    )
}
