import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ArrowLeft, Loader2, AlertCircle, BarChart3, Table2, ListChecks, Settings2, Scale, Download } from 'lucide-react'
import { cn } from '@/lib/utils'
import { api } from '@/api/client'
import { buildMembers } from './compare-utils'
import { MetricMatrix } from './MetricMatrix'
import { ComparisonCharts } from './ComparisonCharts'
import { PerQuestionTable } from './PerQuestionTable'
import { ConfigDiff } from './ConfigDiff'
import { LegalRagBenchComparison } from './LegalRagBenchComparison'

interface ComparisonDetailProps {
    comparisonId: string
    onBack: () => void
}

type Section = 'metrics' | 'legal' | 'charts' | 'questions' | 'config'

const SECTIONS: { id: Section; name: string; icon: typeof Table2 }[] = [
    { id: 'metrics', name: 'Metrics', icon: Table2 },
    { id: 'legal', name: 'Legal RAG Bench', icon: Scale },
    { id: 'charts', name: 'Charts', icon: BarChart3 },
    { id: 'questions', name: 'Per-question', icon: ListChecks },
    { id: 'config', name: 'Config diff', icon: Settings2 },
]

/** Article-ready export links surfaced when a comparison has Legal RAG Bench data. */
const EXPORTS: { label: string; format: 'markdown' | 'csv' | 'jsonl'; table?: 'headline' | 'taxonomy' }[] = [
    { label: 'Markdown', format: 'markdown' },
    { label: 'Headline CSV', format: 'csv', table: 'headline' },
    { label: 'Taxonomy CSV', format: 'csv', table: 'taxonomy' },
    { label: 'JSONL', format: 'jsonl' },
]

export function ComparisonDetail({ comparisonId, onBack }: ComparisonDetailProps) {
    const [section, setSection] = useState<Section>('metrics')
    const [baselineId, setBaselineId] = useState<string | null>(null)

    const { data, isLoading, isError } = useQuery({
        queryKey: ['comparison', comparisonId],
        queryFn: () => api.comparisons.get(comparisonId),
    })

    if (isLoading) {
        return (
            <div className="flex justify-center py-20">
                <Loader2 className="h-8 w-8 animate-spin text-primary/50" />
            </div>
        )
    }

    if (isError || !data?.data) {
        return (
            <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-border py-20">
                <AlertCircle className="h-10 w-10 text-destructive/50" />
                <p className="mt-4 font-medium">Could not load this comparison.</p>
                <button onClick={onBack} className="mt-4 text-sm text-primary hover:underline">Back to comparisons</button>
            </div>
        )
    }

    const comparison = data.data
    const members = buildMembers(comparison.aggregate_metrics)
    const storedBaselineId = comparison.baseline_evaluation_id
    const activeBaselineId = baselineId ?? storedBaselineId
    const hasLegalRagBench = members.some((m) => m.legalRagBench)
    const sections = SECTIONS.filter((s) => s.id !== 'legal' || hasLegalRagBench)
    const activeSection = section === 'legal' && !hasLegalRagBench ? 'metrics' : section

    return (
        <div className="space-y-6">
            <button onClick={onBack} className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors">
                <ArrowLeft className="h-4 w-4" />
                Back to Comparisons
            </button>

            <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
                <div>
                    <h2 className="text-xl font-semibold">{comparison.name || 'Untitled comparison'}</h2>
                    {comparison.description && <p className="text-sm text-muted-foreground">{comparison.description}</p>}
                    <p className="mt-1 text-xs text-muted-foreground">{members.length} evaluations compared</p>
                </div>
                <div className="flex flex-wrap items-end gap-4">
                    {hasLegalRagBench && (
                        <div className="flex flex-col gap-1">
                            <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Export</span>
                            <div className="flex flex-wrap gap-2">
                                {EXPORTS.map((x) => (
                                    <a
                                        key={x.label}
                                        href={api.comparisons.exportUrl(comparisonId, x.format, x.table)}
                                        download
                                        className="flex items-center gap-1.5 rounded-lg border border-border bg-card px-2.5 py-1.5 text-xs font-medium hover:border-primary hover:text-primary transition-colors"
                                    >
                                        <Download className="h-3.5 w-3.5" />
                                        {x.label}
                                    </a>
                                ))}
                            </div>
                        </div>
                    )}
                    <label className="flex items-center gap-2 text-sm">
                        <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Baseline</span>
                        <select
                            value={activeBaselineId}
                            onChange={(e) => setBaselineId(e.target.value)}
                            className="rounded-lg border border-border bg-card px-3 py-1.5 text-sm font-medium focus:border-primary focus:outline-none"
                        >
                            {members.map((m) => (
                                <option key={m.id} value={m.id}>{m.label}</option>
                            ))}
                        </select>
                    </label>
                </div>
            </div>

            <div className="flex gap-6 border-b border-border">
                {sections.map((s) => (
                    <button
                        key={s.id}
                        onClick={() => setSection(s.id)}
                        className={cn(
                            'flex items-center gap-2 pb-3 text-sm font-bold border-b-2 transition-all',
                            activeSection === s.id ? 'border-primary text-primary' : 'border-transparent text-muted-foreground hover:text-foreground',
                        )}
                    >
                        <s.icon className="h-4 w-4" />
                        {s.name}
                    </button>
                ))}
            </div>

            <div>
                {activeSection === 'metrics' && <MetricMatrix members={members} baselineId={activeBaselineId} />}
                {activeSection === 'legal' && <LegalRagBenchComparison members={members} />}
                {activeSection === 'charts' && <ComparisonCharts members={members} />}
                {activeSection === 'questions' && (
                    <PerQuestionTable
                        members={members}
                        storedBaselineId={storedBaselineId}
                        baselineId={activeBaselineId}
                        deltas={comparison.per_question_deltas ?? []}
                    />
                )}
                {activeSection === 'config' && <ConfigDiff members={members} />}
            </div>
        </div>
    )
}
