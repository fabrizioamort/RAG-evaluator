import { useQueries } from '@tanstack/react-query'
import { Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'
import { api, RunManifest } from '@/api/client'
import { ComparisonMember } from './compare-utils'

interface ConfigDiffProps {
    members: ComparisonMember[]
}

/** Flatten the parts of a manifest we want to diff into a label -> displayable string map. */
function flattenManifest(manifest: RunManifest | undefined): Record<string, string> {
    if (!manifest) return {}
    const out: Record<string, string> = {}
    const add = (label: string, value: unknown) => {
        if (value === null || value === undefined) return
        out[label] = typeof value === 'object' ? JSON.stringify(value) : String(value)
    }
    add('Generation model', manifest.generation_model)
    add('Judge model', manifest.eval_judge_model)
    add('Evaluator version', manifest.rag_evaluator_version)
    Object.entries(manifest.build_config_snapshot ?? {}).forEach(([k, v]) => add(`Build: ${k}`, v))
    Object.entries(manifest.query_overrides ?? {}).forEach(([k, v]) => add(`Override: ${k}`, v))
    Object.entries(manifest.effective_config_snapshot ?? manifest.rag_config_snapshot ?? {}).forEach(([k, v]) => add(`Effective: ${k}`, v))
    Object.entries(manifest.kb_version_snapshot ?? {}).forEach(([k, v]) => add(`KB: ${k}`, v))
    return out
}

export function ConfigDiff({ members }: ConfigDiffProps) {
    const queries = useQueries({
        queries: members.map((m) => ({
            queryKey: ['manifest', m.id],
            queryFn: () => api.evaluations.getManifest(m.id).then((r) => r.data),
            retry: false,
            staleTime: 5 * 60 * 1000,
        })),
    })

    if (queries.some((q) => q.isLoading)) {
        return (
            <div className="flex justify-center py-10">
                <Loader2 className="h-6 w-6 animate-spin text-primary/50" />
            </div>
        )
    }

    const flattened = queries.map((q) => flattenManifest(q.data))
    const allKeys = Array.from(new Set(flattened.flatMap((f) => Object.keys(f)))).sort()

    if (!allKeys.length) {
        return <p className="text-sm text-muted-foreground">No run configuration (manifest) is available for these evaluations.</p>
    }

    return (
        <div className="overflow-x-auto rounded-xl border border-border">
            <table className="w-full border-collapse text-sm">
                <thead>
                    <tr className="border-b border-border bg-muted/40">
                        <th className="sticky left-0 z-10 bg-muted/40 px-4 py-3 text-left text-[10px] font-bold uppercase tracking-widest text-muted-foreground">Field</th>
                        {members.map((m) => (
                            <th key={m.id} className="min-w-[160px] px-4 py-3 text-left font-bold">
                                <span className="block truncate max-w-[200px]">{m.label}</span>
                            </th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {allKeys.map((key) => {
                        const values = flattened.map((f) => f[key] ?? null)
                        const distinct = new Set(values.map((v) => v ?? '∅'))
                        const differs = distinct.size > 1
                        return (
                            <tr key={key} className={cn('border-b border-border/60 last:border-0', differs && 'bg-amber-500/5')}>
                                <td className="sticky left-0 z-10 bg-card px-4 py-2.5 text-[11px] font-bold uppercase tracking-wider text-muted-foreground whitespace-nowrap">
                                    {key}
                                    {differs && (
                                        <span className="ml-1.5 rounded bg-amber-500/20 px-1 py-0.5 text-[9px] font-bold text-amber-600">
                                            {key.startsWith('Build:') ? 'new index' : 'diff'}
                                        </span>
                                    )}
                                </td>
                                {values.map((v, i) => (
                                    <td key={members[i].id} className={cn('px-4 py-2.5 text-xs', differs ? 'font-semibold' : 'text-muted-foreground')}>
                                        <span className="block max-w-[240px] truncate" title={v ?? '—'}>{v ?? '—'}</span>
                                    </td>
                                ))}
                            </tr>
                        )
                    })}
                </tbody>
            </table>
        </div>
    )
}
