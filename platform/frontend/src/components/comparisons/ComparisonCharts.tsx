import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
    RadarChart,
    Radar,
    PolarGrid,
    PolarAngleAxis,
    PolarRadiusAxis,
    ScatterChart,
    Scatter,
    ZAxis,
} from 'recharts'
import { ComparisonMember, METRIC_ROWS } from './compare-utils'

const COLORS = ['#2563eb', '#16a34a', '#d97706', '#dc2626', '#7c3aed', '#0891b2', '#db2777', '#65a30d', '#ea580c', '#4f46e5']

interface ComparisonChartsProps {
    members: ComparisonMember[]
}

export function ComparisonCharts({ members }: ComparisonChartsProps) {
    const scoreRows = METRIC_ROWS.filter((r) => r.isScore)

    const barData = scoreRows.map((row) => {
        const entry: Record<string, string | number | null> = { metric: row.label }
        members.forEach((m) => {
            entry[m.id] = row.get(m)
        })
        return entry
    })

    const radarData = barData // same shape works for radar

    const overallRow = METRIC_ROWS.find((r) => r.key === 'overall')!
    const costRow = METRIC_ROWS.find((r) => r.key === 'avg_cost')!
    const scatterData = members
        .map((m, i) => ({
            name: m.label,
            cost: costRow.get(m),
            overall: overallRow.get(m),
            fill: COLORS[i % COLORS.length],
        }))
        .filter((d) => d.cost !== null && d.overall !== null)

    return (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <ChartCard title="Quality metrics">
                <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={barData} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                        <XAxis dataKey="metric" tick={{ fontSize: 11 }} />
                        <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                        <Tooltip
                            contentStyle={{ fontSize: 12, borderRadius: 8 }}
                            formatter={(v: number) => (v === null ? '—' : v.toFixed(3))}
                            labelFormatter={() => ''}
                        />
                        <Legend wrapperStyle={{ fontSize: 11 }} />
                        {members.map((m, i) => (
                            <Bar key={m.id} dataKey={m.id} name={m.label} fill={COLORS[i % COLORS.length]} radius={[3, 3, 0, 0]} />
                        ))}
                    </BarChart>
                </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Metric shape (radar)">
                <ResponsiveContainer width="100%" height={300}>
                    <RadarChart data={radarData} outerRadius="70%">
                        <PolarGrid className="stroke-border" />
                        <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11 }} />
                        <PolarRadiusAxis domain={[0, 1]} tick={{ fontSize: 10 }} />
                        <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} formatter={(v: number) => (v === null ? '—' : v.toFixed(3))} />
                        <Legend wrapperStyle={{ fontSize: 11 }} />
                        {members.map((m, i) => (
                            <Radar key={m.id} dataKey={m.id} name={m.label} stroke={COLORS[i % COLORS.length]} fill={COLORS[i % COLORS.length]} fillOpacity={0.15} />
                        ))}
                    </RadarChart>
                </ResponsiveContainer>
            </ChartCard>

            {scatterData.length > 0 && (
                <ChartCard title="Cost vs. quality" subtitle="Lower cost and higher overall is better (top-left)">
                    <ResponsiveContainer width="100%" height={300}>
                        <ScatterChart margin={{ top: 8, right: 16, left: -16, bottom: 8 }}>
                            <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                            <XAxis type="number" dataKey="cost" name="Cost / query" tick={{ fontSize: 11 }} tickFormatter={(v) => `$${Number(v).toFixed(4)}`} />
                            <YAxis type="number" dataKey="overall" name="Overall" domain={[0, 1]} tick={{ fontSize: 11 }} />
                            <ZAxis range={[120, 120]} />
                            <Tooltip
                                cursor={{ strokeDasharray: '3 3' }}
                                contentStyle={{ fontSize: 12, borderRadius: 8 }}
                                formatter={(v: number, name: string) => (name === 'Cost / query' ? `$${Number(v).toFixed(4)}` : Number(v).toFixed(3))}
                            />
                            <Scatter data={scatterData} />
                        </ScatterChart>
                    </ResponsiveContainer>
                </ChartCard>
            )}
        </div>
    )
}

function ChartCard({ title, subtitle, children }: { title: string; subtitle?: string; children: React.ReactNode }) {
    return (
        <div className="rounded-xl border border-border bg-card p-4">
            <div className="mb-3">
                <h4 className="text-sm font-bold">{title}</h4>
                {subtitle && <p className="text-xs text-muted-foreground">{subtitle}</p>}
            </div>
            {children}
        </div>
    )
}
