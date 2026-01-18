import React, { useMemo } from 'react'
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
} from 'recharts'
import { ProjectTrends } from '@/api/client'

interface TrendChartProps {
    trends: ProjectTrends
}

const COLORS = [
    '#2563eb', // blue-600
    '#16a34a', // green-600
    '#dc2626', // red-600
    '#d97706', // amber-600
    '#7c3aed', // violet-600
    '#db2777', // pink-600
]

const METRICS = [
    { key: 'faithfulness_avg', label: 'Faithfulness' },
    { key: 'relevancy_avg', label: 'Relevancy' },
    { key: 'precision_avg', label: 'Precision' },
    { key: 'recall_avg', label: 'Recall' },
    { key: 'g_eval_avg', label: 'Correctness' },
    { key: 'overall_avg', label: 'Overall' },
]

export const TrendChart: React.FC<TrendChartProps> = ({ trends }) => {
    const [selectedMetric, setSelectedMetric] = React.useState('overall_avg')

    // Prepare data for the chart
    const chartData = useMemo(() => {
        // Collect all unique timestamps and sort them
        const allTimestamps = new Set<string>()
        trends.trends.forEach((trend) => {
            trend.data_points.forEach((dp) => {
                allTimestamps.add(dp.timestamp)
            })
        })

        const sortedTimestamps = Array.from(allTimestamps).sort()

        return sortedTimestamps.map((timestamp) => {
            const date = new Date(timestamp)
            const formattedDate = new Intl.DateTimeFormat('en-US', {
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
            }).format(date)

            const dataPoint: Record<string, string | number> = {
                timestamp,
                formattedDate,
            }

            trends.trends.forEach((trend) => {
                const dp = trend.data_points.find((p) => p.timestamp === timestamp)
                if (dp) {
                    const configName = trend.rag_config_name || 'Generic'
                    dataPoint[configName] = dp.metrics[selectedMetric] || 0
                }
            })

            return dataPoint
        })
    }, [trends, selectedMetric])

    const ragConfigs = useMemo(() => {
        return trends.trends.map((t) => t.rag_config_name || 'Generic')
    }, [trends])

    if (!trends.trends.length || !trends.trends.some(t => t.data_points.length > 0)) {
        return (
            <div className="rounded-xl border border-border bg-card p-6 h-64 flex items-center justify-center text-muted-foreground shadow-sm">
                No trend data available for this project.
            </div>
        )
    }

    return (
        <div className="space-y-6">
            <div className="flex flex-wrap gap-2">
                {METRICS.map((metric) => (
                    <button
                        key={metric.key}
                        onClick={() => setSelectedMetric(metric.key)}
                        className={`px-4 py-1.5 rounded-full text-xs font-bold uppercase tracking-wider transition-all shadow-sm ${selectedMetric === metric.key
                            ? 'bg-primary text-primary-foreground'
                            : 'bg-muted text-muted-foreground hover:bg-secondary hover:text-foreground'
                            }`}
                    >
                        {metric.label}
                    </button>
                ))}
            </div>

            <div className="rounded-xl border border-border bg-card overflow-hidden shadow-sm">
                <div className="p-6 border-b border-border">
                    <h3 className="text-lg font-bold">{METRICS.find((m) => m.key === selectedMetric)?.label} Trend</h3>
                    <p className="text-sm text-muted-foreground mt-1">
                        Performance of different RAG configurations over time
                    </p>
                </div>
                <div className="p-6">
                    <div className="h-[400px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={chartData} margin={{ top: 5, right: 30, left: 10, bottom: 5 }}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e7eb" />
                                <XAxis
                                    dataKey="formattedDate"
                                    tick={{ fontSize: 11, fontWeight: 500 }}
                                    tickMargin={10}
                                    stroke="#9ca3af"
                                />
                                <YAxis
                                    domain={[0, 1]}
                                    tick={{ fontSize: 11, fontWeight: 500 }}
                                    stroke="#9ca3af"
                                    label={{ value: 'Score', angle: -90, position: 'insideLeft', fontSize: 11, fontWeight: 600, offset: -5 }}
                                />
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: 'white',
                                        borderRadius: '12px',
                                        border: '1px solid #e5e7eb',
                                        boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)',
                                        fontSize: '12px',
                                        fontWeight: '500'
                                    }}
                                />
                                <Legend iconType="circle" wrapperStyle={{ paddingTop: '24px', fontSize: '12px', fontWeight: '600' }} />
                                {ragConfigs.map((configName, index) => (
                                    <Line
                                        key={configName}
                                        type="monotone"
                                        dataKey={configName}
                                        stroke={COLORS[index % COLORS.length]}
                                        strokeWidth={3}
                                        dot={{ r: 4, strokeWidth: 2, fill: 'white' }}
                                        activeDot={{ r: 6, strokeWidth: 0 }}
                                        connectNulls
                                    />
                                ))}
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>
        </div>
    )
}
