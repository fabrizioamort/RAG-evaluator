import React, { useMemo } from 'react'
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Cell
} from 'recharts'
import { EvaluationResult } from '../../api/client'

interface DifficultyChartProps {
    results: EvaluationResult[]
}

const COLORS: Record<string, string> = {
    easy: '#16a34a', // green-600
    medium: '#d97706', // amber-600
    hard: '#dc2626', // red-600
}

export const DifficultyChart: React.FC<DifficultyChartProps> = ({ results }) => {
    const chartData = useMemo(() => {
        const difficulties = ['easy', 'medium', 'hard']
        const scores = difficulties.map(diff => {
            const items = results.filter(r => r.difficulty === diff)
            if (items.length === 0) return null

            const avg = items.reduce((acc, curr) => {
                const sum = (curr.faithfulness_score || 0) +
                    (curr.relevancy_score || 0) +
                    (curr.precision_score || 0) +
                    (curr.recall_score || 0)
                const count = [
                    curr.faithfulness_score,
                    curr.relevancy_score,
                    curr.precision_score,
                    curr.recall_score
                ].filter(v => v !== null && v !== undefined).length

                return acc + (count > 0 ? sum / count : 0)
            }, 0) / items.length

            return {
                name: diff.charAt(0).toUpperCase() + diff.slice(1),
                score: parseFloat(avg.toFixed(2)),
                count: items.length,
                originalDiff: diff
            }
        }).filter(v => v !== null)

        return scores
    }, [results])

    if (chartData.length === 0) {
        return (
            <div className="flex h-40 items-center justify-center text-sm text-muted-foreground border border-dashed rounded-lg">
                No difficulty data available
            </div>
        )
    }

    return (
        <div className="h-[200px] w-full mt-4">
            <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, left: 10, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="#e5e7eb" />
                    <XAxis type="number" domain={[0, 1]} hide />
                    <YAxis
                        dataKey="name"
                        type="category"
                        tick={{ fontSize: 12, fontWeight: 600 }}
                        width={70}
                        axisLine={false}
                        tickLine={false}
                    />
                    <Tooltip
                        cursor={{ fill: 'transparent' }}
                        contentStyle={{
                            backgroundColor: 'white',
                            borderRadius: '8px',
                            border: '1px solid #e5e7eb',
                            boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
                        }}
                        formatter={(value: number) => [`${value}`, 'Avg Score']}
                        labelFormatter={(label, payload) => {
                            if (payload && payload[0]) {
                                return `${label} (${payload[0].payload.count} cases)`
                            }
                            return label
                        }}
                    />
                    <Bar dataKey="score" radius={[0, 4, 4, 0]} barSize={20}>
                        {chartData.map((entry, index: number) => (
                            <Cell key={`cell-${index}`} fill={COLORS[entry.originalDiff]} />
                        ))}
                    </Bar>
                </BarChart>
            </ResponsiveContainer>
        </div>
    )
}
