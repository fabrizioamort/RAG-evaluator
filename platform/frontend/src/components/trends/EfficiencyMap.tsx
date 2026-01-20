import React, { useEffect, useMemo, useState } from 'react'
import {
    CartesianGrid,
    Cell,
    ResponsiveContainer,
    Scatter,
    ScatterChart,
    Tooltip,
    XAxis,
    YAxis,
    ZAxis,
} from 'recharts'
import { Pause, Play } from 'lucide-react'
import { ProjectTrends } from '@/api/client'
import { cn } from '@/lib/utils'

interface EfficiencyMapProps {
    trends: ProjectTrends
}

interface EfficiencyPoint {
    evaluationId: string
    ragConfigName: string
    timestamp: string
    timestampMs: number
    formattedDate: string
    cost: number
    latency: number
    overall: number | null
    correctness: number | null
    overallSize: number
}

const DEFAULT_COLOR = '#94a3b8'
const FRONTIER_COLOR = '#0ea5e9'

const formatCost = (value: number) => `$${value.toFixed(4)}`
const formatLatency = (value: number) => `${value.toFixed(2)}s`
const formatScore = (value: number | null) => (value === null ? 'N/A' : value.toFixed(2))

const getCorrectnessColor = (value: number | null) => {
    if (value === null || Number.isNaN(value)) {
        return DEFAULT_COLOR
    }
    const clamped = Math.min(1, Math.max(0, value))
    const hue = Math.round(clamped * 120)
    return `hsl(${hue}, 70%, 45%)`
}

const computeParetoFrontier = (points: EfficiencyPoint[]) => {
    return points.filter((candidate) => {
        return !points.some((competitor) => {
            if (competitor.evaluationId === candidate.evaluationId) {
                return false
            }
            const candidateOverall = candidate.overall ?? 0
            const competitorOverall = competitor.overall ?? 0
            const dominates =
                competitor.cost <= candidate.cost &&
                competitor.latency <= candidate.latency &&
                competitorOverall >= candidateOverall &&
                (competitor.cost < candidate.cost ||
                    competitor.latency < candidate.latency ||
                    competitorOverall > candidateOverall)
            return dominates
        })
    })
}

export const EfficiencyMap: React.FC<EfficiencyMapProps> = ({ trends }) => {
    const dateFormatter = useMemo(
        () =>
            new Intl.DateTimeFormat('en-US', {
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
            }),
        []
    )

    const points = useMemo(() => {
        const items: EfficiencyPoint[] = []

        trends.trends.forEach((trend) => {
            const ragConfigName = trend.rag_config_name || 'Generic'
            trend.data_points.forEach((dp) => {
                const cost = dp.metrics.total_cost_usd
                const latency = dp.metrics.avg_latency_seconds
                if (typeof cost !== 'number' || typeof latency !== 'number') {
                    return
                }
                const overall =
                    typeof dp.metrics.overall_avg === 'number' ? dp.metrics.overall_avg : null
                const correctness =
                    typeof dp.metrics.g_eval_avg === 'number' ? dp.metrics.g_eval_avg : null
                const timestampMs = new Date(dp.timestamp).getTime()
                items.push({
                    evaluationId: dp.evaluation_id,
                    ragConfigName,
                    timestamp: dp.timestamp,
                    timestampMs,
                    formattedDate: dateFormatter.format(new Date(dp.timestamp)),
                    cost,
                    latency,
                    overall,
                    correctness,
                    overallSize: overall === null ? 0.08 : Math.max(0.08, overall),
                })
            })
        })

        return items.sort((a, b) => a.timestampMs - b.timestampMs)
    }, [dateFormatter, trends])

    const timelineStops = useMemo(() => {
        const unique = Array.from(new Set(points.map((point) => point.timestampMs)))
        return unique.sort((a, b) => a - b)
    }, [points])

    const [timeIndex, setTimeIndex] = useState(Math.max(0, timelineStops.length - 1))
    const [isPlaying, setIsPlaying] = useState(false)
    const [costScale, setCostScale] = useState<'linear' | 'log'>('linear')
    const [latencyScale, setLatencyScale] = useState<'linear' | 'log'>('log')

    useEffect(() => {
        setTimeIndex(Math.max(0, timelineStops.length - 1))
        setIsPlaying(false)
    }, [timelineStops.length])

    useEffect(() => {
        if (!isPlaying || timelineStops.length <= 1) {
            return
        }
        const interval = setInterval(() => {
            setTimeIndex((current) => {
                if (current >= timelineStops.length - 1) {
                    setIsPlaying(false)
                    return current
                }
                return current + 1
            })
        }, 1100)
        return () => clearInterval(interval)
    }, [isPlaying, timelineStops.length])

    const activeTimestamp = timelineStops[timeIndex]
    const filteredPoints = useMemo(() => {
        if (!activeTimestamp) {
            return points
        }
        return points.filter((point) => point.timestampMs <= activeTimestamp)
    }, [activeTimestamp, points])

    const scalePoints = useMemo(() => {
        return filteredPoints.filter((point) => {
            if (costScale === 'log' && point.cost <= 0) {
                return false
            }
            if (latencyScale === 'log' && point.latency <= 0) {
                return false
            }
            return true
        })
    }, [costScale, filteredPoints, latencyScale])

    const hiddenByScale = filteredPoints.length - scalePoints.length

    const costDomain = useMemo(() => {
        const values = scalePoints.map((point) => point.cost).filter((value) => value > 0)
        if (!values.length) {
            return [0.01, 1]
        }
        const min = Math.min(...values)
        const max = Math.max(...values)
        return costScale === 'log' ? [min, max] : [0, max]
    }, [costScale, scalePoints])

    const latencyDomain = useMemo(() => {
        const values = scalePoints.map((point) => point.latency).filter((value) => value > 0)
        if (!values.length) {
            return [0.01, 1]
        }
        const min = Math.min(...values)
        const max = Math.max(...values)
        return latencyScale === 'log' ? [min, max] : [0, max]
    }, [latencyScale, scalePoints])

    const formatCostAxis = (value: number) => {
        if (!Number.isFinite(value)) {
            return ''
        }
        if (value < 1) {
            return `${(value * 100).toFixed(2)}¢`
        }
        return `$${value.toFixed(2)}`
    }

    const paretoPoints = useMemo(() => computeParetoFrontier(scalePoints), [scalePoints])
    const paretoIds = useMemo(
        () => new Set(paretoPoints.map((point) => point.evaluationId)),
        [paretoPoints]
    )
    const paretoLine = useMemo(
        () =>
            [...paretoPoints]
                .sort((a, b) => a.cost - b.cost)
                .map((point) => ({ cost: point.cost, latency: point.latency })),
        [paretoPoints]
    )

    const maxOverall = useMemo(() => {
        return scalePoints.reduce((max, point) => {
            const value = point.overall ?? 0
            return value > max ? value : max
        }, 0)
    }, [scalePoints])

    if (!points.length) {
        return (
            <div className="rounded-xl border border-border bg-card p-6 h-64 flex items-center justify-center text-muted-foreground shadow-sm">
                No evaluation data available for the efficiency map.
            </div>
        )
    }

    const timeLabel = activeTimestamp ? dateFormatter.format(new Date(activeTimestamp)) : 'All time'

    return (
        <div className="space-y-6">
            <div className="rounded-xl border border-border bg-card p-4 sm:p-5">
                <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                    <div>
                        <p className="text-xs font-bold uppercase tracking-wider text-muted-foreground">
                            Time Scrubber
                        </p>
                        <p className="text-sm font-semibold">
                            Showing {scalePoints.length} of {points.length} evaluations - Up to{' '}
                            {timeLabel}
                        </p>
                    </div>
                    <div className="flex items-center gap-3">
                        <button
                            onClick={() => setIsPlaying((prev) => !prev)}
                            disabled={timelineStops.length <= 1}
                            className={cn(
                                "flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-bold uppercase tracking-wider transition-all",
                                isPlaying
                                    ? "border-primary/40 bg-primary/10 text-primary"
                                    : "border-border bg-muted text-muted-foreground hover:text-foreground"
                            )}
                        >
                            {isPlaying ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
                            {isPlaying ? 'Pause' : 'Play'}
                        </button>
                        <button
                            onClick={() => setTimeIndex(timelineStops.length - 1)}
                            className="rounded-full border border-border px-3 py-1.5 text-xs font-bold uppercase tracking-wider text-muted-foreground hover:text-foreground"
                        >
                            Show All
                        </button>
                    </div>
                </div>
                {timelineStops.length > 1 && (
                    <input
                        type="range"
                        min={0}
                        max={timelineStops.length - 1}
                        value={timeIndex}
                        onChange={(event) => {
                            setIsPlaying(false)
                            setTimeIndex(Number(event.target.value))
                        }}
                        className="mt-4 w-full accent-primary"
                    />
                )}
                <div className="mt-4 flex flex-wrap items-center gap-4">
                    <div className="flex items-center gap-2">
                        <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Cost Axis</span>
                        <button
                            onClick={() => setCostScale('linear')}
                            className={cn(
                                "rounded-full border px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider transition-all",
                                costScale === 'linear'
                                    ? "border-primary/40 bg-primary/10 text-primary"
                                    : "border-border bg-muted text-muted-foreground hover:text-foreground"
                            )}
                        >
                            Linear
                        </button>
                        <button
                            onClick={() => setCostScale('log')}
                            className={cn(
                                "rounded-full border px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider transition-all",
                                costScale === 'log'
                                    ? "border-primary/40 bg-primary/10 text-primary"
                                    : "border-border bg-muted text-muted-foreground hover:text-foreground"
                            )}
                        >
                            Log
                        </button>
                    </div>
                    <div className="flex items-center gap-2">
                        <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Latency Axis</span>
                        <button
                            onClick={() => setLatencyScale('linear')}
                            className={cn(
                                "rounded-full border px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider transition-all",
                                latencyScale === 'linear'
                                    ? "border-primary/40 bg-primary/10 text-primary"
                                    : "border-border bg-muted text-muted-foreground hover:text-foreground"
                            )}
                        >
                            Linear
                        </button>
                        <button
                            onClick={() => setLatencyScale('log')}
                            className={cn(
                                "rounded-full border px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider transition-all",
                                latencyScale === 'log'
                                    ? "border-primary/40 bg-primary/10 text-primary"
                                    : "border-border bg-muted text-muted-foreground hover:text-foreground"
                            )}
                        >
                            Log
                        </button>
                    </div>
                </div>
                {hiddenByScale > 0 && (
                    <p className="mt-2 text-xs text-muted-foreground">
                        {hiddenByScale} evaluation{hiddenByScale === 1 ? '' : 's'} hidden because log
                        scale requires positive values.
                    </p>
                )}
            </div>

            <div className="relative overflow-hidden rounded-xl border border-border bg-card shadow-sm">
                <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top,_#dbeafe_0,_transparent_60%)] opacity-70" />
                <div className="relative border-b border-border p-6">
                    <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                        <div>
                            <h3 className="text-xl font-black tracking-tight">Efficiency Map</h3>
                            <p className="text-sm text-muted-foreground">
                                Cost vs latency, sized by overall score, colored by correctness.
                            </p>
                        </div>
                        <div className="flex flex-wrap items-center gap-4 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                            <div className="flex items-center gap-2">
                                <span className="text-[10px]">Correctness</span>
                                <div className="h-2 w-24 rounded-full bg-gradient-to-r from-red-500 via-yellow-400 to-green-500" />
                                <span className="text-[10px]">High</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <span className="text-[10px]">Overall</span>
                                <div className="flex items-center gap-1">
                                    {[0.3, 0.6, Math.max(0.9, maxOverall)].map((value, index) => (
                                        <span
                                            key={`${value}-${index}`}
                                            className="inline-flex items-center justify-center rounded-full bg-muted/80 text-[9px] text-muted-foreground"
                                            style={{ width: 10 + index * 6, height: 10 + index * 6 }}
                                        >
                                            {index === 2 ? value.toFixed(2) : ''}
                                        </span>
                                    ))}
                                </div>
                            </div>
                            <div className="flex items-center gap-2">
                                <span className="text-[10px]">Pareto Frontier</span>
                                <span className="h-[2px] w-8 bg-sky-500/70" />
                            </div>
                        </div>
                    </div>
                </div>
                <div className="relative p-6">
                    <div className="h-[420px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart margin={{ top: 10, right: 25, bottom: 15, left: 0 }}>
                                <CartesianGrid strokeDasharray="4 4" vertical={false} stroke="#e2e8f0" />
                                <XAxis
                                    type="number"
                                    dataKey="cost"
                                    scale={costScale === 'log' ? 'log' : 'linear'}
                                    domain={costDomain}
                                    tick={{ fontSize: 11, fontWeight: 500 }}
                                    tickMargin={10}
                                    stroke="#94a3b8"
                                    tickFormatter={(value) => formatCostAxis(Number(value))}
                                    name="Total Cost"
                                    label={{
                                        value: 'Total Cost (USD)',
                                        position: 'insideBottom',
                                        offset: -5,
                                        fontSize: 11,
                                        fontWeight: 600,
                                    }}
                                />
                                <YAxis
                                    type="number"
                                    dataKey="latency"
                                    scale={latencyScale === 'log' ? 'log' : 'linear'}
                                    domain={latencyDomain}
                                    tick={{ fontSize: 11, fontWeight: 500 }}
                                    stroke="#94a3b8"
                                    tickFormatter={(value) => `${Number(value).toFixed(2)}s`}
                                    name="Avg Latency"
                                    label={{
                                        value: 'Avg Latency (s)',
                                        angle: -90,
                                        position: 'insideLeft',
                                        fontSize: 11,
                                        fontWeight: 600,
                                        offset: 0,
                                    }}
                                />
                                <ZAxis dataKey="overallSize" range={[80, 320]} />
                                <Tooltip
                                    cursor={{ strokeDasharray: '3 3' }}
                                    content={({ active, payload }) => {
                                        if (!active || !payload || !payload.length) {
                                            return null
                                        }
                                        const point = payload[0].payload as EfficiencyPoint
                                        return (
                                            <div className="rounded-lg border border-border bg-background/95 p-3 text-xs shadow-lg">
                                                <div className="flex items-center justify-between gap-3">
                                                    <span className="font-bold">{point.ragConfigName}</span>
                                                    {paretoIds.has(point.evaluationId) && (
                                                        <span className="rounded-full bg-sky-500/10 px-2 py-0.5 text-[10px] font-bold uppercase text-sky-600">
                                                            Frontier
                                                        </span>
                                                    )}
                                                </div>
                                                <p className="text-[10px] uppercase tracking-wider text-muted-foreground">
                                                    Eval #{point.evaluationId.slice(0, 8)} - {point.formattedDate}
                                                </p>
                                                <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1">
                                                    <span className="text-muted-foreground">Cost</span>
                                                    <span className="font-semibold">{formatCost(point.cost)}</span>
                                                    <span className="text-muted-foreground">Latency</span>
                                                    <span className="font-semibold">{formatLatency(point.latency)}</span>
                                                    <span className="text-muted-foreground">Overall</span>
                                                    <span className="font-semibold">{formatScore(point.overall)}</span>
                                                    <span className="text-muted-foreground">Correctness</span>
                                                    <span className="font-semibold">{formatScore(point.correctness)}</span>
                                                </div>
                                            </div>
                                        )
                                    }}
                                />
                                {paretoLine.length > 1 && (
                                    <Scatter
                                        data={paretoLine}
                                        line={{
                                            stroke: FRONTIER_COLOR,
                                            strokeWidth: 2,
                                            strokeDasharray: '6 4',
                                        }}
                                        fill="none"
                                        shape="circle"
                                    />
                                )}
                                <Scatter data={scalePoints} shape="circle">
                                    {scalePoints.map((point) => (
                                        <Cell
                                            key={point.evaluationId}
                                            fill={getCorrectnessColor(point.correctness)}
                                            stroke={paretoIds.has(point.evaluationId) ? FRONTIER_COLOR : 'transparent'}
                                            strokeWidth={paretoIds.has(point.evaluationId) ? 2 : 1}
                                            opacity={paretoIds.has(point.evaluationId) ? 1 : 0.85}
                                        />
                                    ))}
                                </Scatter>
                            </ScatterChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>
        </div>
    )
}
