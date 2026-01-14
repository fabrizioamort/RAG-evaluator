import React, { useState } from 'react';
import { ChevronDown, AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';

interface MetricExplainabilityProps {
    label: string;
    score: number | null | undefined;
    reason: string | null | undefined;
}

export function MetricExplainability({ label, score, reason }: MetricExplainabilityProps) {
    const [isExpanded, setIsExpanded] = useState(false);

    if (score === null || score === undefined) return null;

    const getScoreColor = (s: number) => {
        if (s >= 0.7) return 'bg-green-500';
        if (s >= 0.4) return 'bg-yellow-500';
        return 'bg-red-500';
    };

    const getScoreTextClass = (s: number) => {
        if (s >= 0.7) return 'text-green-600';
        if (s >= 0.4) return 'text-yellow-600';
        return 'text-red-600';
    };

    const getScoreBgClass = (s: number) => {
        if (s >= 0.7) return 'bg-green-500/5';
        if (s >= 0.4) return 'bg-yellow-500/5';
        return 'bg-red-500/5';
    };

    const barColor = getScoreColor(score);
    const textColor = getScoreTextClass(score);
    const bgColor = getScoreBgClass(score);

    return (
        <div className={cn(
            "rounded-lg border border-border overflow-hidden transition-all duration-200",
            bgColor
        )}>
            <div
                className="p-3 cursor-pointer hover:bg-accent/30 transition-colors"
                onClick={() => setIsExpanded(!isExpanded)}
            >
                <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                        <span className="text-sm font-semibold text-foreground">{label}</span>
                    </div>
                    <div className="flex items-center gap-3">
                        <span className={cn("text-sm font-bold", textColor)}>
                            {score.toFixed(2)}
                        </span>
                        <ChevronDown className={cn(
                            "h-4 w-4 text-muted-foreground transition-transform duration-200",
                            isExpanded && "rotate-180"
                        )} />
                    </div>
                </div>

                {/* Progress Bar */}
                <div className="h-1.5 w-full bg-secondary/50 rounded-full overflow-hidden">
                    <div
                        className={cn("h-full transition-all duration-500", barColor)}
                        style={{ width: `${score * 100}%` }}
                    />
                </div>
            </div>

            {isExpanded && (
                <div className="px-3 pb-3 pt-1 animate-in slide-in-from-top-2 duration-200">
                    <div className="text-xs text-muted-foreground bg-background/50 rounded-md p-2 border border-border/50 leading-relaxed italic">
                        {reason ? (
                            <div className="whitespace-pre-wrap">{reason}</div>
                        ) : (
                            <div className="flex items-center gap-1.5">
                                <AlertCircle className="h-3 w-3" />
                                No reasoning provided for this score.
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
