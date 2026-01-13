import { useNavigate } from 'react-router-dom'
import {
    FolderOpen,
    Database,
    FileText,
    Settings2,
    FlaskConical,
    MoreVertical,
    Calendar,
    Tag
} from 'lucide-react'
import { Project } from '@/api/client'
import { cn } from '@/lib/utils'

interface ProjectCardProps {
    project: Project
}

export function ProjectCard({ project }: ProjectCardProps) {
    const navigate = useNavigate()

    return (
        <div
            onClick={() => navigate(`/projects/${project.id}`)}
            className="group relative flex flex-col rounded-xl border border-border bg-card p-5 transition-all hover:border-primary/50 hover:shadow-lg cursor-pointer"
        >
            {/* Header */}
            <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                    <div className="rounded-lg bg-primary/10 p-2 text-primary group-hover:bg-primary group-hover:text-primary-foreground transition-colors">
                        <FolderOpen className="h-5 w-5" />
                    </div>
                    <div>
                        <h3 className="font-semibold text-lg leading-none">{project.name}</h3>
                        <div className="mt-1.5 flex items-center gap-2 text-xs text-muted-foreground whitespace-nowrap">
                            <Calendar className="h-3 w-3" />
                            {new Date(project.created_at).toLocaleDateString()}
                            <span className="inline-flex h-1 w-1 rounded-full bg-muted-foreground/30" />
                            <div className={cn(
                                "px-1.5 py-0.5 rounded-full text-[10px] font-medium uppercase tracking-wider",
                                project.status === 'active' ? "bg-green-500/10 text-green-600" : "bg-muted text-muted-foreground"
                            )}>
                                {project.status}
                            </div>
                        </div>
                    </div>
                </div>
                <button
                    onClick={(e) => {
                        e.stopPropagation()
                        // Dropdown menu logic would go here
                    }}
                    className="rounded-md p-1 hover:bg-muted text-muted-foreground transition-colors"
                >
                    <MoreVertical className="h-4 w-4" />
                </button>
            </div>

            {/* Description */}
            <p className="mt-4 text-sm text-muted-foreground line-clamp-2 min-h-[40px]">
                {project.description || 'No description provided.'}
            </p>

            {/* Tags */}
            <div className="mt-4 flex flex-wrap gap-1.5">
                {project.tags.slice(0, 3).map((tag) => (
                    <div
                        key={tag}
                        className="flex items-center gap-1 rounded-md bg-accent px-2 py-0.5 text-[10px] font-medium text-accent-foreground"
                    >
                        <Tag className="h-2.5 w-2.5" />
                        {tag}
                    </div>
                ))}
                {project.tags.length > 3 && (
                    <span className="text-[10px] text-muted-foreground font-medium pl-1">
                        +{project.tags.length - 3} more
                    </span>
                )}
            </div>

            {/* Stats */}
            <div className="mt-6 pt-4 border-t border-border grid grid-cols-4 gap-2">
                <StatItem icon={<Database />} value={project.knowledge_base_count} label="KB" />
                <StatItem icon={<FileText />} value={project.test_set_count} label="Tests" />
                <StatItem icon={<Settings2 />} value={project.rag_config_count} label="RAGs" />
                <StatItem icon={<FlaskConical />} value={project.evaluation_count} label="Evals" />
            </div>
        </div>
    )
}

function StatItem({ icon, value, label }: { icon: React.ReactNode, value: number, label: string }) {
    return (
        <div className="flex flex-col items-center gap-1">
            <div className="text-muted-foreground/60 child-svg:h-3.5 child-svg:w-3.5">
                {icon}
            </div>
            <span className="text-xs font-bold leading-none">{value}</span>
            <span className="text-[9px] uppercase tracking-tighter text-muted-foreground font-medium">{label}</span>
        </div>
    )
}
