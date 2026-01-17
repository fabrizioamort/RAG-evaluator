import { Project } from '@/api/client'
import { ProjectCard } from './ProjectCard'
import { FolderPlus } from 'lucide-react'

interface ProjectListProps {
    projects: Project[]
    onCreateClick: () => void
}

export function ProjectList({ projects, onCreateClick }: ProjectListProps) {
    if (projects.length === 0) {
        return (
            <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-border py-20 bg-card/50">
                <div className="rounded-full bg-primary/10 p-5 text-primary">
                    <FolderPlus className="h-10 w-10" />
                </div>
                <h3 className="mt-5 text-xl font-semibold">No projects found</h3>
                <p className="mt-2 text-center text-muted-foreground max-w-sm">
                    You haven't created any RAG evaluation projects yet. Create your first project to get started.
                </p>
                <button
                    onClick={onCreateClick}
                    className="mt-6 flex items-center gap-2 rounded-lg bg-primary px-6 py-2.5 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-all shadow-md active:scale-95"
                >
                    <FolderPlus className="h-4 w-4" />
                    Create First Project
                </button>
            </div>
        )
    }

    return (
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {projects.map((project) => (
                <ProjectCard key={project.id} project={project} />
            ))}
        </div>
    )
}
