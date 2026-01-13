import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Plus, Loader2, RefreshCcw } from 'lucide-react'
import { api, ProjectCreate } from '@/api/client'
import { ProjectList } from '@/components/projects/ProjectList'
import { CreateProjectDialog } from '@/components/projects/CreateProjectDialog'

export function Projects() {
  const [isDialogOpen, setIsDialogOpen] = useState(false)
  const queryClient = useQueryClient()

  const { data, isLoading, isError, refetch, isRefetching } = useQuery({
    queryKey: ['projects'],
    queryFn: () => api.projects.list({ limit: 100 }),
  })

  const createMutation = useMutation({
    mutationFn: (newProject: ProjectCreate) => api.projects.create(newProject),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['projects'] })
    },
  })

  const handleCreateProject = async (projectData: ProjectCreate) => {
    await createMutation.mutateAsync(projectData)
  }

  if (isLoading) {
    return (
      <div className="flex h-[60vh] items-center justify-center">
        <Loader2 className="h-10 w-10 animate-spin text-primary" />
      </div>
    )
  }

  if (isError) {
    return (
      <div className="flex h-[60vh] flex-col items-center justify-center space-y-4">
        <p className="text-destructive font-medium">Failed to load projects</p>
        <button
          onClick={() => refetch()}
          className="flex items-center gap-2 rounded-md bg-secondary px-4 py-2 text-sm hover:bg-secondary/80"
        >
          <RefreshCcw className="h-4 w-4" />
          Try Again
        </button>
      </div>
    )
  }

  const projects = data?.data?.items || []

  return (
    <div className="space-y-8 pb-12">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Projects</h1>
          <p className="mt-1.5 text-muted-foreground max-w-2xl">
            Design and organize your RAG evaluation experiments. Each project can contain multiple
            knowledge bases, test sets, and configurations.
          </p>
        </div>
        <button
          onClick={() => setIsDialogOpen(true)}
          className="flex items-center justify-center gap-2 rounded-lg bg-primary px-5 py-2.5 text-sm font-bold text-primary-foreground hover:bg-primary/90 transition-all shadow-md active:scale-95 group"
        >
          <Plus className="h-4 w-4 transition-transform group-hover:rotate-90" />
          <span>New Project</span>
        </button>
      </div>

      <div className="relative">
        {isRefetching && (
          <div className="absolute -top-6 right-0 flex items-center gap-2 text-[10px] text-muted-foreground animate-pulse">
            <RefreshCcw className="h-2.5 w-2.5 animate-spin" />
            Updating...
          </div>
        )}
        <ProjectList
          projects={projects}
          onCreateClick={() => setIsDialogOpen(true)}
        />
      </div>

      <CreateProjectDialog
        isOpen={isDialogOpen}
        onClose={() => setIsDialogOpen(false)}
        onSubmit={handleCreateProject}
      />
    </div>
  )
}
