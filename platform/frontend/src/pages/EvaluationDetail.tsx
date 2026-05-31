import { useNavigate, useParams } from 'react-router-dom'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { AlertCircle, Loader2 } from 'lucide-react'
import { api } from '@/api/client'
import { EvaluationProgress } from '@/components/evaluations/EvaluationProgress'
import { EvaluationResults } from '@/components/evaluations/EvaluationResults'

export function EvaluationDetail() {
  const { projectId, evaluationId } = useParams<{ projectId: string; evaluationId: string }>()
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const { data, isLoading, isError } = useQuery({
    queryKey: ['evaluation', evaluationId],
    queryFn: () => api.evaluations.get(evaluationId!),
    enabled: !!evaluationId,
  })

  const goBack = () => {
    if (projectId) {
      navigate(`/projects/${projectId}?tab=evals`)
    } else {
      navigate('/projects')
    }
  }

  if (isLoading) {
    return (
      <div className="flex h-[60vh] items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary/50" />
      </div>
    )
  }

  if (isError || !data?.data || !evaluationId) {
    return (
      <div className="flex h-[60vh] flex-col items-center justify-center gap-3">
        <AlertCircle className="h-10 w-10 text-destructive" />
        <p className="font-medium">Evaluation not found</p>
        <button onClick={goBack} className="text-sm text-primary hover:underline">
          Back to evaluations
        </button>
      </div>
    )
  }

  if (data.data.status === 'completed') {
    return (
      <EvaluationResults
        evaluationId={evaluationId}
        onBack={() => {
          queryClient.invalidateQueries({ queryKey: ['evaluations', projectId] })
          goBack()
        }}
      />
    )
  }

  return (
    <EvaluationProgress
      evaluationId={evaluationId}
      onClose={() => {
        queryClient.invalidateQueries({ queryKey: ['evaluations', projectId] })
        goBack()
      }}
    />
  )
}
