import { useNavigate, useParams } from 'react-router-dom'
import { ComparisonDetail } from '@/components/comparisons/ComparisonDetail'

export function ComparisonRouteDetail() {
  const { projectId, comparisonId } = useParams<{ projectId: string; comparisonId: string }>()
  const navigate = useNavigate()

  if (!projectId || !comparisonId) {
    return null
  }

  return (
    <ComparisonDetail
      comparisonId={comparisonId}
      onBack={() => navigate(`/projects/${projectId}?tab=compare`)}
    />
  )
}
