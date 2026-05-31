import { useNavigate, useParams } from 'react-router-dom'
import { TestSetDetail } from '@/components/test-sets/TestSetDetail'

export function TestSetRouteDetail() {
  const { projectId, testSetId } = useParams<{ projectId: string; testSetId: string }>()
  const navigate = useNavigate()

  if (!projectId || !testSetId) {
    return null
  }

  return (
    <TestSetDetail
      projectId={projectId}
      testSetId={testSetId}
      onBack={() => navigate(`/projects/${projectId}?tab=tests`)}
    />
  )
}
