import { lazy, Suspense } from 'react'
import { Routes, Route } from 'react-router-dom'
import { Loader2 } from 'lucide-react'
import { Layout } from './components/layout/Layout'

const Dashboard = lazy(() => import('./pages/Dashboard').then((module) => ({ default: module.Dashboard })))
const Projects = lazy(() => import('./pages/Projects').then((module) => ({ default: module.Projects })))
const ProjectDetail = lazy(() => import('./pages/ProjectDetail').then((module) => ({ default: module.ProjectDetail })))
const KBDetail = lazy(() => import('./pages/KBDetail').then((module) => ({ default: module.KBDetail })))
const Indexes = lazy(() => import('./pages/Indexes').then((module) => ({ default: module.Indexes })))
const IndexDetail = lazy(() => import('./pages/IndexDetail').then((module) => ({ default: module.IndexDetail })))
const Playground = lazy(() => import('./pages/Playground').then((module) => ({ default: module.Playground })))
const NotFound = lazy(() => import('./pages/NotFound').then((module) => ({ default: module.NotFound })))
const EvaluationDetail = lazy(() => import('./pages/EvaluationDetail').then((module) => ({ default: module.EvaluationDetail })))
const TestSetRouteDetail = lazy(() => import('./pages/TestSetRouteDetail').then((module) => ({ default: module.TestSetRouteDetail })))
const ComparisonRouteDetail = lazy(() => import('./pages/ComparisonRouteDetail').then((module) => ({ default: module.ComparisonRouteDetail })))

function RouteFallback() {
  return (
    <div className="flex h-[60vh] items-center justify-center">
      <Loader2 className="h-8 w-8 animate-spin text-primary/50" />
    </div>
  )
}

function App() {
  return (
    <Suspense fallback={<RouteFallback />}>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="projects" element={<Projects />} />
          <Route path="projects/:id" element={<ProjectDetail />} />
          <Route path="projects/:projectId/test-sets/:testSetId" element={<TestSetRouteDetail />} />
          <Route path="projects/:projectId/evaluations/:evaluationId" element={<EvaluationDetail />} />
          <Route path="projects/:projectId/comparisons/:comparisonId" element={<ComparisonRouteDetail />} />
          <Route path="knowledge-bases/:id" element={<KBDetail />} />
          <Route path="indexes" element={<Indexes />} />
          <Route path="indexes/:id" element={<IndexDetail />} />
          <Route path="playground" element={<Playground />} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </Suspense>
  )
}

export default App
