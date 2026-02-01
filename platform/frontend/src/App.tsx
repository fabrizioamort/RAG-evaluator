import { Routes, Route } from 'react-router-dom'
import { Layout } from './components/layout/Layout'
import { Dashboard } from './pages/Dashboard'
import { Projects } from './pages/Projects'
import { ProjectDetail } from './pages/ProjectDetail'
import { KBDetail } from './pages/KBDetail'
import { Indexes } from './pages/Indexes'
import { IndexDetail } from './pages/IndexDetail'
import { Playground } from './pages/Playground'
import { NotFound } from './pages/NotFound'

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="projects" element={<Projects />} />
        <Route path="projects/:id" element={<ProjectDetail />} />
        <Route path="knowledge-bases/:id" element={<KBDetail />} />
        <Route path="indexes" element={<Indexes />} />
        <Route path="indexes/:id" element={<IndexDetail />} />
        <Route path="playground" element={<Playground />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  )
}

export default App
