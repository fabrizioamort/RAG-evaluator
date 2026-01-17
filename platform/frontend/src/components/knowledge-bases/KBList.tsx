import { KnowledgeBase } from '@/api/client'
import { KBCard } from './KBCard'
import { Database, Plus } from 'lucide-react'

interface KBListProps {
    knowledgeBases: KnowledgeBase[]
    onCreateClick: () => void
}

export function KBList({ knowledgeBases, onCreateClick }: KBListProps) {
    if (knowledgeBases.length === 0) {
        return (
            <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-border py-20 bg-card/50">
                <div className="rounded-full bg-primary/10 p-5 text-primary">
                    <Database className="h-10 w-10" />
                </div>
                <h3 className="mt-5 text-xl font-semibold">No knowledge bases yet</h3>
                <p className="mt-2 text-center text-muted-foreground max-w-sm">
                    A knowledge base contains the documents that your RAG system will use for retrieval.
                </p>
                <button
                    onClick={onCreateClick}
                    className="mt-6 flex items-center gap-2 rounded-lg bg-primary px-6 py-2.5 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-all shadow-md"
                >
                    <Plus className="h-4 w-4" />
                    Create Knowledge Base
                </button>
            </div>
        )
    }

    return (
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
            {knowledgeBases.map((kb) => (
                <KBCard key={kb.id} kb={kb} />
            ))}
        </div>
    )
}
