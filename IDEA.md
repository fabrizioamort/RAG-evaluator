## IDEA

Idea for a project to compare different methodologies and technologies for the development of a RAG retrieval augmented generation system for AI agents.

## SCOPE

The scope of this project is to develop different kinds of RAG systems to evaluate and compare them. The project should also be a public github project to showcase the expertise and ability of his ideator.

## TECHNOLOGIES

The project should implement at least this type of RAG:

1) Vector db - only semantic search (with ChromaDB)
2) Vector db - hybrid search (what is the best open source tool for hybrid search ? )
3) Graph RAG (Neo4J and Graphite or GraphRAG ? )
4) Filesystem RAG (Giving LLM access to filesystem to search for relevant filesas successfully done by Claude code)

## EVALUATION

The evaluation of each of the four different technologies should focus on these keys indicators:

Accuracy:
Faithfulness: Is the answer derived only from the context?
Answer Relevance: Does it actually answer the user's question?
Context Precision: Did the retriever find the right documents?
Speed
Cost

For the evaluation use tools like DeepEval ( or as alternative Ragas )

## WORKFLOW

The project should start from a set of documents in different formats (pdf, docx, ecc.) and implement these steps:

Data preparation: Analyze the set of documents and prepare them for the next phases. This preparation must be done for each technology and depends by the technology.
Test definition: Create a set of question and answer that will be used to evaluate the different RAG systems
Evaluation execution: Evaluate the accuracy, speed and cost of each RAG implementation
Reporting: Generate a final report to compare the performance of the RAG implementation
