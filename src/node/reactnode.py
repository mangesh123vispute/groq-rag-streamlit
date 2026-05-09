"""LangGraph nodes for RAG workflow.

Answers are generated only from retrieved chunks (uploaded / indexed documents).
No Wikipedia or other external knowledge is injected into the prompt.
"""

from src.state.rag_state import RAGState


class RAGNodes:
    """Contains node functions for RAG workflow"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def retrieve_docs(self, state: RAGState) -> RAGState:
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs,
        )

    def generate_answer(self, state: RAGState) -> RAGState:
        context = "\n\n".join(doc.page_content for doc in state.retrieved_docs)

        prompt = f"""Answer the question using ONLY the context below (from the user's indexed documents).
If the context does not contain enough information, say so clearly—do not guess and do not use outside knowledge.

Context:
{context or "(no passages were retrieved for this question.)"}

Question: {state.question}"""

        response = self.llm.invoke(prompt)
        answer = getattr(response, "content", None) or str(response)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer,
        )
