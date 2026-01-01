"""Streamlit web interface for RAG Evaluator."""

import streamlit as st


def main() -> None:
    """Main Streamlit application."""
    st.set_page_config(
        page_title="RAG Evaluator",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 RAG Evaluator")
    st.markdown(
        """
        Compare and evaluate different RAG (Retrieval Augmented Generation) implementations.
        """
    )

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        rag_type = st.selectbox(
            "Select RAG Implementation",
            [
                "ChromaDB Semantic Search",
                "Hybrid Search",
                "Neo4j Graph RAG",
                "Filesystem RAG",
            ],
        )

        st.divider()

        st.header("Evaluation Settings")
        top_k = st.slider("Top K Results", min_value=1, max_value=10, value=5)

    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["Query", "Evaluate", "Compare"])

    with tab1:
        st.header("Query RAG System")
        question = st.text_area(
            "Enter your question:",
            placeholder="What is the main topic of the documents?",
        )

        if st.button("Submit Query", type="primary"):
            if question:
                with st.spinner("Processing query..."):
                    # TODO: Implement actual query processing
                    st.info("Implementation coming soon!")
            else:
                st.warning("Please enter a question first.")

    with tab2:
        st.header("Run Evaluation")
        st.markdown("Evaluate RAG implementations against test cases.")

        col1, col2 = st.columns(2)
        with col1:
            test_set_file = st.file_uploader(
                "Upload Test Set (JSON)", type=["json"]
            )
        with col2:
            st.info(
                """
                Test set should contain:
                - question
                - expected_answer
                - (optional) context
                """
            )

        if st.button("Run Evaluation"):
            if test_set_file:
                with st.spinner("Running evaluation..."):
                    # TODO: Implement evaluation
                    st.info("Implementation coming soon!")
            else:
                st.warning("Please upload a test set first.")

    with tab3:
        st.header("Compare Implementations")
        st.markdown("Compare multiple RAG implementations side-by-side.")

        selected_implementations = st.multiselect(
            "Select implementations to compare:",
            [
                "ChromaDB Semantic Search",
                "Hybrid Search",
                "Neo4j Graph RAG",
                "Filesystem RAG",
            ],
        )

        if st.button("Run Comparison"):
            if len(selected_implementations) >= 2:
                with st.spinner("Running comparison..."):
                    # TODO: Implement comparison
                    st.info("Implementation coming soon!")
            else:
                st.warning("Please select at least 2 implementations to compare.")


if __name__ == "__main__":
    main()
