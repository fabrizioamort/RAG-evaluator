import threading

from rag_evaluator.common.token_tracker import TokenUsage


def test_token_usage_to_dict_no_deadlock():
    """Test that to_dict() does not deadlock even when called concurrently or nested."""
    usage = TokenUsage(prompt_tokens=10, completion_tokens=20, embedding_tokens=30)

    # This would have deadlocked before the fix because to_dict()
    # acquired the lock and then called total_tokens property which
    # also tried to acquire the same non-reentrant lock.
    data = usage.to_dict()

    assert data["prompt_tokens"] == 10
    assert data["completion_tokens"] == 20
    assert data["embedding_tokens"] == 30
    assert data["total_tokens"] == 60

def test_token_usage_concurrent_updates():
    """Test that TokenUsage handles concurrent updates correctly."""
    usage = TokenUsage()
    num_threads = 10
    updates_per_thread = 100

    def update_task():
        for _ in range(updates_per_thread):
            usage.add_prompt_tokens(1)
            usage.add_completion_tokens(1)
            usage.add_embedding_tokens(1)
            # Mix in some to_dict calls to check for deadlocks during updates
            usage.to_dict()

    threads = [threading.Thread(target=update_task) for _ in range(num_threads)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert usage.prompt_tokens == num_threads * updates_per_thread
    assert usage.completion_tokens == num_threads * updates_per_thread
    assert usage.embedding_tokens == num_threads * updates_per_thread
    assert usage.total_tokens == 3 * num_threads * updates_per_thread
