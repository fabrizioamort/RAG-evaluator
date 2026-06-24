"""Neo4j connection helpers for Graph RAG."""

from neo4j import Driver, GraphDatabase


class Neo4jConnectionError(RuntimeError):
    """Raised when connecting to Neo4j fails."""


def _is_blank(value: str | None) -> bool:
    return value is None or value.strip() == ""


def resolve_neo4j_connection_params(
    uri: str | None,
    username: str | None,
    password: str | None,
    *,
    default_uri: str,
    default_username: str,
    default_password: str,
) -> tuple[str, str, str]:
    """Resolve Neo4j connection params with blank-aware fallback semantics."""
    if _is_blank(uri):
        resolved_uri = default_uri
    else:
        assert uri is not None
        resolved_uri = uri.strip()

    if _is_blank(username):
        resolved_username = default_username
    else:
        assert username is not None
        resolved_username = username.strip()

    if _is_blank(password):
        resolved_password = default_password
    else:
        # Preserve non-blank passwords exactly as provided.
        resolved_password = password or default_password

    return resolved_uri, resolved_username, resolved_password


def format_neo4j_connection_error(exc: Exception, *, uri: str, username: str) -> str:
    """Build a clear, user-facing Neo4j connection error message."""
    exc_name = exc.__class__.__name__
    base = f"Cannot connect to Neo4j at '{uri}' using username '{username}'."

    if exc_name in {"AuthError", "AuthErrorV2", "Forbidden"}:
        hint = " Authentication failed. Check Neo4j username/password."
    elif exc_name in {"ServiceUnavailable", "SessionExpired"}:
        hint = " Neo4j is unreachable. Verify the URI and that the service is running."
    elif exc_name in {"ConfigurationError"}:
        hint = " Invalid Neo4j connection configuration."
    else:
        hint = " Connection attempt failed."

    return f"{base}{hint} Original error: {exc}"


def test_neo4j_connection(uri: str, username: str, password: str) -> None:
    """Verify Neo4j connectivity and auth, raising Neo4jConnectionError on failure."""
    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        if hasattr(driver, "verify_connectivity"):
            driver.verify_connectivity()
        else:
            with driver.session() as session:
                session.run("RETURN 1").consume()
    except Exception as exc:
        raise Neo4jConnectionError(
            format_neo4j_connection_error(exc, uri=uri, username=username)
        ) from exc
    finally:
        driver.close()


def create_verified_neo4j_driver(uri: str, username: str, password: str) -> Driver:
    """Create a Neo4j driver and verify connectivity before returning it."""
    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        if hasattr(driver, "verify_connectivity"):
            driver.verify_connectivity()
        else:
            with driver.session() as session:
                session.run("RETURN 1").consume()
    except Exception as exc:
        driver.close()
        raise Neo4jConnectionError(
            format_neo4j_connection_error(exc, uri=uri, username=username)
        ) from exc
    return driver
