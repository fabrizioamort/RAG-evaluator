"""Custom exception hierarchy for the application."""

from typing import Any


class AppException(Exception):
    """Base category for all application-specific exceptions."""

    def __init__(
        self,
        detail: str,
        status_code: int = 500,
        errors: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code
        self.errors = errors



class BadRequestError(AppException):
    """Exception raised for bad requests (400)."""

    def __init__(self, detail: str = "Bad request") -> None:
        super().__init__(detail, status_code=400)


class NotFoundError(AppException):
    """Exception raised when a resource is not found."""

    def __init__(self, detail: str = "Resource not found") -> None:
        super().__init__(detail, status_code=404)


class ValidationError(AppException):
    """Exception raised when validation fails."""

    def __init__(
        self, detail: str = "Validation failed", errors: list[dict[str, Any]] | None = None
    ) -> None:
        super().__init__(detail, status_code=422, errors=errors)


class ConflictError(AppException):
    """Exception raised when there is a conflict (e.g., unique constraint)."""

    def __init__(self, detail: str = "Conflict occurred") -> None:
        super().__init__(detail, status_code=409)


class UnauthorizedError(AppException):
    """Exception raised when authentication is required or fails."""

    def __init__(self, detail: str = "Not authenticated") -> None:
        super().__init__(detail, status_code=401)


class ForbiddenError(AppException):
    """Exception raised when a user does not have permission."""

    def __init__(self, detail: str = "Permission denied") -> None:
        super().__init__(detail, status_code=403)
