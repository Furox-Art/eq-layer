"""Typed boundary for the optional ``ml`` extra.

The trained adapters import scikit-learn and their artifact loaders import
joblib; both ship in ``eq-layer[ml]``. Every optional import in the package
goes through this module so a missing or unusable extra surfaces as one typed,
actionable error that names the package and the install command, instead of a
bare ``ImportError`` traceback at request time (audit M-2/M-7).
"""

from __future__ import annotations

ML_EXTRA = "ml"
ML_PACKAGE = "scikit-learn"
JOBLIB_PACKAGE = "joblib"
ML_INSTALL_COMMAND = 'pip install "eq-layer[ml]"'


class OptionalDependencyError(ImportError):
    """An optional dependency of the ``ml`` extra is missing or unusable.

    Subclasses ``ImportError`` so existing ``except ImportError`` fallbacks
    (the CLI's heuristic intent fallback under ``--intent-mode auto``) keep
    working, while callers that want to map the failure to their own exit code
    or message can catch the typed error instead of matching on message text.
    """

    def __init__(
        self,
        purpose: str,
        *,
        package: str = ML_PACKAGE,
        cause: BaseException | None = None,
        install_command: str = ML_INSTALL_COMMAND,
    ) -> None:
        detail = f" ({cause})" if cause is not None else ""
        super().__init__(
            f"{purpose} requires the optional ML dependencies: "
            f"{package} is unavailable{detail}. "
            f"Install with: {install_command}"
        )
        self.purpose = purpose
        self.package = package
        self.install_command = install_command


def missing_ml_dependency(purpose: str, cause: BaseException) -> OptionalDependencyError:
    """Build the typed error for a failed scikit-learn import."""
    return OptionalDependencyError(purpose, package=ML_PACKAGE, cause=cause)


def missing_joblib(purpose: str, cause: BaseException) -> OptionalDependencyError:
    """Build the typed error for a failed joblib import."""
    return OptionalDependencyError(purpose, package=JOBLIB_PACKAGE, cause=cause)


def ml_extra_available() -> bool:
    """Report whether the optional ML extra can be imported at all."""
    try:
        import sklearn  # noqa: F401
    except ImportError:
        return False
    return True
