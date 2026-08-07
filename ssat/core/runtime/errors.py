"""Execution-layer exceptions."""


class RuntimeExecutionError(RuntimeError):
    """Indicate a terminal execution failure or violated runtime invariant."""


class RuntimeContractError(RuntimeExecutionError):
    """Indicate invalid data crossing a worker/main-process boundary."""
