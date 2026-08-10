"""Execution-layer exceptions."""


class RuntimeExecutionError(RuntimeError):
    """Indicate a terminal execution failure or violated runtime invariant."""


class RuntimeContractError(RuntimeExecutionError):
    """Indicate invalid data crossing a worker/main-process boundary."""


class RuntimeCancelledError(RuntimeExecutionError):
    """Indicate cooperative cancellation after durable runtime flush."""
