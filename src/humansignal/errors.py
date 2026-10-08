"""Errors raised for caller mistakes (empty input), not detector failures."""


class InputError(ValueError):
    """The request has no usable text or transcript."""
