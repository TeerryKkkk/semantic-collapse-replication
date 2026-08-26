"""Canonical five-class model-attribution classifier."""

from .pipeline import CLASS_ORDER, FAMILY_ORDER

PRIMARY_WORKFLOW = "independent_reference"

__all__ = ["CLASS_ORDER", "FAMILY_ORDER", "PRIMARY_WORKFLOW"]
