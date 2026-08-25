"""Canonical five-class model-attribution classifier."""

from .pipeline import CLASS_ORDER, FAMILY_ORDER, FOLDS

PRIMARY_WORKFLOW = "independent_reference"

__all__ = ["CLASS_ORDER", "FAMILY_ORDER", "FOLDS", "PRIMARY_WORKFLOW"]
