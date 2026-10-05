"""
responses.py
=============

A small registry formalising the response-function "contract" described in
CLAUDE.md's design decisions, so swapping in a built-in alternative or a
custom response function is a one-line, discoverable operation rather than
something you have to reverse-engineer from ``model.py``/``echofit.py``.

The contract (unchanged, just named here): a response function is any
callable ``psi(tau_grid, log_mdot, wavelength, inclination, M_BH, **kwargs)``
returning an array the same shape as ``tau_grid`` that is causal
(``psi[tau_grid < 0] == 0``) and area-normalised on ``tau_grid``
(``trapz(psi, tau_grid) == 1``). ``transfer_coeffs``/``compute_echo`` and the
plotting code never assume anything more specific than that.

Only "physical"-mode responses (the ``log_mdot``/``inclination``-driven
family) are covered here -- "free"-mode's ``tophat_response_free`` takes a
different contract (an explicit ``tau_mean`` rather than the
mass-accretion-rate-and-wavelength parameters) since its whole point is an
independently inferred lag per band, so it is deliberately not part of this
registry.

Registering a response makes it look-up-able by name; it does not, on its
own, change what a fit uses. To actually use one for "physical"-mode bands,
assign it to ``pycream2.model.response_function`` (both the NumPyro model and
the plotting code look this name up dynamically, so the swap is picked up
everywhere)::

    import pycream2.model as model
    from pycream2.responses import get_response

    model.response_function = get_response("thin_disk")

To add your own::

    from pycream2.responses import register_response

    def my_response(tau_grid, log_mdot, wavelength, inclination, M_BH, **kw):
        ...
        return psi

    register_response("my_response", my_response)
    model.response_function = get_response("my_response")
"""

from __future__ import annotations

from typing import Callable, Dict, Protocol

from .forward_model import response_function, thin_disk_response
from .rippled_disc import rippled_disc_response


class ResponseFunction(Protocol):
    """The response-function contract (see module docstring)."""

    def __call__(self, tau_grid, log_mdot, wavelength, inclination, M_BH, **kwargs):
        ...  # pragma: no cover


_REGISTRY: Dict[str, Callable] = {
    "skew_normal": response_function,
    "thin_disk": thin_disk_response,
    # Rimmed and rippled discs (Starkey, Huang, Horne & Lin 2023); flat by default,
    # the geometry set with functools.partial (see pycream2.rippled_disc).
    "rippled_disc": rippled_disc_response,
}


def register_response(name: str, fn: Callable) -> None:
    """Register a response function under ``name`` so :func:`get_response`
    can find it. Overwrites any existing entry with the same name."""
    _REGISTRY[name] = fn


def get_response(name: str) -> Callable:
    """Look up a registered response function by name.

    Raises
    ------
    KeyError
        If ``name`` isn't registered -- lists the available names.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"No response function registered as {name!r}. Available: {available}") from None


def available_responses() -> list[str]:
    """Names of all currently registered response functions."""
    return sorted(_REGISTRY)
