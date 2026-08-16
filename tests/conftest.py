"""Shared pytest configuration for the invisicode suite."""

from __future__ import annotations

from hypothesis import HealthCheck, settings

# A profile for everyday local runs: fast enough to keep in an edit/test loop.
settings.register_profile(
	"dev",
	max_examples=50,
	deadline=None,
	suppress_health_check=(HealthCheck.too_slow,),
)

# CI can afford more examples, which is where rare shrinking cases surface.
settings.register_profile(
	"ci",
	max_examples=300,
	deadline=None,
	suppress_health_check=(HealthCheck.too_slow,),
)

# For deliberate, long fuzzing sessions after a format or performance change.
settings.register_profile(
	"thorough",
	max_examples=5000,
	deadline=None,
	suppress_health_check=(HealthCheck.too_slow,),
)

settings.load_profile("dev")
