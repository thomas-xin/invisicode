"""Shared Hypothesis strategies and helpers for the invisicode test suite.

Centralising the strategies here means a future format change (a wider alphabet,
a new prefix) only needs updating in one place, and every property test picks the
change up automatically.
"""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st

from invisicode import BASE, RANGE, STRINGPREFIXES

#: Code points that are legal in a Python ``str``. Surrogates are excluded because
#: they cannot survive a UTF-32 round-trip.
MAX_CODEPOINT = 0x10FFFF
SURROGATE_LO = 0xD800
SURROGATE_HI = 0xDFFF


def _no_surrogates(cp: int) -> bool:
	return not SURROGATE_LO <= cp <= SURROGATE_HI


# --- Text strategies -------------------------------------------------------

#: Text drawn from the whole Unicode range, excluding surrogates.
any_text = st.text(
	alphabet=st.characters(blacklist_categories=("Cs",)),
	max_size=64,
)

#: Code points sitting exactly on the LEB128 width boundaries, where off-by-one
#: errors in the vectorised encoder would show up first.
BOUNDARY_CODEPOINTS = (
	0x00,
	0x01,
	0x7F,  # last 1-byte value
	0x80,  # first 2-byte value
	0x81,
	0x3FFF,  # last 2-byte value
	0x4000,  # first 3-byte value
	0x4001,
	0xD7FF,  # just below the surrogate block
	0xE000,  # just above the surrogate block
	0xFFFF,
	0x10000,  # first astral code point
	MAX_CODEPOINT,
)

#: Text built only from boundary code points, in any order and multiplicity.
boundary_text = st.lists(
	st.sampled_from([chr(cp) for cp in BOUNDARY_CODEPOINTS]),
	max_size=48,
).map("".join)

#: A mixture of arbitrary and boundary-heavy text.
text_strategy = st.one_of(any_text, boundary_text)


# --- Bytes strategies ------------------------------------------------------

#: Arbitrary byte strings. Sizes deliberately span every ``len % 3`` case.
bytes_strategy = st.binary(max_size=96)

#: Byte strings whose length is pinned to a specific residue mod 3, to force the
#: padding branches to be exercised even under aggressive shrinking.
def bytes_with_residue(residue: int) -> st.SearchStrategy[bytes]:
	"""Return a strategy for byte strings where ``len(x) % 3 == residue``."""
	return st.integers(min_value=0, max_value=24).flatmap(
		lambda groups: st.binary(
			min_size=groups * 3 + residue,
			max_size=groups * 3 + residue,
		)
	)


#: Payloads of either supported type, for tests that should not care which.
payload_strategy = st.one_of(bytes_strategy, text_strategy)

#: Payloads that leave at least one character behind once encoded, and so can be
#: located by :func:`invisicode.detect`. An empty *bytes* payload encodes to the
#: empty string and is therefore inherently undetectable; an empty *text* payload
#: still leaves its prefix, so it remains findable.
detectable_payload_strategy = payload_strategy.filter(lambda p: p != b"")


# --- Invisicode-shaped strategies -----------------------------------------

#: A single valid payload code point.
invisicode_codepoint = st.integers(min_value=BASE, max_value=BASE + RANGE - 1)

def _is_foreign(ch: str) -> bool:
	"""Whether a character is neither payload nor a recognised type prefix.

	Prefixes must be excluded: they are meaningful to the decoder, so text
	containing one is not inert filler.
	"""
	cp = ord(ch)
	return not (BASE <= cp < BASE + RANGE) and cp not in STRINGPREFIXES


#: Characters that are never valid invisicode payload, for contamination tests.
foreign_text = st.text(
	alphabet=st.characters(
		blacklist_categories=("Cs",),
		max_codepoint=0xFFFF,
	).filter(_is_foreign),
	min_size=1,
	max_size=8,
)


# --- NumPy array strategies ----------------------------------------------

#: dtypes that `encode` should accept by reinterpreting the underlying bytes.
ARRAY_DTYPES = (
	np.uint8,
	np.int8,
	np.uint16,
	np.int16,
	np.uint32,
	np.int32,
	np.uint64,
	np.int64,
	np.float32,
	np.float64,
)


@st.composite
def uint8_arrays(draw, max_size: int = 64) -> np.ndarray:
	"""Draw a 1-D C-contiguous ``uint8`` array."""
	data = draw(st.binary(max_size=max_size))
	return np.frombuffer(data, dtype=np.uint8).copy()


@st.composite
def typed_arrays(draw, max_items: int = 16) -> np.ndarray:
	"""Draw a 1-D array of an arbitrary supported dtype."""
	dtype = draw(st.sampled_from(ARRAY_DTYPES))
	n = draw(st.integers(min_value=0, max_value=max_items))
	raw = draw(st.binary(min_size=n * np.dtype(dtype).itemsize, max_size=n * np.dtype(dtype).itemsize))
	return np.frombuffer(raw, dtype=dtype).copy()


# --- Helpers --------------------------------------------------------------

def contaminate(payload: str, junk: str, position: int) -> str:
	"""Insert ``junk`` into ``payload`` at a wrapped index, for non-strict tests."""
	if not payload:
		return junk
	i = position % (len(payload) + 1)
	return payload[:i] + junk + payload[i:]
