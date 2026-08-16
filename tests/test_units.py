"""Example-based unit tests for individual helpers.

These pin down the exact contract of each small function, including the awkward
cases that property tests describe only in general terms. Where a value is
asserted literally, a comment explains where the number comes from, so that a
future change can tell a deliberate format edit from a regression.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import invisicode
from invisicode import (
	BASE,
	PADDING,
	RANGE,
	STRINGPREFIX,
	STRINGPREFIXES,
	InvisicodeEncodeError,
	as_u32,
	as_u8,
	decode_leb128,
	is_invisicode,
	is_invisicode_codepoint,
	l128_encode,
	leb128,
	str_to_u32,
	u32_to_str,
)

from .reference import ref_leb128_encode_int


class TestConstants:
	"""The format constants must stay mutually consistent."""

	def test_alphabet_is_4096_wide(self):
		assert RANGE == 0x1000

	def test_padding_is_last_codepoint_in_alphabet(self):
		assert PADDING == BASE + RANGE - 1

	def test_canonical_prefix_is_accepted(self):
		assert STRINGPREFIX in STRINGPREFIXES

	def test_prefixes_are_outside_the_payload_alphabet(self):
		"""A prefix must never be confusable with payload."""
		for prefix in STRINGPREFIXES:
			assert not BASE <= prefix < BASE + RANGE

	def test_alphabet_fits_three_bytes_in_two_codepoints(self):
		"""Two base-4096 digits must span a full 24-bit group."""
		assert RANGE**2 == 1 << 24

	def test_public_api_is_exported(self):
		assert invisicode.__all__
		missing = [name for name in invisicode.__all__ if not hasattr(invisicode, name)]
		assert missing == []

	def test_version_is_present(self):
		assert isinstance(invisicode.__version__, str)


class TestU32Conversion:
	"""``str_to_u32`` and ``u32_to_str`` must be exact inverses."""

	def test_ascii(self):
		np.testing.assert_array_equal(
			str_to_u32("hello"),
			np.array([104, 101, 108, 108, 111], dtype=np.uint32),
		)

	def test_empty(self):
		assert str_to_u32("").size == 0
		assert u32_to_str(np.empty(0, dtype=np.uint32)) == ""

	def test_dtype_is_uint32(self):
		assert str_to_u32("x").dtype == np.uint32

	@given(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=32))
	def test_round_trip(self, text: str):
		assert u32_to_str(str_to_u32(text)) == text

	def test_astral_codepoint_is_one_element(self):
		"""UTF-32 keeps astral characters as a single code point, unlike UTF-16."""
		assert str_to_u32("\U0001F30D").size == 1


class TestAsU32:
	"""``as_u32`` normalises both accepted input types."""

	def test_accepts_str(self):
		np.testing.assert_array_equal(as_u32("a"), np.array([97], dtype=np.uint32))

	def test_accepts_array(self):
		arr = np.array([97, 98], dtype=np.uint32)
		np.testing.assert_array_equal(as_u32(arr), arr)

	def test_accepts_non_contiguous_array(self):
		"""Regression: ``ndarray.view`` cannot reinterpret a strided buffer."""
		arr = np.arange(8, dtype=np.uint32)[::2]
		np.testing.assert_array_equal(as_u32(arr), np.array([0, 2, 4, 6], dtype=np.uint32))


class TestAsU8:
	"""``as_u8`` flattens anything bytes-like to a contiguous uint8 view."""

	@pytest.mark.parametrize(
		"value",
		(b"abc", bytearray(b"abc"), memoryview(b"abc")),
	)
	def test_bytes_like(self, value):
		np.testing.assert_array_equal(as_u8(value), np.frombuffer(b"abc", dtype=np.uint8))

	def test_wide_dtype_becomes_raw_bytes(self):
		"""Regression: wide dtypes used to reshape-crash or encode padding bytes."""
		arr = np.array([1], dtype=np.uint32)
		np.testing.assert_array_equal(as_u8(arr), np.array([1, 0, 0, 0], dtype=np.uint8))

	def test_non_contiguous(self):
		"""Regression: strided arrays used to raise ``BufferError``."""
		arr = np.arange(6, dtype=np.uint8)[::2]
		np.testing.assert_array_equal(as_u8(arr), np.array([0, 2, 4], dtype=np.uint8))

	def test_multidimensional_flattens_in_c_order(self):
		arr = np.arange(4, dtype=np.uint8).reshape((2, 2))
		np.testing.assert_array_equal(as_u8(arr), np.array([0, 1, 2, 3], dtype=np.uint8))

	def test_empty(self):
		assert as_u8(b"").size == 0

	def test_rejects_unsupported_type(self):
		"""Non-buffer input raises the module's own error, not a raw TypeError."""
		with pytest.raises(InvisicodeEncodeError):
			as_u8(object())  # type: ignore[arg-type]


class TestLeb128:
	"""The scalar LEB128 codec, including the custom negative-number extension."""

	@pytest.mark.parametrize(
		"value, expected",
		(
			(0, b"\x00"),  # zero must still occupy one byte
			(1, b"\x01"),
			(127, b"\x7f"),  # largest single-byte value
			(128, b"\x80\x01"),  # first two-byte value
		),
	)
	def test_positive(self, value: int, expected: bytes):
		assert bytes(leb128(value)) == expected

	@pytest.mark.parametrize(
		"value, expected",
		(
			(-1, b"\x81\x00"),  # continuation set, then the 00 sign byte
			(-127, b"\xff\x00"),
			(-128, b"\x80\x81\x00"),
		),
	)
	def test_negative(self, value: int, expected: bytes):
		"""Negatives are signed by a trailing 00 byte, unlike SLEB128."""
		assert bytes(leb128(value)) == expected

	def test_zero_is_not_treated_as_negative(self):
		"""Regression: ``n <= 0`` once sent zero down the negative branch."""
		assert bytes(leb128(0)) == b"\x00"

	def test_positive_encoding_matches_standard_leb128(self):
		"""Non-negative values must stay wire-compatible with standard LEB128."""
		for value in (0, 1, 63, 64, 127, 128, 300, 16383, 16384, 1 << 20):
			assert bytes(leb128(value)) == ref_leb128_encode_int(value)

	@given(st.integers(min_value=-(1 << 40), max_value=1 << 40))
	def test_round_trip(self, value: int):
		decoded, remaining = decode_leb128(leb128(value))
		assert decoded == value
		assert len(remaining) == 0

	def test_decode_returns_remaining_data(self):
		value, remaining = decode_leb128(bytearray([1, 2, 3]))
		assert value == 1
		assert bytes(remaining) == b"\x02\x03"

	@pytest.mark.parametrize("container", (bytes, bytearray, memoryview))
	def test_decode_accepts_any_bytes_like(self, container):
		value, _ = decode_leb128(container(b"\x80\x01"))
		assert value == 128

	def test_stream_decoding(self):
		"""Concatenated values can be pulled off one at a time."""
		payload = leb128(1) + leb128(300) + leb128(-7)
		remaining: object = bytes(payload)
		out = []
		while remaining:
			value, remaining = decode_leb128(remaining)  # type: ignore[arg-type]
			out.append(value)
		assert out == [1, 300, -7]


class TestL128:
	"""The vectorised text codec."""

	def test_returns_memoryview_even_when_empty(self):
		"""Regression: the empty case once returned ``bytes`` instead."""
		assert isinstance(l128_encode(""), memoryview)
		assert isinstance(l128_encode("a"), memoryview)

	def test_ascii_is_unchanged(self):
		"""ASCII is a fixed point of the encoding, like UTF-8."""
		assert bytes(l128_encode("test")) == b"test"

	def test_beats_utf8_for_this_example(self):
		"""The documented efficiency claim from the README."""
		text = "Hello World! \u2764\ufe0f"
		assert len(l128_encode(text)) == 18
		assert len(text.encode("utf-8")) == 19

	def test_matches_concatenated_scalar_encoder(self):
		text = "".join(chr(cp) for cp in (0, 1, 0x7F, 0x80, 0x4000, 0x10FFFF))
		expected = b"".join(ref_leb128_encode_int(ord(ch)) for ch in text)
		assert bytes(l128_encode(text)) == expected


class TestIsInvisicodeCodepoint:
	"""Single code point classification."""

	@pytest.mark.parametrize("cp", (BASE, BASE + 1, PADDING))
	def test_in_alphabet(self, cp: int):
		assert is_invisicode_codepoint(cp)

	@pytest.mark.parametrize("cp", (BASE - 1, BASE + RANGE, ord("a")))
	def test_outside_alphabet(self, cp: int):
		assert not is_invisicode_codepoint(cp)

	def test_prefix_depends_on_allow_prefixes(self):
		assert is_invisicode_codepoint(STRINGPREFIX, allow_prefixes=True)
		assert not is_invisicode_codepoint(STRINGPREFIX, allow_prefixes=False)

	def test_accepts_numpy_scalar(self):
		"""``decode`` passes array elements in, so NumPy scalars must work."""
		assert is_invisicode_codepoint(np.uint32(BASE))


class TestIsInvisicode:
	"""Whole-string classification, in both strictness modes."""

	def test_strict_rejects_empty(self):
		assert not is_invisicode("", strict=True)

	def test_non_strict_accepts_empty(self):
		assert is_invisicode("", strict=False)

	def test_strict_accepts_bare_prefix(self):
		"""A lone prefix is the valid encoding of an empty string."""
		assert is_invisicode(chr(STRINGPREFIX), strict=True)

	def test_strict_rejects_mixed(self):
		assert not is_invisicode(invisicode.encode(b"test") + "a")

	def test_non_strict_accepts_mixed(self):
		assert is_invisicode(invisicode.encode(b"test") + "a", strict=False)

	def test_non_strict_rejects_pure_foreign_text(self):
		assert not is_invisicode("hello", strict=False)

	def test_multi_element_array(self):
		"""Regression: arrays once raised an ambiguous-truth ``ValueError``."""
		assert not is_invisicode(np.array([1, 2, 3], dtype=np.uint32))
		assert is_invisicode(str_to_u32(invisicode.encode(b"test")))

	@pytest.mark.parametrize("strict", (True, False))
	def test_empty_array_matches_empty_string(self, strict: bool):
		empty = np.empty(0, dtype=np.uint32)
		assert is_invisicode(empty, strict=strict) == is_invisicode("", strict=strict)
