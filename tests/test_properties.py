"""Property-based and differential tests for the invisicode format.

These are the tests that should survive any future refactor of the module. They
assert *behaviour and invariants* rather than particular internal steps, so the
vectorised implementation can be rewritten freely:

* **Round-trip properties** — ``decode(encode(x)) == x`` for every input class.
* **Differential tests** — the optimised implementation must agree exactly with
  the naive oracle in :mod:`tests.reference`.
* **Format invariants** — output alphabet, length formula, and type marking.

A failure here means either a real bug or an intentional format change.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import invisicode
from invisicode import (
	BASE,
	PADDING,
	RANGE,
	STRINGPREFIX,
	STRINGPREFIXES,
	InvisicodeDecodeError,
	decode,
	detect,
	detect_and_decode,
	encode,
	is_invisicode,
	l128_decode,
	l128_encode,
	str_to_u32,
)

from .reference import ref_decode, ref_encode, ref_l128_decode, ref_l128_encode
from .strategies import (
	BOUNDARY_CODEPOINTS,
	bytes_strategy,
	bytes_with_residue,
	contaminate,
	detectable_payload_strategy,
	foreign_text,
	payload_strategy,
	text_strategy,
	typed_arrays,
	uint8_arrays,
)


# ---------------------------------------------------------------------------
# Round-trip properties
# ---------------------------------------------------------------------------

class TestRoundTrip:
	"""``decode`` must invert ``encode`` for every supported input."""

	@given(bytes_strategy)
	def test_bytes(self, data: bytes):
		assert decode(encode(data)) == data

	@given(text_strategy)
	def test_text(self, text: str):
		assert decode(encode(text)) == text

	@given(text_strategy)
	def test_l128(self, text: str):
		assert l128_decode(l128_encode(text)) == text

	@pytest.mark.parametrize("residue", (0, 1, 2))
	@given(st.data())
	def test_every_padding_branch(self, residue: int, data):
		"""Each ``len % 3`` case round-trips, including both padded forms."""
		payload = data.draw(bytes_with_residue(residue))
		assert len(payload) % 3 == residue
		assert decode(encode(payload)) == payload

	@given(uint8_arrays())
	def test_uint8_array(self, arr: np.ndarray):
		assert decode(encode(arr)) == arr.tobytes()

	@given(typed_arrays())
	def test_any_dtype_array(self, arr: np.ndarray):
		"""Arrays of any dtype are encoded as their raw bytes."""
		assert decode(encode(arr)) == arr.tobytes()

	@given(bytes_strategy)
	def test_bytes_like_types_agree(self, data: bytes):
		"""``bytes``, ``bytearray`` and ``memoryview`` encode identically."""
		expected = encode(data)
		assert encode(bytearray(data)) == expected
		assert encode(memoryview(data)) == expected

	@given(payload_strategy)
	def test_decoding_an_array_matches_decoding_a_str(self, payload):
		"""``decode`` accepts its own output as either ``str`` or code point array."""
		encoded = encode(payload)
		assert decode(encoded) == decode(str_to_u32(encoded))

	@given(payload_strategy)
	def test_type_is_preserved(self, payload):
		"""Bytes decode back to bytes and text back to text."""
		result = decode(encode(payload))
		assert type(result) is type(payload)

	@given(bytes_strategy)
	def test_encode_is_deterministic(self, data: bytes):
		assert encode(data) == encode(data)


# ---------------------------------------------------------------------------
# Differential tests against the naive oracle
# ---------------------------------------------------------------------------

class TestAgainstReference:
	"""The vectorised implementation must match the pure-Python oracle exactly."""

	@given(bytes_strategy)
	def test_encode_bytes(self, data: bytes):
		assert encode(data) == ref_encode(data)

	@given(text_strategy)
	def test_encode_text(self, text: str):
		assert encode(text) == ref_encode(text)

	@given(text_strategy)
	def test_l128_encode(self, text: str):
		assert bytes(l128_encode(text)) == ref_l128_encode(text)

	@given(text_strategy)
	def test_l128_decode(self, text: str):
		payload = ref_l128_encode(text)
		assert l128_decode(payload) == ref_l128_decode(payload)

	@given(bytes_strategy)
	def test_decode_bytes(self, data: bytes):
		encoded = ref_encode(data)
		assert decode(encoded) == ref_decode(encoded)

	@given(text_strategy)
	def test_decode_text(self, text: str):
		encoded = ref_encode(text)
		assert decode(encoded) == ref_decode(encoded)

	@pytest.mark.parametrize("residue", (0, 1, 2))
	@given(st.data())
	def test_padding_branches_match(self, residue: int, data):
		payload = data.draw(bytes_with_residue(residue))
		assert encode(payload) == ref_encode(payload)

	@given(bytes_strategy)
	def test_reference_agrees_with_itself(self, data: bytes):
		"""Sanity-check the oracle, so a broken oracle cannot hide a real bug."""
		assert ref_decode(ref_encode(data)) == data


# ---------------------------------------------------------------------------
# Format invariants
# ---------------------------------------------------------------------------

class TestFormatInvariants:
	"""Structural guarantees that callers and the demo page rely on."""

	@given(bytes_strategy)
	def test_bytes_output_alphabet(self, data: bytes):
		"""Byte payloads use only the payload alphabet, with no prefix."""
		for ch in encode(data):
			assert BASE <= ord(ch) < BASE + RANGE

	@given(text_strategy)
	def test_text_output_is_prefixed(self, text: str):
		"""Text payloads start with the canonical prefix, then payload only."""
		encoded = encode(text)
		assert ord(encoded[0]) == STRINGPREFIX
		for ch in encoded[1:]:
			assert BASE <= ord(ch) < BASE + RANGE

	@given(bytes_strategy)
	def test_length_formula(self, data: bytes):
		"""Output length follows directly from the three-bytes-to-two-glyphs rule."""
		groups, excess = divmod(len(data), 3)
		expected = groups * 2 + (0, 1, 3)[excess]
		assert len(encode(data)) == expected

	@given(bytes_strategy)
	def test_expansion_ratio(self, data: bytes):
		"""Encoding never exceeds two thirds of the input length, plus padding."""
		assert len(encode(data)) <= len(data) * 2 / 3 + 3

	@given(bytes_strategy)
	def test_is_invisicode_accepts_own_output(self, data: bytes):
		assume(data)
		assert is_invisicode(encode(data))

	@given(text_strategy)
	def test_is_invisicode_accepts_prefixed_output(self, text: str):
		assert is_invisicode(encode(text))

	def test_empty_inputs(self):
		"""Empty payloads are representable and distinguishable by type."""
		assert encode(b"") == ""
		assert encode("") == chr(STRINGPREFIX)
		assert decode("") == b""
		assert decode(chr(STRINGPREFIX)) == ""

	def test_padding_character_may_appear_in_the_body(self):
		"""The padding marker is an ordinary payload value, not a reserved one.

		``0x0FFF`` is a perfectly reachable base-4096 digit, so ``PADDING`` can
		occur mid-string. It is unambiguous anyway: a body always contributes an
		even number of code points, so the marker is only *interpreted* as
		padding when the total length is odd.
		"""
		data = b"\xff\x0f\x00"  # low digit becomes 0x0FFF, i.e. PADDING
		encoded = encode(data)
		assert ord(encoded[0]) == PADDING
		assert len(encoded) % 2 == 0
		assert decode(encoded) == data

	@given(bytes_strategy)
	def test_padding_is_only_interpreted_at_odd_lengths(self, data: bytes):
		"""Whatever the body contains, the payload still round-trips."""
		encoded = encode(data)
		trailing = len(data) % 3
		# A padding marker terminates the string only in the two-byte case.
		if trailing == 2:
			assert ord(encoded[-1]) == PADDING
		assert decode(encoded) == data

	@pytest.mark.parametrize("byte", range(256))
	def test_single_byte_never_collides_with_padding(self, byte: int):
		"""Exhaustive proof that the one-byte suffix cannot be mistaken for padding."""
		assert (byte | BASE) != PADDING
		assert decode(encode(bytes((byte,)))) == bytes((byte,))

	@given(st.integers(min_value=0, max_value=0xFFFF))
	def test_two_byte_suffix_roundtrips(self, value: int):
		data = value.to_bytes(2, "little")
		assert decode(encode(data)) == data


# ---------------------------------------------------------------------------
# Type expectations
# ---------------------------------------------------------------------------

class TestExpect:
	"""The ``expect`` argument must gate on the recorded payload type."""

	@given(text_strategy)
	def test_expect_str_accepts_text(self, text: str):
		assert decode(encode(text), expect=str) == text

	@given(bytes_strategy)
	def test_expect_bytes_accepts_bytes(self, data: bytes):
		assert decode(encode(data), expect=bytes) == data

	@given(text_strategy)
	def test_expect_bytes_rejects_text(self, text: str):
		with pytest.raises(InvisicodeDecodeError):
			decode(encode(text), expect=bytes)

	@given(bytes_strategy)
	def test_expect_bytes_rejects_text_payload(self, data: bytes):
		assume(data)
		with pytest.raises(InvisicodeDecodeError):
			decode(encode(data), expect=str)

	@given(text_strategy)
	def test_all_accepted_prefixes_mark_text(self, text: str):
		"""Every prefix in ``STRINGPREFIXES`` is honoured on decode."""
		body = encode(text)[1:]
		for prefix in STRINGPREFIXES:
			assert decode(chr(prefix) + body) == text

	def test_only_canonical_prefix_is_emitted(self):
		"""Encoding must emit exactly one, stable prefix."""
		assert encode("x")[0] == chr(STRINGPREFIX)


# ---------------------------------------------------------------------------
# Strict and non-strict decoding
# ---------------------------------------------------------------------------

class TestStrictness:
	"""Strict mode rejects contamination; non-strict mode tolerates it."""

	@given(bytes_strategy, foreign_text, st.integers(min_value=0))
	def test_strict_rejects_contamination(self, data: bytes, junk: str, pos: int):
		polluted = contaminate(encode(data), junk, pos)
		assume(polluted != encode(data))
		with pytest.raises(InvisicodeDecodeError):
			decode(polluted)

	@given(bytes_strategy, foreign_text, foreign_text)
	def test_non_strict_strips_surrounding_junk(self, data: bytes, before: str, after: str):
		"""Leading and trailing foreign text is discarded, not fatal."""
		assert decode(before + encode(data) + after, strict=False) == data

	@given(text_strategy, foreign_text, foreign_text)
	def test_non_strict_preserves_text_type(self, text: str, before: str, after: str):
		"""Stripping junk must not discard the type prefix."""
		assert decode(before + encode(text) + after, strict=False) == text

	@given(bytes_strategy)
	def test_non_strict_matches_strict_on_clean_input(self, data: bytes):
		"""On valid input the two modes agree."""
		encoded = encode(data)
		assert decode(encoded, strict=False) == decode(encoded)

	@given(foreign_text)
	def test_non_strict_on_pure_junk_is_empty(self, junk: str):
		"""Input with no payload at all yields an empty result rather than raising.

		Documented behaviour: non-strict decoding can return meaningless output
		instead of reporting an error.
		"""
		assert decode(junk, strict=False) == b""


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

class TestDetect:
	"""``detect`` must return usable slice bounds that preserve payload types."""

	@given(
		st.lists(detectable_payload_strategy, min_size=1, max_size=5),
		st.lists(foreign_text, min_size=6, max_size=6),
	)
	def test_interleaved_payloads_round_trip(self, payloads, separators):
		"""Payloads embedded in arbitrary text are each recovered, in order."""
		text = separators[0]
		for payload, sep in zip(payloads, separators[1:]):
			text += encode(payload) + sep
		assert detect_and_decode(text) == payloads

	@given(
		st.lists(detectable_payload_strategy, min_size=1, max_size=4),
		st.lists(foreign_text, min_size=5, max_size=5),
	)
	def test_ranges_are_valid_slices(self, payloads, separators):
		"""Each returned range slices out something decodable."""
		text = separators[0]
		for payload, sep in zip(payloads, separators[1:]):
			text += encode(payload) + sep
		for start, end in detect(text):
			assert decode(text[start:end]) in payloads

	@given(detectable_payload_strategy)
	def test_payload_at_index_zero(self, payload):
		"""A payload at the very start keeps its prefix and is found."""
		assert detect_and_decode(encode(payload)) == [payload]

	@given(detectable_payload_strategy, foreign_text)
	def test_payload_after_leading_junk(self, payload, junk: str):
		"""Regression: leading foreign text once broke start/prefix alignment."""
		assert detect_and_decode(junk + encode(payload)) == [payload]

	def test_empty_bytes_payload_is_undetectable(self):
		"""An empty bytes payload encodes to "" and so leaves nothing to find.

		This is inherent to the format rather than a defect: there is no
		character in which to record that a payload was ever present.
		"""
		assert encode(b"") == ""
		assert detect_and_decode("junk" + encode(b"") + "junk") == []

	def test_empty_text_payload_is_detectable(self):
		"""An empty text payload still leaves its prefix behind."""
		assert detect_and_decode("junk" + encode("") + "junk") == [""]

	def test_prefix_immediately_before_a_bytes_payload_is_absorbed(self):
		"""A prefix character in the surrounding text is claimed by the payload.

		``detect`` cannot tell an intentional type marker from a prefix code point
		that merely happens to sit in the carrier text, so a bytes payload placed
		directly after one is read as text. This is inherent to embedding a
		self-delimiting format in arbitrary text; callers who need certainty
		should use :func:`decode` on a known slice instead.
		"""
		marker = chr(min(STRINGPREFIXES))
		text = marker + encode(b"\x00\x00\x00")
		# Detected as a single text payload, not as bytes.
		((start, end),) = detect(text)
		assert start == 0
		assert isinstance(decode(text[start:end]), str)
		# Decoding the payload alone, without the stray marker, is unambiguous.
		assert decode(encode(b"\x00\x00\x00")) == b"\x00\x00\x00"

	@given(foreign_text)
	def test_no_payload_found(self, junk: str):
		assert len(detect(junk)) == 0
		assert detect_and_decode(junk) == []

	def test_mixed_prefixed_and_unprefixed_segments(self):
		"""Regression: mixing text and bytes segments after index 0."""
		text = encode(b"xyz") + "b" + encode("hi") + "c" + encode(b"q")
		assert detect_and_decode(text) == [b"xyz", "hi", b"q"]

	@given(detectable_payload_strategy, foreign_text)
	def test_detect_ranges_are_half_open(self, payload, junk: str):
		"""Ranges are directly usable as ``[start:end]`` bounds."""
		encoded = encode(payload)
		text = junk + encoded + junk
		(start, end), = detect(text)
		assert text[start:end] == encoded

	@given(
		st.lists(detectable_payload_strategy, min_size=2, max_size=4),
		st.lists(foreign_text, min_size=5, max_size=5),
	)
	def test_ranges_are_ordered_and_disjoint(self, payloads, separators):
		text = separators[0]
		for payload, sep in zip(payloads, separators[1:]):
			text += encode(payload) + sep
		ranges = detect(text)
		for (_, prev_end), (next_start, _) in zip(ranges, ranges[1:]):
			assert prev_end <= next_start


# ---------------------------------------------------------------------------
# Malformed input
# ---------------------------------------------------------------------------

class TestMalformed:
	"""Corrupt input must raise a documented error, never crash or corrupt."""

	@pytest.mark.parametrize(
		"payload, reason",
		(
			(chr(PADDING), "lone padding marker is not a valid trailing byte"),
			(chr(BASE | 0x100), "single trailing group above 0xFF"),
			(chr(BASE | 0x100) * 2 + chr(PADDING), "two-byte trailing group above 0xFF"),
			(chr(STRINGPREFIX) + chr(BASE | 0x80), "truncated LEB128 body"),
		),
	)
	def test_rejected(self, payload: str, reason: str):
		with pytest.raises(InvisicodeDecodeError):
			decode(payload)

	@pytest.mark.parametrize("data", (b"\x80", b"abc\x80", b"\xff", b"\x80\x80\x80"))
	def test_truncated_l128(self, data: bytes):
		"""An unterminated continuation sequence is rejected either way it is caught."""
		with pytest.raises(InvisicodeDecodeError):
			l128_decode(data)
		with pytest.raises(UnicodeDecodeError):
			l128_decode(data)

	@pytest.mark.parametrize(
		"data",
		(
			b"\x80\x80\x80\x00",  # four bytes for one code point
			b"\x80\x80\x80\x80\x00",  # five
			b"\xff\xff\xff\x00",
		),
	)
	def test_overlong_l128_sequence(self, data: bytes):
		"""No code point needs more than three base-128 digits, so longer is invalid."""
		with pytest.raises(InvisicodeDecodeError):
			l128_decode(data)

	@pytest.mark.parametrize("container", (bytes, bytearray, memoryview))
	def test_error_reports_position_for_any_container(self, container):
		"""``UnicodeDecodeError`` needs real bytes, whatever the input type was."""
		with pytest.raises(UnicodeDecodeError) as info:
			l128_decode(container(b"ab\x80"))
		assert info.value.object == b"ab\x80"
		assert info.value.encoding == "invisicode"

	@given(text_strategy)
	def test_truncating_l128_payload_is_detected_or_lossy(self, text: str):
		"""Chopping a payload never silently invents a longer string."""
		payload = bytes(l128_encode(text))
		assume(len(payload) > 1)
		truncated = payload[:-1]
		try:
			result = l128_decode(truncated)
		except InvisicodeDecodeError:
			return
		assert len(result) <= len(text)

	@given(st.integers(min_value=0, max_value=BASE - 1))
	def test_out_of_range_codepoint_rejected(self, cp: int):
		assume(cp not in STRINGPREFIXES)
		# Surrogates cannot exist in a well-formed str and cannot be UTF-32 encoded,
		# so they are out of scope for the decoder's input contract.
		assume(not 0xD800 <= cp <= 0xDFFF)
		with pytest.raises(InvisicodeDecodeError):
			decode(chr(cp) * 2)

	def test_surrogates_are_out_of_contract(self):
		"""A ``str`` containing lone surrogates cannot be UTF-32 encoded at all.

		Such a string cannot arise from any real text source, so the resulting
		``UnicodeEncodeError`` is left to propagate rather than being wrapped.
		"""
		with pytest.raises(UnicodeEncodeError):
			decode("\ud800\ud800")

	def test_error_hierarchy(self):
		"""A single ``except`` clause can catch everything this module raises."""
		assert issubclass(invisicode.InvisicodeEncodeError, invisicode.InvisicodeError)
		assert issubclass(invisicode.InvisicodeDecodeError, invisicode.InvisicodeError)
		assert issubclass(invisicode.InvisicodeError, ValueError)
		assert issubclass(invisicode.InvisicodeUnicodeDecodeError, InvisicodeDecodeError)
		assert issubclass(invisicode.InvisicodeUnicodeDecodeError, UnicodeDecodeError)


# ---------------------------------------------------------------------------
# Boundary code points
# ---------------------------------------------------------------------------

class TestBoundaries:
	"""Explicit coverage of the values where off-by-one bugs live."""

	@pytest.mark.parametrize("cp", BOUNDARY_CODEPOINTS)
	def test_codepoint_round_trip(self, cp: int):
		s = chr(cp)
		assert decode(encode(s)) == s
		assert l128_decode(l128_encode(s)) == s
		assert encode(s) == ref_encode(s)

	@pytest.mark.parametrize("cp", (BASE, BASE + 1, PADDING - 1, PADDING))
	def test_alphabet_bounds(self, cp: int):
		assert invisicode.is_invisicode_codepoint(cp)

	@pytest.mark.parametrize("cp", (BASE - 1, BASE + RANGE, 0, 0x10FFFF))
	def test_outside_alphabet(self, cp: int):
		assert not invisicode.is_invisicode_codepoint(cp)

	@pytest.mark.parametrize("prefix", sorted(STRINGPREFIXES))
	def test_prefixes_are_codepoints_but_not_payload(self, prefix: int):
		"""Prefixes count as invisicode only when prefixes are allowed."""
		assert invisicode.is_invisicode_codepoint(prefix, allow_prefixes=True)
		assert not invisicode.is_invisicode_codepoint(prefix, allow_prefixes=False)


# ---------------------------------------------------------------------------
# Larger inputs
# ---------------------------------------------------------------------------

class TestScale:
	"""A couple of larger cases, kept small enough for routine CI runs."""

	@pytest.mark.parametrize("size", (3 * 1024, 3 * 1024 + 1, 3 * 1024 + 2))
	def test_kilobyte_scale_round_trip(self, size: int):
		rng = np.random.default_rng(0xA11CE)
		data = rng.integers(0, 256, size=size, dtype=np.uint8)
		assert decode(encode(data)) == data.tobytes()

	@settings(max_examples=10, deadline=None)
	@given(st.integers(min_value=0, max_value=4096))
	def test_arbitrary_sizes_match_reference(self, size: int):
		rng = np.random.default_rng(size)
		data = rng.integers(0, 256, size=size, dtype=np.uint8).tobytes()
		assert encode(data) == ref_encode(data)

	@pytest.mark.slow
	def test_megabyte_scale_round_trip(self):
		"""Deselect with ``-m 'not slow'`` for a fast inner-loop run."""
		rng = np.random.default_rng(0xBEEF)
		data = rng.integers(0, 256, size=3 * 10**6 + 2, dtype=np.uint8)
		encoded = encode(data)
		assert decode(encoded) == data.tobytes()
