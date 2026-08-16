"""Encode arbitrary data into strings that render invisibly on Unicode-aware platforms.

The payload alphabet contains the 4096 code points ``U+E0000`` to ``U+E0FFF``, classified ``Default_Ignorable_Code_Point`` in the Unicode standard (the assigned Tags and Variation Selectors Supplement characters directly, and the unassigned remainder via ``Other_Default_Ignorable_Code_Point``), meaning the majority of platforms will "correctly" display nothing.
"""

from __future__ import annotations

from typing import overload

import numpy as np

__version__ = "1.2.1"

__all__ = [
	"BASE",
	"RANGE",
	"STRINGPREFIX",
	"STRINGPREFIXES",
	"PADDING",
	"InvisicodeError",
	"InvisicodeEncodeError",
	"InvisicodeDecodeError",
	"InvisicodeUnicodeDecodeError",
	"u32_to_str",
	"str_to_u32",
	"as_u32",
	"as_u8",
	"leb128",
	"decode_leb128",
	"l128_encode",
	"l128_decode",
	"encode",
	"decode",
	"is_invisicode_codepoint",
	"is_invisicode",
	"detect",
	"detect_and_decode",
]

BASE = 0xe0000
RANGE = 0x1000
# Code point emitted to mark a payload as text rather than bytes.
# `U+1D17A` (MUSICAL SYMBOL END PHRASE) is `Cf`, default-ignorable and normalisation-stable under NFC/NFKC.
STRINGPREFIX = 0x1d17a
# Every code point accepted as a string marker when decoding. Emitting only
# `STRINGPREFIX` while accepting this whole set allows the emitted prefix to be changed in future without invalidating existing payloads.
STRINGPREFIXES = frozenset((STRINGPREFIX,))
PADDING = BASE + RANGE - 1


class InvisicodeError(ValueError):
	"""Base class for every error raised by this module."""
class InvisicodeEncodeError(InvisicodeError):
	"""Raised when input cannot be encoded."""
class InvisicodeDecodeError(InvisicodeError):
	"""Raised when input cannot be decoded."""
class InvisicodeUnicodeDecodeError(InvisicodeDecodeError, UnicodeDecodeError):
	"""Malformed base-128 text payload.

	Subclasses both :class:`InvisicodeDecodeError` and :class:`UnicodeDecodeError`
	so that either may be caught, since a failure to decode text is both.
	"""


def u32_to_str(arr: np.ndarray) -> str:
	"""Convert an array of UTF-32 code points into a Python string."""
	enc = "utf-32-le"
	return np.asanyarray(arr, dtype=np.uint32).tobytes().decode(enc)
def str_to_u32(s: str) -> np.ndarray:
	"""Encode a string into a NumPy array of UTF-32 little-endian code points."""
	enc = "utf-32-le"
	return np.frombuffer(s.encode(enc), dtype=np.uint32)

def as_u32(s: str | np.ndarray) -> np.ndarray:
	"""Return a UTF-32 view for either a string or an existing NumPy array.

	Arrays are made C-contiguous first, since ``ndarray.view`` cannot reinterpret
	a strided buffer.
	"""
	if isinstance(s, np.ndarray):
		return np.ascontiguousarray(s).view(np.uint32)
	return str_to_u32(s)


def as_u8(b: bytes | bytearray | memoryview | np.ndarray) -> np.ndarray:
	"""Return a flat, C-contiguous ``uint8`` view of any bytes-like or array input.

	Arrays of a wider dtype are reinterpreted as their underlying bytes, which
	makes the little-endian byte order of the input significant.
	"""
	if isinstance(b, np.ndarray):
		return np.ascontiguousarray(b).reshape(-1).view(np.uint8)
	try:
		return np.frombuffer(b, dtype=np.uint8)
	except (TypeError, BufferError) as ex:
		raise InvisicodeEncodeError(f"Cannot interpret {type(b).__name__} as bytes") from ex


def leb128(n: int) -> bytearray:
	"Encodes an integer using a custom LEB128 algorithm. Supports a sign for negative integers via an additional 00 byte, maintaining compatibility with standard LEB128 (unlike SLEB128)."
	if n <= 0:
		was_negative = True
		n = -n
	else:
		was_negative = False
	data = bytearray()
	while n > 0:
		data.append(n & 0x7F)
		n >>= 7
		if n:
			data[-1] |= 0x80
	if was_negative:
		if len(data):
			data[-1] |= 0x80
		data.append(0)
	return data
def decode_leb128(data: bytes | bytearray | memoryview) -> tuple[int, bytes | bytearray | memoryview]:
	"Decodes an integer from LEB128 encoded data; returns a tuple of decoded and remaining data. The remaining data is a slice of the input, and so keeps the input's type."
	i = n = 0
	shift = 0
	for i, byte in enumerate(data):
		n |= (byte & 0x7F) << shift
		if byte & 0x80 == 0:
			if byte == 0:
				n = -n
			break
		else:
			shift += 7
	return n, data[i + 1:]


def l128_encode(s: str) -> memoryview:
	"""Encode a text string using variable-length base-128 encoding.

	Always returns a ``memoryview``, including for empty input.
	"""
	cp = str_to_u32(s) if s else np.empty(0, dtype=np.uint32)
	if cp.size == 0:
		return memoryview(b"")

	ge_128 = cp >= 0x80
	ge_16384 = cp >= 0x4000
	ge_128_c = np.count_nonzero(ge_128)
	ge_16384_c = np.count_nonzero(ge_16384)
	total_length = int(cp.size + ge_128_c + ge_16384_c)
	out = np.empty(total_length, dtype=np.uint8)
	size = cp.size

	lengths = np.ones(size, dtype=np.uint8)
	lengths[ge_128] = 2
	lengths[ge_16384] = 3
	lengths64 = lengths.astype(np.int64)
	offsets = lengths64.cumsum()
	offsets -= lengths64
	low = (cp & 0x7F).view(np.uint8)[::4]
	mid = ((cp >> 7) & 0x7F).view(np.uint8)[::4]
	high = (cp >> 14).view(np.uint8)[::4]

	mask1 = lengths == 1
	if mask1.any():
		out[offsets[mask1]] = low[mask1]
	mask2 = lengths == 2
	if mask2.any():
		pos = offsets[mask2]
		out[pos] = low[mask2] | 0x80
		out[pos + 1] = mid[mask2]
	mask3 = lengths == 3
	if mask3.any():
		pos = offsets[mask3]
		out[pos] = low[mask3] | 0x80
		out[pos + 1] = mid[mask3] | 0x80
		out[pos + 2] = high[mask3]
	return out.data

def l128_decode(b: bytes | bytearray | memoryview) -> str:
	"""Decode bytes produced by l128_encode back into a Unicode string.

	Raises :class:`InvisicodeUnicodeDecodeError` (both an
	:class:`InvisicodeDecodeError` and a :class:`UnicodeDecodeError`) on
	malformed input.
	"""
	if not b:
		return ""
	data = np.frombuffer(b, dtype=np.uint8)

	termination_mask = (data & 0x80) == 0
	if not termination_mask[-1]:
		raw = bytes(b)
		raise InvisicodeUnicodeDecodeError("invisicode", raw, len(raw) - 1, len(raw), "Incomplete LEB128 sequence")
	ends = np.flatnonzero(termination_mask)
	if data.size < 2 ** 32:
		ends = ends.astype(np.uint32)
	starts = ends.copy()
	starts[0] = 0
	if starts.size > 1:
		starts[1:] = ends[:-1] + 1
	lengths = ends - starts + 1
	if np.any((lengths < 1) | (lengths > 3)):
		raw = bytes(b)
		raise InvisicodeUnicodeDecodeError("invisicode", raw, 0, len(raw), "Invalid LEB128 codepoint length")

	cp = np.empty(ends.size, dtype=np.uint32)
	mask1 = lengths == 1
	if mask1.any():
		idx = starts[mask1]
		cp[mask1] = data[idx]
	mask2 = lengths == 2
	if mask2.any():
		idx = starts[mask2]
		mid = data[idx + 1].astype(np.uint32)
		cp[mask2] = (data[idx] & 0x7F) | (mid << 7)
	mask3 = lengths == 3
	if mask3.any():
		idx = starts[mask3]
		mid = data[idx + 1].astype(np.uint32)
		high = data[idx + 2].astype(np.uint32)
		cp[mask3] = ((data[idx] & 0x7F) | ((mid & 0x7F) << 7) | (high << 14))
	return u32_to_str(cp)


def encode(b: str | bytes | bytearray | memoryview | np.ndarray) -> str:
	"""Encode bytes or text into invisicode's invisible glyph sequence.

	``str`` input is first converted with :func:`l128_encode` and marked with
	:data:`STRINGPREFIX`, so that :func:`decode` can restore the original type.
	NumPy arrays of any dtype are reinterpreted as their raw little-endian bytes.
	"""
	if isinstance(b, str):
		was_string = True
		data = as_u8(l128_encode(b))
	else:
		was_string = False
		data = as_u8(b)

	excess = data.size % 3
	if excess:
		body, end = data[:data.size - excess], data[data.size - excess:]
	else:
		body, end = data, data[:0]

	# Lay the prefix, body and suffix into one buffer so the result is built by a
	# single decode rather than by repeated string concatenation.
	prefix_len = 1 if was_string else 0
	# One trailing byte costs one code point; two cost two plus a padding marker.
	suffix_len = (0, 1, 3)[excess]
	cp = np.empty(prefix_len + body.size // 3 * 2 + suffix_len, dtype=np.uint32)
	if was_string:
		cp[0] = STRINGPREFIX

	a = body.reshape((body.size // 3, 3))
	c = np.pad(a, ((0, 0), (0, 1)), constant_values=0).view(np.uint32).ravel()
	y, x = c >> 12, c & (RANGE - 1)
	x |= BASE
	y |= BASE
	body_end = prefix_len + x.size * 2
	cp[prefix_len:body_end:2] = x
	cp[prefix_len + 1:body_end:2] = y

	if excess == 1:
		cp[body_end] = int(end[0]) | BASE
	elif excess == 2:
		cp[body_end] = int(end[0]) | BASE
		cp[body_end + 1] = int(end[1]) | BASE
		cp[body_end + 2] = PADDING
	return u32_to_str(cp)

@overload
def decode(s: str | np.ndarray, expect: type[str], strict: bool = True) -> str: ...
@overload
def decode(s: str | np.ndarray, expect: type[bytes], strict: bool = True) -> bytes: ...
@overload
def decode(s: str | np.ndarray, expect: type | None = None, strict: bool = True) -> bytes | str: ...
def decode(s: str | np.ndarray, expect: type | None = None, strict: bool = True) -> bytes | str:
	"""Decode an invisicode glyph sequence into bytes or text.

	:param expect: If ``str`` or ``bytes``, raise :class:`InvisicodeDecodeError`
		when the payload's recorded type does not match. Leave as ``None`` to
		accept whichever type was encoded.
	:param strict: When ``True``, any code point outside the invisicode range
		raises :class:`InvisicodeDecodeError`. When ``False``, surrounding and
		interleaved foreign characters are stripped and decoding proceeds on
		whatever remains; note that this can therefore return truncated or
		meaningless output instead of reporting an error.
	"""
	buf = as_u32(s)
	if not strict:
		while buf.size and not is_invisicode_codepoint(buf[0]):
			buf = buf[1:]
		# Trim from the end, but never consume a leading string prefix: on its own
		# it is the valid encoding of an empty string, and it is not payload.
		floor = 1 if buf.size and int(buf[0]) in STRINGPREFIXES else 0
		while buf.size > floor and not is_invisicode_codepoint(buf[-1], allow_prefixes=False):
			buf = buf[:-1]
	if buf.size and int(buf[0]) in STRINGPREFIXES:
		if expect is bytes:
			raise InvisicodeDecodeError("A string encoding was detected where a bytes output was expected.")
		was_string = True
		buf = buf[1:]
	else:
		if expect is str:
			raise InvisicodeDecodeError("A bytes encoding was detected where a string output was expected.")
		was_string = False

	invalid = (buf < BASE) | (buf >= BASE + RANGE)
	if invalid.any():
		if strict:
			raise InvisicodeDecodeError(f"Unexpected character {chr(buf[invalid][0])}")
		buf = buf[np.logical_not(invalid, out=invalid)]

	if buf.size & 1:
		if buf.size >= 3 and buf[-1] == PADDING:
			first, second = int(buf[-2]) - BASE, int(buf[-3]) - BASE
			if first > 0xFF or second > 0xFF:
				raise InvisicodeDecodeError("Malformed two-byte trailing group")
			suffix = bytes((second, first))
			buf = buf[:-3]
		else:
			# A lone trailing byte can only ever have been widened from 0x00..0xFF.
			last = int(buf[-1]) - BASE
			if last > 0xFF:
				raise InvisicodeDecodeError("Malformed single-byte trailing group")
			suffix = bytes((last,))
			buf = buf[:-1]
	else:
		suffix = b""

	b4096 = buf - BASE
	x, y = b4096[::2], b4096[1::2]
	y <<= 12
	c = y | x
	# Drop the unused high byte of each uint32 straight into the output buffer,
	# leaving room for the suffix so the result needs no further concatenation.
	body_size = c.size * 3
	out = np.empty(body_size + len(suffix), dtype=np.uint8)
	out[:body_size].reshape((c.size, 3))[:] = c.view(np.uint8).reshape((c.size, 4))[:, :-1]
	if suffix:
		out[body_size:] = np.frombuffer(suffix, dtype=np.uint8)
	b = out.tobytes()

	if was_string:
		return l128_decode(b)
	return b


def _is_prefix(buf: np.ndarray) -> np.ndarray:
	"""Return a boolean mask of the positions in ``buf`` holding a string prefix."""
	mask = np.zeros(buf.shape, dtype=bool)
	for prefix in STRINGPREFIXES:
		mask |= buf == prefix
	return mask

def is_invisicode_codepoint(c: int, allow_prefixes: bool = True) -> bool:
	"""Return whether a code point belongs to the invisicode range or allowed prefixes."""
	if allow_prefixes and int(c) in STRINGPREFIXES:
		return True
	return BASE <= c < BASE + RANGE
def is_invisicode(s: str | np.ndarray, strict: bool = True) -> bool:
	"""Return whether a string or array holds invisicode content.

	:param strict: When ``True``, every code point (after an optional string
		prefix) must lie in the invisicode range, and empty input is rejected.
		When ``False``, return ``True`` if *any* code point is invisicode, and
		accept empty input.
	"""
	buf = as_u32(s)
	if not buf.size:
		return not strict
	if strict and int(buf[0]) in STRINGPREFIXES:
		buf = buf[1:]
		if not buf.size:
			# A bare prefix is the valid encoding of an empty string.
			return True
	invalid = (buf < BASE) | (buf >= BASE + RANGE)
	if strict:
		return not bool(invalid.any())
	return not bool(invalid.all())

def detect(s: str | np.ndarray) -> np.ndarray:
	"""Locate contiguous invisicode segments within the provided text.

	Returns an ``(N, 2)`` array of ``[start, end)`` half-open index pairs, ready
	to use as slice bounds. Where a segment is immediately preceded by a string
	prefix, ``start`` is extended backwards by one to include it, so that
	:func:`decode` can recover the payload's original type. A string prefix with
	no payload after it is reported as a zero-length segment covering just the
	prefix, since that is the valid encoding of an empty string.
	"""
	buf = as_u32(s)
	invalid = (buf < BASE) | (buf >= BASE + RANGE)
	valid = np.logical_not(invalid, out=invalid)
	padded_arr = np.concatenate([[False], valid, [False]])
	diff = np.diff(padded_arr.astype(np.int8))
	starts = np.flatnonzero(diff == 1)
	ends = np.flatnonzero(diff == -1)

	prefixed = _is_prefix(buf)
	# Shift all segments that start with a prefix
	has_prefix = np.zeros(starts.shape, dtype=bool)
	if starts.size:
		shiftable = starts != 0
		if shiftable.any():
			has_prefix[shiftable] = prefixed[starts[shiftable] - 1]
		starts[has_prefix] -= 1

	# Lone prefix denotes empty text payload
	lone = prefixed.copy()
	if lone.size:
		lone[:-1] &= np.logical_not(valid[1:])
	lone_idx = np.flatnonzero(lone)
	if lone_idx.size:
		starts = np.concatenate([starts, lone_idx])
		ends = np.concatenate([ends, lone_idx + 1])
		order = np.argsort(starts, kind="stable")
		starts, ends = starts[order], ends[order]
	return np.stack([starts, ends]).swapaxes(0, 1)
def detect_and_decode(s: str | np.ndarray, expect: type | None = None) -> list[bytes | str]:
	"""Detect all invisicode substrings in the input and decode each one."""
	buf = as_u32(s)
	ranges = detect(buf)
	out = []
	for start, end in ranges:
		out.append(decode(buf[start:end], expect=expect))
	return out