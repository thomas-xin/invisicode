"""A deliberately naive, pure-Python reference implementation of the invisicode format.

This module exists to be *obviously correct* rather than fast. It uses no NumPy
and no vectorisation: every code point is handled one at a time with plain
integer arithmetic, mirroring the prose in the README's Protocol section as
directly as possible.

The differential tests compare the real implementation against this oracle. That
gives the vectorised code freedom to be rewritten or optimised, while still
proving byte-for-byte behavioural equivalence. If a future change makes a test
here fail, exactly one of two things is true:

* the optimised implementation has a bug, or
* the format itself was intentionally changed, and this file must be updated to
  match (which is a useful, explicit signal that the wire format moved).

Keep this file free of imports from :mod:`invisicode` other than the format
constants, so that a bug in the module under test cannot mask itself.
"""

from __future__ import annotations

BASE = 0xe0000
RANGE = 0x1000
PADDING = BASE + RANGE - 1
STRINGPREFIX = 0x1d17a


def ref_leb128_encode_int(n: int) -> bytes:
	"""Encode a single non-negative integer as unsigned LEB128."""
	if n == 0:
		return b"\x00"
	out = bytearray()
	while n > 0:
		byte = n & 0x7F
		n >>= 7
		if n:
			byte |= 0x80
		out.append(byte)
	return bytes(out)


def ref_l128_encode(s: str) -> bytes:
	"""Encode text by concatenating the LEB128 form of each code point."""
	return b"".join(ref_leb128_encode_int(ord(ch)) for ch in s)


def ref_l128_decode(b: bytes) -> str:
	"""Decode a concatenation of LEB128 code points back into text."""
	out = []
	n = 0
	shift = 0
	started = False
	for byte in b:
		started = True
		n |= (byte & 0x7F) << shift
		if byte & 0x80:
			shift += 7
			continue
		out.append(chr(n))
		n = 0
		shift = 0
		started = False
	if started:
		raise ValueError("Incomplete LEB128 sequence")
	return "".join(out)


def ref_encode(data: bytes | str) -> str:
	"""Encode bytes or text into invisicode, one group of three bytes at a time."""
	if isinstance(data, str):
		prefix = chr(STRINGPREFIX)
		payload = ref_l128_encode(data)
	else:
		prefix = ""
		payload = bytes(data)

	out = []
	whole = len(payload) - len(payload) % 3
	for i in range(0, whole, 3):
		# Little-endian base-16777216 number from three bytes.
		n = payload[i] | (payload[i + 1] << 8) | (payload[i + 2] << 16)
		low = n & (RANGE - 1)
		high = n >> 12
		out.append(chr(BASE + low))
		out.append(chr(BASE + high))

	rest = payload[whole:]
	if len(rest) == 1:
		out.append(chr(BASE + rest[0]))
	elif len(rest) == 2:
		out.append(chr(BASE + rest[0]))
		out.append(chr(BASE + rest[1]))
		out.append(chr(PADDING))
	return prefix + "".join(out)


def ref_decode(s: str) -> bytes | str:
	"""Decode an invisicode sequence, rejecting anything malformed."""
	cps = [ord(ch) for ch in s]
	was_string = False
	if cps and cps[0] == STRINGPREFIX:
		was_string = True
		cps = cps[1:]

	for cp in cps:
		if not BASE <= cp < BASE + RANGE:
			raise ValueError(f"Unexpected character {cp:#x}")

	suffix = b""
	if len(cps) % 2:
		if len(cps) >= 3 and cps[-1] == PADDING:
			first = cps[-2] - BASE
			second = cps[-3] - BASE
			if first > 0xFF or second > 0xFF:
				raise ValueError("Malformed two-byte trailing group")
			suffix = bytes((second, first))
			cps = cps[:-3]
		else:
			last = cps[-1] - BASE
			if last > 0xFF:
				raise ValueError("Malformed single-byte trailing group")
			suffix = bytes((last,))
			cps = cps[:-1]

	out = bytearray()
	for i in range(0, len(cps), 2):
		low = cps[i] - BASE
		high = cps[i + 1] - BASE
		n = low | (high << 12)
		out.append(n & 0xFF)
		out.append((n >> 8) & 0xFF)
		out.append((n >> 16) & 0xFF)
	out += suffix

	if was_string:
		return ref_l128_decode(bytes(out))
	return bytes(out)
