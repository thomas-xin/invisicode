import numpy as np

BASE = 0xe0000
RANGE = 0x1000
STRINGPREFIX = 0x1d17a
PADDING = BASE + RANGE - 1


class InvisicodeEncodeError(ValueError):
	pass
class InvisicodeDecodeError(ValueError):
	pass


def u32_to_str(arr: np.ndarray) -> str:
	enc = "utf-32-le"
	return np.asanyarray(arr, dtype=np.uint32).tobytes().decode(enc)
def str_to_u32(s: str) -> np.ndarray:
	enc = "utf-32-le"
	return np.frombuffer(s.encode(enc), dtype=np.uint32)


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
def decode_leb128(data: bytes) -> tuple[int, bytes]:
	"Decodes an integer from LEB128 encoded data; returns a tuple of decoded and remaining data."
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


def l128_encode(s: str) -> bytes:
	if not s:
		return b""
	cp = str_to_u32(s)
	if cp.size == 0:
		return b""

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
	return out.tobytes()

def l128_decode(b: bytes) -> str:
	if not b:
		return ""
	data = np.frombuffer(b, dtype=np.uint8)
	if not data.size:
		return ""

	termination_mask = (data & 0x80) == 0
	if not termination_mask[-1]:
		raise UnicodeDecodeError("invisicode", b, len(b) - 1, len(b), "Incomplete LEB128 sequence")
	ends = np.flatnonzero(termination_mask)
	if data.size < 2 ** 32:
		ends = ends.astype(np.uint32)
	starts = ends.copy()
	starts[0] = 0
	if starts.size > 1:
		starts[1:] = ends[:-1] + 1
	lengths = ends - starts + 1
	if np.any((lengths < 1) | (lengths > 3)):
		raise UnicodeDecodeError("invisicode", b, 0, len(b), "Invalid LEB128 codepoint length")

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

	if np.any(cp >= 0x110000):
		raise UnicodeDecodeError("invisicode", b, 0, len(b), "Character out of range")
	return u32_to_str(cp)


def encode(b: str | bytes | bytearray | memoryview | np.ndarray) -> str:
	if isinstance(b, str):
		was_string = True
		b = l128_encode(b)
	else:
		was_string = False
	try:
		b = memoryview(b)
	except TypeError:
		b = memoryview(b.tobytes() if hasattr(b, "tobytes") else bytes(b))

	excess = len(b) % 3
	if excess:
		b, end = b[:-excess], b[-excess:]
	if excess == 1:
		suffix = chr(end[0] | BASE)
	elif excess == 2:
		suffix = chr(end[0] | BASE) + chr(end[1] | BASE) + chr(PADDING)
	else:
		suffix = ""

	a = np.frombuffer(b, dtype=np.uint8).reshape((len(b) // 3, 3))
	c = np.pad(a, ((0, 0), (0, 1)), constant_values=0).view(np.uint32).ravel()
	y, x = c >> 12, c & (RANGE - 1)
	x |= BASE
	y |= BASE
	ids = np.empty(x.size * 2, dtype=x.dtype)
	ids[::2] = x
	ids[1::2] = y
	s = u32_to_str(ids)

	if suffix:
		s += suffix
	if was_string:
		return chr(STRINGPREFIX) + s
	return s

def decode(s: str, expect: type = None) -> bytes | str:
	buf = str_to_u32(s)
	if buf.size and buf[0] == STRINGPREFIX:
		if expect is bytes:
			raise InvisicodeDecodeError("A string encoding was detected where a bytes output was expected.")
		was_string = True
		buf = buf[1:]
	else:
		if expect is str:
			raise InvisicodeDecodeError("A bytes encoding was detected where a string output was expected.")
		was_string = False
	while buf.size and (buf[-1] < BASE or buf[-1] >= BASE + RANGE):
		buf = buf[:-1]

	if buf.size & 1:
		if buf.size >= 3 and buf[-1] == PADDING:
			first, second = buf[-2], buf[-3]
			suffix = bytes([second - BASE, first - BASE])
			buf = buf[:-3]
		else:
			suffix = bytes([buf[-1] - BASE])
			buf = buf[:-1]
	else:
		suffix = b""

	invalid = (buf < BASE) | (buf >= BASE + RANGE)
	if np.any(invalid):
		raise InvisicodeDecodeError(f"Unexpected character {chr(buf[invalid][0])}")
	ins = buf - BASE
	x, y = ins[::2], ins[1::2]
	y <<= 12
	c = y | x
	a = c.view(np.uint8).reshape((c.size, 4))[:, :-1].ravel()
	b = a.tobytes()

	if suffix:
		b += suffix
	if was_string:
		return l128_decode(b)
	return b


def is_invisicode(s, strict=True):
	if not s:
		return not strict
	chars = list(s)
	if ord(chars[0]) == STRINGPREFIX:
		chars.pop(0)
	if strict:
		for c in chars:
			if not (BASE <= ord(c) < BASE + RANGE):
				return False
		return True
	for c in chars:
		if not (BASE <= ord(c) < BASE + RANGE):
			return False
		return True
	return True