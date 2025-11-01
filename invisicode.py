import numpy as np

BASE = 0xe0000
RANGE = 0x1000
PADDING = 0x1d17a

class InvisicodeEncodeError(ValueError):
	pass
class InvisicodeDecodeError(ValueError):
	pass

def u32_to_str(arr: np.ndarray) -> str:
	enc = 'utf-32-le'
	return arr.tobytes().decode(enc)

def str_to_u32(s: str) -> np.ndarray:
	enc = 'utf-32-le'
	return np.frombuffer(s.encode(enc), dtype=np.uint32)

def encode(b: str | bytes | bytearray | memoryview | np.ndarray, padding: int = PADDING) -> str:
	if isinstance(b, str):
		b = b.encode("utf-8")
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
		suffix = chr(end[0] | BASE) + chr(padding) + chr(end[1] | BASE)
	else:
		suffix = ""

	a = np.frombuffer(b, dtype=np.uint8).reshape((len(b) // 3, 3))
	c = np.pad(a, ((0, 0), (0, 1)), constant_values=0).view(np.uint32).ravel()
	y, x = c >> 12, c & (RANGE - 1)
	x |= BASE
	y |= BASE
	ids = np.empty(len(x) * 2, dtype=x.dtype)
	ids[::2] = x
	ids[1::2] = y
	s = u32_to_str(ids)

	if suffix:
		s += suffix
	return s

def decode(s: str, padding: int = PADDING) -> bytes:
	buf = str_to_u32(s)
	while len(buf) and (buf[-1] < BASE or buf[-1] > BASE + RANGE):
		buf = buf[:-1]

	if len(buf) & 1:
		if len(buf) >= 3 and buf[-2] == padding:
			first, second = buf[-1], buf[-3]
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
	a = c.view(np.uint8).reshape((len(c), 4))[:, :-1].ravel()
	b = a.tobytes()

	if suffix:
		b += suffix
	return b