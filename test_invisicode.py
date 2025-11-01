import os
import numpy as np
from invisicode import encode, decode, PADDING

class TestEncodeDecode:
	def test_decode_empty_string(self):
		result = decode("")
		assert result == b""

	def test_decode_single_byte(self):
		encoded = encode(b"A")
		result = decode(encoded)
		assert result == b"A"

	def test_decode_two_bytes(self):
		encoded = encode(b"AB")
		result = decode(encoded)
		assert result == b"AB"

	def test_decode_three_bytes(self):
		encoded = encode(b"ABC")
		result = decode(encoded)
		assert result == b"ABC"

	def test_decode_multiple_three_byte_blocks(self):
		encoded = encode(b"ABCDEF")
		result = decode(encoded)
		assert result == b"ABCDEF"

	def test_decode_with_padding_one_byte(self):
		encoded = encode(b"ABCD")
		result = decode(encoded)
		assert result == b"ABCD"

	def test_decode_with_padding_two_bytes(self):
		encoded = encode(b"ABCDE")
		result = decode(encoded)
		assert result == b"ABCDE"

	def test_decode_utf8_string(self):
		original = "Hello, World! 🌍"
		encoded = encode(original)
		result = decode(encoded)
		assert result == original.encode("utf-8")

	def test_decode_binary_data(self):
		original = bytes(range(256))
		encoded = encode(original)
		result = decode(encoded)
		assert result == original

	def test_decode_roundtrip_various_lengths(self):
		for length in range(20):
			original = bytes(range(length))
			encoded = encode(original)
			result = decode(encoded)
			assert result == original, f"Failed for length {length}"

	def test_decode_null_bytes(self):
		original = b"\x00\x00\x00"
		encoded = encode(original)
		result = decode(encoded)
		assert result == original

	def test_decode_high_byte_values(self):
		original = b"\xff\xfe\xfd"
		encoded = encode(original)
		result = decode(encoded)
		assert result == original

	def test_decode_large_data(self):
		original = np.full(10**8, 232, dtype=np.uint8)
		encoded = encode(original)
		result = decode(encoded)
		assert result == original.tobytes()

	def test_decode_large_random_data(self):
		original = np.random.randint(0, 256, size=10**8, dtype=np.uint8)
		encoded = encode(original)
		result = decode(encoded)
		assert result == original.tobytes()


if __name__ == "__main__":

	def roundtrip(payload: bytes) -> None:
		s = encode(payload)
		back = decode(s)
		assert back == bytes(payload)

	# Edge lengths
	for n in range(0, 10):
		roundtrip(os.urandom(n))

	# Larger randoms
	for n in (32, 33, 1024, 1025):
		roundtrip(os.urandom(n))

	# String input
	roundtrip("hello 🌍".encode('utf-8'))

	# NumPy input
	roundtrip(np.frombuffer(b'\x00\x01\x02\x03\xff', dtype=np.uint8))

	# Malformed input: wrong range in body
	try:
		# Create a string with a non-PUA codepoint in the body position
		bad = chr(0x41) + encode(b'abc')[1:]  # 'A' in first position
		decode(bad)
		raise AssertionError('Expected ValueError for malformed input')
	except ValueError:
		pass