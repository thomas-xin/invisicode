"""Tests that keep the README honest.

Documented examples drift silently as code changes. Each example the README shows
with a literal result is re-asserted here, so a behaviour change that invalidates
the docs fails CI instead of shipping.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

import invisicode
from invisicode import decode, detect, detect_and_decode, encode, l128_decode, l128_encode

README = Path(__file__).resolve().parent.parent / "README.md"


class TestReadmeExamples:
	"""Literal outputs quoted in the README."""

	def test_bytes_example(self):
		data = b"Hello World!"
		expected = (
			"\U000e0548\U000e06c6\U000e0f6c\U000e0206"
			"\U000e0f57\U000e0726\U000e046c\U000e0216"
		)
		assert encode(data) == expected
		assert decode(expected) == data

	def test_string_example(self):
		data = "Hello World! \u2764\ufe0f"
		expected = (
			"\U0001d17a\U000e0548\U000e06c6\U000e0f6c\U000e0206"
			"\U000e0f57\U000e0726\U000e046c\U000e0216\U000e0420"
			"\U000e04ee\U000e0c8f\U000e003f"
		)
		assert encode(data) == expected
		assert decode(expected) == data

	def test_l128_examples(self):
		assert bytes(l128_encode("test")) == b"test"
		assert bytes(l128_encode("Hello World! \u2764\ufe0f")) == b"Hello World! \xe4N\x8f\xfc\x03"
		assert len(l128_encode("Hello World! \u2764\ufe0f")) == 18
		assert len("Hello World! \u2764\ufe0f".encode("utf-8")) == 19
		assert l128_decode(l128_encode("\u9a88\ua36c\u556f\ua372\u1564")) == "\u9a88\ua36c\u556f\ua372\u1564"

	def test_detect_example(self):
		text = "Hello!" + encode("hidden note") + " How are you?" + encode(b"\x01\x02")
		np.testing.assert_array_equal(detect(text), np.array([[6, 16], [29, 32]]))
		assert detect_and_decode(text) == ["hidden note", b"\x01\x02"]

	def test_detect_example_expect_raises(self):
		text = "Hello!" + encode("hidden note") + " How are you?" + encode(b"\x01\x02")
		with pytest.raises(invisicode.InvisicodeDecodeError):
			detect_and_decode(text, expect=str)

	def test_damaged_payload_example(self):
		encoded = encode(b"test")
		damaged = "junk" + encoded[0] + "\u200b" + encoded[1:] + "junk"
		assert decode(damaged, strict=False) == b"test"
		with pytest.raises(invisicode.InvisicodeDecodeError):
			decode(damaged)

	def test_large_data_ratio_example(self):
		"""The README's 3-bytes-to-2-glyphs ratio, at a CI-friendly size."""
		size = 3 * 10**5
		rng = np.random.default_rng(0)
		data = rng.integers(0, 256, size=size, dtype=np.uint8)
		encoded = encode(data)
		assert len(encoded) == size // 3 * 2
		assert decode(encoded) == data.tobytes()


class TestReadmeSample:
	"""The long embedded sample payload in the README must stay decodable."""

	@staticmethod
	def _claim(text: str, pattern: str) -> int:
		"""Pull a single number out of the README's prose about the sample."""
		match = re.search(pattern, text)
		assert match, f"the README no longer states: {pattern}"
		return int(match.group(1))

	def test_sample_decodes(self):
		text = README.read_text(encoding="utf-8")
		match = re.search(r"\nX(.*?)Y", text, re.DOTALL)
		assert match, "the README sample block (between X and Y) was not found"
		sample = match.group(1)
		decoded = decode(sample)
		assert isinstance(decoded, str), "the sample is documented as a text payload"
		# The count includes the leading string prefix, which is also invisible.
		assert len(sample) == self._claim(text, r"contains (\d+) invisible characters")
		assert len(l128_encode(decoded)) == self._claim(text, r"represents (\d+) bytes")
		assert len(decoded) == self._claim(text, r"into (\d+) unicode characters")
		assert len(decoded.encode("utf-8")) == self._claim(
			text, r"UTF-8 would encode the same text as (\d+) bytes"
		)


class TestDocumentedApi:
	"""Every name the README documents must actually exist and be exported."""

	@pytest.mark.parametrize(
		"name",
		(
			"encode",
			"decode",
			"is_invisicode",
			"is_invisicode_codepoint",
			"detect",
			"detect_and_decode",
			"l128_encode",
			"l128_decode",
			"leb128",
			"decode_leb128",
			"u32_to_str",
			"str_to_u32",
			"as_u32",
			"as_u8",
			"BASE",
			"RANGE",
			"PADDING",
			"STRINGPREFIX",
			"STRINGPREFIXES",
			"InvisicodeError",
			"InvisicodeEncodeError",
			"InvisicodeDecodeError",
			"InvisicodeUnicodeDecodeError",
		),
	)
	def test_documented_name_exists(self, name: str):
		assert hasattr(invisicode, name)
		assert name in invisicode.__all__

	def test_every_public_name_is_documented(self):
		"""Guard against adding public API without updating the README."""
		readme = README.read_text(encoding="utf-8")
		undocumented = [name for name in invisicode.__all__ if name not in readme]
		assert undocumented == [], f"add these to the README: {undocumented}"

	def test_no_undeclared_public_names(self):
		"""Anything public but absent from ``__all__`` is probably an oversight.

		Imported modules and typing helpers are not part of the API surface.
		"""
		import types

		ignored = {"annotations", "overload"}
		public = {
			name
			for name, value in vars(invisicode).items()
			if not name.startswith("_")
			and name not in ignored
			and not isinstance(value, types.ModuleType)
		}
		assert public - set(invisicode.__all__) == set()
