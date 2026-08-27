# invisicode

Encodes arbitrary data into strings that display invisibly on devices and platforms supporting unicode.

Operates in base 4096, carrying 1.5 bytes of payload per code point, meaning every 3 input bytes become 2 code points, so the output length is on average ⅔ of input length. In practice the efficiency is marginally lower due to one or two padding characters required in specific cases, but this is negligible as the input size increases.

Originally a coding scheme designed for [Miza](https://github.com/thomas-xin/Miza), as one of the methods to hide small amounts of persistent data in text messages to represent instructions for future edits to the message, while remaining visually undisruptive to users.

Capable of encoding both byte-strings and text-strings, and distinguishing between the two, using an additional signifier character (0x1d17a).

Note: NOT designed for [prompt-injecting LLMs](https://embracethered.com/blog/posts/2024/m365-copilot-prompt-injection-tool-invocation-and-data-exfil-using-ascii-smuggling/).

### Functionality
The payload alphabet is the 4096 code points `U+E0000..U+E0FFF`. The entire range is `Default_Ignorable_Code_Point` in the Unicode standard — the assigned [Tags](https://en.wikipedia.org/wiki/Tags_%28Unicode_block%29) (`E0000..E007F`) and [Variation Selectors Supplement](https://en.wikipedia.org/wiki/Variation_Selectors_Supplement) (`E0100..E01EF`) characters directly, and the *unassigned* remainder (`E0080..E00FF`, `E01F0..E0FFF`) via `Other_Default_Ignorable_Code_Point`, which reserves them so that they remain ignorable if they are ever assigned. Conformant renderers are therefore expected to display none of it, rather than substituting `.notdef` boxes.

In practice every modern web browser and social media platform tested renders the whole range as zero-width. Some older software that predates or ignores the default-ignorable property (notably Microsoft Office) will show `[?]` boxes for the unassigned portions. The [demo page](https://thomas-xin.github.io/invisicode) is the quickest way to check a given target.

No code point used by the format is altered by Unicode normalisation: all 4096 payload code points and the string prefix are stable under NFC, NFD, NFKC and NFKD, so payloads survive platforms that normalise text.

## Demo
See https://thomas-xin.github.io/invisicode for an interactive demo of the encoding! You may use this to verify whether various examples of encoded text or data correctly render invisibly on your device or platform.

## Installation
`pip install invisicode`

## Usage
### Core API
```python
encode(b: str | bytes | bytearray | memoryview | numpy.ndarray) -> str
    # Encode bytes or text into invisicode's invisible glyph sequence. NumPy arrays of
    # any dtype are reinterpreted as their raw little-endian bytes; non-contiguous and
    # multi-dimensional arrays are flattened in C order.
decode(s: str | numpy.ndarray, expect: type = None, strict: bool = True) -> bytes | str
    # Decode an invisicode glyph sequence into bytes or text.
    #   expect: if str or bytes, raise InvisicodeDecodeError when the payload's recorded
    #           type does not match. None accepts whichever type was encoded.
    #   strict: True raises InvisicodeDecodeError on any code point outside the
    #           invisicode range. False strips surrounding and interleaved foreign
    #           characters and decodes whatever remains, which may therefore return
    #           truncated or meaningless output instead of reporting an error.
is_invisicode(s: str | numpy.ndarray, strict: bool = True) -> bool
    # Whether the input holds invisicode content. In strict mode every code point (after
    # an optional string prefix) must be in range, and empty input is rejected; in
    # non-strict mode returns True if any code point is invisicode, and accepts empty input.
detect(s: str | numpy.ndarray) -> numpy.ndarray
    # Locate contiguous invisicode segments, as an (N, 2) array of [start, end) half-open
    # index pairs ready to use as slice bounds. Where a segment is immediately preceded by
    # a string prefix, start is extended back by one to include it.
detect_and_decode(s: str | numpy.ndarray, expect: type = None) -> list[bytes | str]
    # Detect all invisicode substrings in the input and decode each one, preserving types.
```
### Base-128 text coding
```python
l128_encode(s: str) -> memoryview
    # Encode a text string using variable-length base-128 encoding.
l128_decode(b: bytes | bytearray | memoryview) -> str
    # Decode bytes produced by l128_encode back into a Unicode string.
leb128(n: int) -> bytearray
    # Reference scalar encoder for a single integer. Signs negative values with a trailing
    # 00 byte, keeping compatibility with standard LEB128 (unlike SLEB128).
decode_leb128(data: bytes) -> tuple[int, bytes]
    # Reference scalar decoder; returns the value and the remaining data.
```
### Helpers
```python
is_invisicode_codepoint(c: int, allow_prefixes: bool = True) -> bool
    # Whether a single code point is in the invisicode range, or is a string prefix.
u32_to_str(arr: numpy.ndarray) -> str      # UTF-32 code points -> str
str_to_u32(s: str) -> numpy.ndarray        # str -> UTF-32 code points
as_u32(s: str | numpy.ndarray) -> numpy.ndarray   # UTF-32 view of either input type
as_u8(b) -> numpy.ndarray                  # flat C-contiguous uint8 view of any bytes-like
```
### Constants and exceptions
```python
BASE = 0xe0000          # First code point of the payload alphabet
RANGE = 0x1000          # Size of the alphabet (4096)
PADDING = 0xe0fff       # Marks a two-byte trailing group
STRINGPREFIX = 0x1d17a  # Emitted to mark a payload as text
STRINGPREFIXES          # frozenset of every prefix accepted when decoding

InvisicodeError(ValueError)                 # Base class for all errors below
├── InvisicodeEncodeError                   # Input cannot be encoded
└── InvisicodeDecodeError                   # Input cannot be decoded
    └── InvisicodeUnicodeDecodeError        # Also a UnicodeDecodeError; malformed text payload
```
### Examples
- Encoding and decoding regular binary data
```python
import invisicode
data = b"Hello World!"
encoded = invisicode.encode(data) # '\U000e0548\U000e06c6\U000e0f6c\U000e0206\U000e0f57\U000e0726\U000e046c\U000e0216'
assert invisicode.decode(encoded) == data # b"Hello World!"
```
- Encoding and decoding a regular string
```python
import invisicode
data = "Hello World! ❤️"
encoded = invisicode.encode(data) # '\U0001d17a\U000e0548\U000e06c6\U000e0f6c\U000e0206\U000e0f57\U000e0726\U000e046c\U000e0216\U000e0420\U000e04ee\U000e0c8f\U000e003f'
assert invisicode.decode(encoded) == data # 'Hello World! ❤️'
```
- Encoding and decoding a (relatively) large amount of binary data
```python
import invisicode
import numpy as np
data = np.random.randint(0, 256, size=10 ** 8, dtype=np.uint8)
encoded = invisicode.encode(data) # '\U000e05b7\U000e0504\U000e02cc\U000e09a9\U000e0df5\U000e0066\U000e0d96󠅋\U000e0959\U000e0469...
len(data), len(encoded) # (100000000, 66666667)
assert invisicode.decode(encoded) == data.tobytes()
```
- Invisicode exposes LEB128 encodings for strings, which is also internally used for marginal coding efficiency improvements over UTF-8 (as we are reencoding the information anyway, the redundancy/error checking normally provided by UTF-8 is of no use to us).
```python
import invisicode
bytes(invisicode.l128_encode("test")) # b'test'
bytes(invisicode.l128_encode("Hello World! ❤️")) # b'Hello World! \xe4N\x8f\xfc\x03'; 18 bytes vs 19 for utf-8
assert invisicode.l128_decode(invisicode.l128_encode("驈ꍬ啯ꍲᕤ")) == "驈ꍬ啯ꍲᕤ"
```
- Extracting payloads embedded in ordinary text
```python
import invisicode
text = "Hello!" + invisicode.encode("hidden note") + " How are you?" + invisicode.encode(b"\x01\x02")
invisicode.detect(text) # array([[ 6, 16], [29, 32]])
invisicode.detect_and_decode(text) # ['hidden note', b'\x01\x02']
# Types are preserved per-segment; pass expect= to require one or the other.
invisicode.detect_and_decode(text, expect=str) # raises InvisicodeDecodeError on the bytes segment
```
- Recovering a payload from text that has been damaged or has stray characters
```python
import invisicode
encoded = invisicode.encode(b"test")
damaged = "junk" + encoded[0] + "\u200b" + encoded[1:] + "junk"
invisicode.decode(damaged, strict=False) # b'test'
invisicode.decode(damaged) # raises InvisicodeDecodeError
```

## Protocol
- Note: All numbers are encoded as little-endian bytes where applicable.

The encoding is performed as follows:
- If the input is a string, encode it as leb128 representation (slightly more space-efficient than utf-8), and start with a string prefix character 0x1d17a (MUSICAL SYMBOL END PHRASE);  a `Cf` format character, default-ignorable, and outside the normal invisicode range. Decoders accept any code point in `STRINGPREFIXES` as this marker, so that the emitted prefix can change in future without invalidating existing payloads.
- Each group of 3 bytes from the input is converted to two base-4096 numbers, by reinterpreting as a base-16777216 number and then splitting.
- 0xE0000 is added to each resulting number, placing it in the [Tags](https://en.wikipedia.org/wiki/Tags_%28Unicode_block%29) block and the reserved default-ignorable range that follows it, which renders as non-printable, non-breaking space (see [above](#why-these-code-points-render-invisibly)).
- If there is a single trailing byte (length % 3 == 1), it is encoded by itself by adding 0xE0000.
- If there are two trailing bytes (length % 3 == 2), they are encoded similarly, but with a padding character 0xE0FFF appended at the end. This enables the string to still contain an odd amount of characters and stay within invisicode's normal range, while being distinct from the (length % 3 == 1) case.

The decoding is performed as follows:
- If the string begins with a recognised string prefix character, remove that and flag the content as string.
- If there are an odd number of characters, there are trailing bytes present. Attempt to detect the padding character to determine whether one or two bytes should be extracted. A trailing group whose value exceeds 0xFF cannot have come from a single byte, and is rejected.
- 0xE0000 is subtracted from remaining characters. In strict mode, any character outside the range raises `InvisicodeDecodeError`; in non-strict mode such characters are stripped instead.
- The results are interpreted as base-16777216 numbers, split into three base-256 numbers each, and reinterpreted as bytes.
- If necessary, convert the result back to a string.

## Development
- Note: The entire test suite under /tests, as well as the paragraph below, are AI-generated from the spec, and may be subject to change.
```
pip install -e ".[test]"
pytest                      # fast run, suitable for an edit/test loop
pytest -m "not slow"        # skip the megabyte-scale cases
```
The suite is property-based, using [Hypothesis](https://hypothesis.readthedocs.io/). It is designed so that the vectorised implementation can be rewritten or optimised freely: tests assert observable behaviour and format invariants, never internal steps.

| File | Purpose |
| --- | --- |
| `tests/reference.py` | A deliberately naive, NumPy-free implementation of the format, used as an oracle. It is written to be *obviously correct* rather than fast. |
| `tests/strategies.py` | Shared Hypothesis strategies. A format change (wider alphabet, new prefix) should only need editing here. |
| `tests/test_properties.py` | Round-trip properties, differential tests against the oracle, format invariants, strictness, detection, and malformed input. |
| `tests/test_units.py` | Example-based tests pinning the exact contract of each helper. |
| `tests/test_documentation.py` | Re-asserts every literal result quoted in this README, including the sample payload below, so the docs cannot drift. |

Three Hypothesis profiles are registered in `tests/conftest.py`:
```
pytest                                  # dev: 50 examples per property
pytest --hypothesis-profile=ci          # ci: 300 examples
pytest --hypothesis-profile=thorough    # 5000 examples, for after a format or performance change
```
If a differential test fails, exactly one of two things is true: the optimised implementation has a bug, or the format was changed intentionally and `tests/reference.py` must be updated to match — which is itself a useful signal that the wire format moved.

The suite maintains 100% statement coverage of `invisicode.py`, enforced in CI:
```
pytest --cov=invisicode --cov-report=term-missing --cov-fail-under=100
```

### Known format limitations
These are inherent to embedding a self-delimiting format in arbitrary text, and are covered by explicit tests so they cannot regress silently:
- An empty *bytes* payload encodes to the empty string, so it cannot be located by `detect`. An empty *text* payload still leaves its prefix, and remains detectable.
- If the carrier text happens to contain a prefix code point immediately before a bytes payload, `detect` will read that payload as text. Use `decode` on a known slice where the boundaries matter.
- The padding code point `0xE0FFF` is an ordinary payload value and can occur mid-string. It is unambiguous regardless, because a body always contributes an even number of code points, so it is only interpreted as padding when the total length is odd. However, this may lead to ambiguities when decoding concatenated payloads.

## Sample
The text between the characters "X" and "Y" below may be decoded as invisicode. It contains 2173 invisible characters, and represents 3258 bytes of leb128-data that may then be further decoded into 2568 unicode characters. For comparison, UTF-8 would encode the same text as 3635 bytes.

X𝅺󠕗󠦖󠉀󠙗󠸠󠛶󠌠󠻊󠀇󠜲󠉴󠘗󠝮󠙖󠍲󠈇󠫵󠁾󠲏󠀿󠂍󠰤󠽌󠿈󠀃󠝂󠁯󠛂󠙯󠙗󠴠󠻈󠰇󠂢󠽙󠝖󠤠󠺌󠐇󠺎󠀇󠚲󠽮󠝶󠴠󠺚󠀇󠝂󠕨󠈆󠕲󠛇󠍥󠈇󠶷󠁾󠄠󠛦󠁤󠜲󠁯󠙂󠁯󠒒󠄠󠺌󠸇󠂢󠁁󠙢󠱵󠛆󠔠󠹋󠀇󠘲󠵯󠛖󠑩󠛗󠹥󠝆󠂙󠜴󠜠󠚇󠑡󠈇󠥉󠐉󠁭󠝂󠥨󠛦󠥫󠛦󠁧󠫒󠟩󠈀󠙯󠋆󠤊󠛵󠁵󠾢󠟨󠲀󠟨󠮐󠁂󠝲󠕯󠛇󠹤󠝆󠤠󠻊󠀇󠙲󠑥󠈇󠪟󠁾󠐠󠚇󠍩󠈇󠉦󠛷󠁭󠘒󠥮󠈇󠑯󠚇󠉥󠈇󠕧󠞗󠘠󠺎󠼇󠹿󠜇󠺎󠰇󠹿󠐇󠼽󠼇󠹿󠸇󠂢󠤊󠈄󠣁󠁾󠨠󠝖󠑳󠈇󠅷󠛦󠅮󠈆󠳈󠁾󠣀󠁾󠐠󠙗󠱬󠈆󠦬󠁾󠤠󠛷󠁵󠲂󠟨󠈀󠽨󠝶󠤠󠦔󠵀󠈆󠕦󠙖󠥬󠛦󠁧󠠒󠟬󠋀󠜊󠛴󠑴󠘗󠤠󠺌󠀇󠛒󠭡󠙖󠔠󠺹󠀇󠞒󠕯󠈇󠣉󠁾󠔠󠛧󠕤󠜦󠑳󠘗󠑮󠩦󠩀󠂠󠕎󠝦󠉥󠈇󠻌󠈄󠽧󠛦󠅮󠈆󠪥󠁾󠜠󠚖󠕶󠈆󠞁󠁾󠤠󠛷󠁵󠲂󠟨󠈀󠁵󠈇󠲝󠋄󠸊󠙔󠕶󠜦󠰠󠓬󠜠󠛶󠹮󠘖󠔠󠺪󠀇󠛂󠑥󠈇󠣨󠁾󠂍󠺔󠟨󠣐󠝀󠺎󠴇󠐈󠣧󠁾󠳆󠁾󠦂󠁾󠤠󠛷󠁵󠲂󠟨󠈀󠽤󠝶󠁮󠡲󠱖󠂢󠕎󠝦󠉥󠈇󠶫󠁾󠜠󠛶󠹮󠘖󠔠󠺪󠀇󠜢󠹵󠈆󠟃󠁾󠦨󠁾󠄠󠜦󠕯󠛧󠁤󠼒󠟳󠭠󠟧󠭐󠟧󠈀󠹡󠙆󠐠󠙖󠕳󠜦󠁴󠪲󠟨󠈀󠽹󠝖󠠠󠺌󠸇󠂢󠕎󠝦󠉥󠈇󠷑󠁾󠻌󠈄󠽧󠛦󠅮󠈆󠪥󠁾󠴠󠘖󠕫󠈆󠮕󠁾󠤠󠛷󠁵󠬂󠟬󠲀󠟨󠈀󠉣󠞗󠴠󠻊󠰇󠂢󠕎󠝦󠉥󠈇󠳅󠁾󠻌󠈄󠽧󠛦󠅮󠈆󠪥󠁾󠌠󠘗󠁹󠫂󠟩󠈀󠽧󠛶󠉤󠞖󠁥󠲲󠟨󠋀󠸊󠙔󠕶󠜦󠰠󠓬󠜠󠛶󠹮󠘖󠤠󠺌󠀇󠝂󠱥󠛆󠌠󠺾󠀇󠘒󠰠󠚖󠁥󠩂󠟬󠈀󠹡󠙆󠠠󠝖󠑲󠈇󠊕󠁿󠤠󠛷󠁵󠲂󠟨󠋠󠨊󠕰󠥥󠐉󠕶󠈆󠹫󠛶󠹷󠈆󠞓󠁾󠔠󠘖󠡣󠈆󠑯󠚇󠉥󠈇󠽦󠜦󠌠󠛷󠰠󠛶󠝮󠈆󠫚󠁾󠤊󠛵󠉵󠈇󠣉󠁾󠠠󠙖󠉡󠝇󠂙󠜴󠈠󠙖󠹥󠈆󠍡󠚆󠹩󠙶󠀠󠻌󠨇󠐠󠑵󠈇󠛑󠁾󠤠󠛷󠥵󠐉󠕲󠈆󠽴󠛶󠌠󠚇󠁹󠥢󠟬󠪰󠟬󠬰󠟬󠈀󠽴󠈆󠅳󠞖󠌠󠺾󠀇󠚒󠹴󠂢󠹉󠜶󠑩󠙖󠀠󠺚󠀇󠝲󠁥󠘢󠑯󠚇󠬠󠛦󠝯󠈇󠊔󠁿󠜠󠚇󠑡󠦗󠍀󠈇󠕢󠙖󠁮󠙲󠥯󠛦󠁧󠺲󠟭󠰰󠟧󠾰󠟧󠪀󠟩󠈀󠹯󠈆󠪛󠁾󠨬󠕰󠁥󠚲󠽮󠝶󠐠󠼩󠀇󠝂󠕨󠈆󠅧󠛖󠁥󠫢󠟧󠈀󠹡󠙆󠜠󠙗󠂙󠜤󠁥󠙲󠹯󠛦󠁡󠩒󠟪󠈀󠱰󠘖󠁹󠩲󠟧󠫠󠟧󠈀󠑩󠋧󠨊󠐐󠹮󠛦󠹮󠙆󠤠󠙦󠤠󠛷󠁵󠲂󠟨󠈀󠍡󠚷󠼠󠻌󠀇󠛒󠁥󠚂󠝯󠈇󠥉󠐉󠁭󠙢󠕥󠛆󠹩󠙶󠄠󠻈󠰇󠂢󠽄󠛦󠂙󠝄󠐠󠙗󠱬󠈆󠯣󠁾󠴠󠙖󠤠󠛷󠥵󠐉󠕲󠈆󠽴󠛶󠈠󠛆󠹩󠙆󠠠󠻌󠀇󠝂󠁯󠜲󠕥󠩦󠩀󠂠󠕎󠝦󠉥󠈇󠶫󠁾󠜠󠛶󠹮󠘖󠤠󠺌󠀇󠙲󠙩󠙗󠤠󠺌󠀇󠞒󠕯󠈇󠣈󠁾󠔠󠜇󠴠󠓉󠨬󠓠󠙥󠙗󠁲󠳂󠁎󠙲󠹯󠛦󠁡󠩒󠟪󠈀󠕬󠝆󠘠󠻌󠀇󠞒󠕯󠈇󠣈󠁾󠐠󠛶󠹷󠈆󠚇󠋅󠸊󠙔󠕶󠜦󠔠󠻌󠀇󠙲󠹯󠛦󠁡󠩒󠟪󠈀󠕲󠛧󠌠󠹼󠀇󠘒󠽲󠝖󠑮󠈆󠪃󠁾󠄠󠛦󠁤󠙂󠍥󠙗󠑲󠈇󠟜󠁾󠤠󠛷󠁵󠲂󠟨󠋠󠸊󠙔󠕶󠜦󠔠󠻌󠀇󠙲󠹯󠛦󠁡󠩒󠟪󠈀󠅭󠚶󠁥󠥒󠟫󠈀󠽹󠝖󠤠󠺌󠀇󠘲󠥲󠈇󠲭󠁾󠦦󠁾󠨬󠓠󠙥󠙗󠁲󠱒󠟬󠈀󠽧󠛦󠅮󠈆󠣉󠁾󠌠󠘗󠁹󠸲󠟫󠈀󠽧󠛶󠉤󠞖󠁥󠲲󠟨󠋀󠸊󠙔󠕶󠜦󠬠󠻚󠀇󠙲󠹯󠛦󠁡󠲒󠟨󠈀󠕴󠛆󠁬󠫂󠟩󠈀󠁡󠛂󠕩󠈆󠲤󠁾󠄠󠛦󠁤󠚂󠉵󠝇󠔠󠼩󠀇󠞒󠕯󠈇󠣈󠁾󠨮󠂠󠕎󠝦󠉥󠈇󠻌󠈄󠽧󠛦󠅮󠈆󠣉󠁾󠜠󠚖󠕶󠈆󠞁󠁾󠤠󠛷󠁵󠲂󠟨󠈀󠁵󠈇󠲝󠋄󠸊󠙔󠕶󠜦󠔠󠻌󠀇󠙲󠹯󠛦󠁡󠲒󠟨󠈀󠕬󠝆󠘠󠻌󠀇󠞒󠕯󠈇󠣈󠁾󠐠󠛶󠹷󠈆󠚇󠋅󠸊󠙔󠕶󠜦󠰠󠓬󠜠󠛶󠹮󠘖󠔠󠺪󠀇󠜢󠹵󠈆󠟃󠁾󠟻󠁾󠂍󠰄󠽌󠿈󠀃󠘒󠽲󠝖󠑮󠈆󠪃󠁾󠄠󠛦󠁤󠙂󠍥󠙗󠑲󠈇󠢪󠁾󠤠󠛷󠁵󠲂󠟨󠋠󠸊󠙔󠕶󠜦󠰠󠓬󠜠󠛶󠹮󠘖󠔠󠺪󠀇󠛒󠭡󠙖󠠠󠺙󠀇󠞒󠕯󠈇󠣈󠁾󠌠󠜦󠁹󠫒󠟬󠋀󠸊󠙔󠕶󠜦󠄠󠻊󠔇󠻌󠀇󠙲󠹯󠛦󠁡󠩒󠟪󠈀󠅳󠞖󠄠󠺏󠼇󠹿󠰇󠺚󠠇󠻌󠀇󠙲󠽯󠙆󠥢󠙗󠬠󠺌󠰇󠂢󠕎󠝦󠉥󠈇󠻌󠈄󠽧󠛦󠅮󠈆󠣉󠁾󠶠󠈄󠕴󠛆󠁬󠸲󠟫󠴐󠟭󠈀󠁡󠛂󠕩󠈆󠲤󠁾󠄠󠛦󠁤󠚂󠉵󠝇󠔠󠼩󠀇󠞒󠕯󠈇󠣈󠁾󠨮󠂠󠥇󠝦󠁥󠲒󠟨󠈀󠽹󠝖󠠠󠺌󠀇󠝒󠁰󠦒󠟣󠡠󠹖󠈂󠥧󠝦󠁥󠠒󠟧󠈀󠽹󠝖󠤠󠺌󠬇󠹿󠀇󠝒󠁰󠧒󠹌󠂢󠥇󠝦󠁥󠢢󠅎󠹸󠀇󠞒󠕯󠈇󠚝󠁾󠣈󠁾󠔠󠜇󠘠󠕨󠀬󠙲󠙩󠙗󠄠󠹸󠀇󠞒󠕯󠈇󠣈󠁾󠔠󠜇󠴠󠓉󠨮󠓠󠙥󠙗󠁲󠱒󠟬󠾰󠟧󠣐󠁀󠓌󠲏󠀿󠜠󠛶󠹮󠘖󠔠󠺪󠀇󠙲󠙩󠙗󠤠󠺌󠨇󠓠󠙥󠙗󠁲󠳂󠩎󠻈󠀇󠙲󠹯󠛦󠁡󠩒󠟪󠈀󠥧󠝦󠁥󠲒󠟨󠋀󠜠󠚖󠕶󠈆󠞁󠁾󠤠󠛷󠁵󠲂󠟨󠈀󠁵󠈇󠚆󠋥󠸊󠙔󠕶󠜦󠰠󠓬󠜠󠛶󠹮󠘖󠔠󠺪󠀇󠙲󠙩󠙗󠤠󠺌󠨇󠓠󠙥󠙗󠁲󠱒󠟬󠮀󠟧󠈀󠽧󠛦󠅮󠈆󠪥󠁾󠜠󠚖󠕶󠈆󠣉󠁾󠀬󠙲󠙩󠙗󠄠󠹸󠀇󠞒󠕯󠈇󠣈󠁾󠔠󠜇󠴠󠓉󠨮󠂠󠕗󠦖󠙀󠙗󠬠󠛦󠝯󠛧󠌠󠹹󠐇󠓩󠔠󠘖󠡣󠈆󠑯󠚇󠉥󠈇󠽦󠜦󠌠󠛷󠰠󠛶󠝮󠈆󠫚󠁾󠤊󠛵󠉵󠈇󠣈󠁾󠠠󠙖󠉡󠝇󠂙󠜴󠈠󠙖󠹥󠈆󠍡󠚆󠹩󠙶󠬠󠻊󠐇󠻉󠨇󠐠󠑵󠈇󠛑󠁾󠤠󠛷󠥵󠐉󠕲󠈆󠽴󠛶󠌠󠚇󠁹󠢢󠟬󠈀󠽴󠈆󠅳󠞖󠌠󠺾󠀇󠚒󠹴󠂢󠹉󠜶󠑩󠙖󠀠󠺚󠀇󠝲󠁥󠘢󠑯󠚇󠬠󠛦󠝯󠈇󠊔󠁿󠜠󠚇󠑡󠦗󠍀󠈇󠕢󠙖󠁮󠙲󠥯󠛦󠁧󠲒󠟨󠿀󠟧󠈀󠹯󠈆󠪛󠁾󠨬󠕰󠁥󠚲󠽮󠝶󠐠󠼩󠀇󠝂󠕨󠈆󠅧󠛖󠁥󠬒󠟧󠈀󠹡󠙆󠜠󠙗󠂙󠜤󠁥󠙲󠹯󠛦󠁡󠩒󠟪󠈀󠱰󠘖󠁹󠫢󠟧󠈀󠑩󠋧󠨊󠒐󠄠󠺌󠀇󠚢󠍵󠝇󠜠󠘗󠹮󠘖󠠠󠻌󠀇󠝂󠱥󠛆󠌠󠺾󠀇󠞒󠕯󠈇󠣈󠁾󠠠󠛶󠁷󠒒󠂙󠛔󠘠󠙖󠱥󠚖󠝮󠈆󠲁󠁾󠨬󠑰󠑯󠝇󠁡󠪲󠟪󠣀󠱎󠹿󠀇󠛒󠭡󠙖󠠠󠺙󠀇󠞒󠕯󠈇󠲉󠁾󠣉󠁾󠣌󠁾󠔠󠛧󠕤󠜦󠑳󠘗󠑮󠩦󠩀󠂠󠕎󠝦󠉥󠈇󠳅󠁾󠜠󠛶󠹮󠘖󠔠󠺪󠀇󠙲󠙩󠙗󠄠󠹸󠀇󠞒󠕯󠈇󠣈󠁾󠔠󠜇󠴠󠓉󠨬󠓠󠙥󠙗󠁲󠪲󠟭󠨐󠟩󠈀󠽧󠛦󠅮󠈆󠪥󠁾󠰠󠙖󠁴󠱢󠟬󠈀󠽹󠝖󠠠󠺌󠀇󠙂󠝯󠛧󠜠󠕨󠨬󠓠󠙥󠙗󠁲󠱒󠟬󠈀󠽧󠛦󠅮󠈆󠪥󠁾󠈠󠝗󠁮󠰲󠟧󠈀󠉡󠛷󠹵󠙆󠌠󠺨󠀇󠘒󠑮󠈆󠕤󠜶󠉥󠝇󠨠󠺊󠀇󠞒󠕯󠈇󠣈󠁾󠟽󠁾󠨮󠓠󠙥󠙗󠁲󠨒󠟬󠱐󠟬󠈀󠽧󠛦󠅮󠈆󠪥󠁾󠴠󠘖󠕫󠈆󠦘󠁾󠤠󠛷󠁵󠫢󠟬󠲀󠟨󠿠󠟧󠧐󠟲󠈀󠉣󠞗󠈠󠻈󠰇󠂢󠕎󠝦󠉥󠈇󠳅󠁾󠜠󠛶󠹮󠘖󠐠󠻊󠔇󠺪󠀇󠜲󠥡󠈇󠯣󠁾󠻓󠥄󠟲󠈀󠽧󠛶󠉤󠞖󠁥󠲲󠟨󠋀󠸊󠙔󠕶󠜦󠔠󠻌󠀇󠙲󠹯󠛦󠁡󠩒󠟪󠈀󠕴󠛆󠁬󠸲󠟫󠈀󠁡󠛂󠕩󠈆󠊥󠁿󠛳󠁾󠣡󠁾󠄠󠛦󠁤󠚂󠉵󠝇󠔠󠼩󠀇󠞒󠕯󠈇󠣈󠁾󠨮󠂠󠕎󠝦󠉥󠈇󠶫󠁾󠲥󠁾󠜠󠛶󠹮󠘖󠔠󠺪󠀇󠙲󠙩󠙗󠄠󠹸󠀇󠞒󠕯󠈇󠣈󠁾󠔠󠜇󠴠󠓉󠨬󠓠󠙥󠙗󠁲󠩂󠟬󠳀󠁎󠙲󠹯󠛦󠁡󠲒󠟨󠈀󠕬󠝆󠬠󠺎󠀇󠞒󠕯󠈇󠣈󠁾󠐠󠛶󠹷󠈆󠚇󠋅󠸊󠙔󠕶󠜦󠬠󠻚󠀇󠙲󠹯󠛦󠁡󠲒󠟨󠈀󠕲󠛧󠌠󠹼󠀇󠘒󠽲󠝖󠑮󠈆󠏱󠁿󠞶󠁾󠞵󠁾󠄠󠛦󠁤󠙂󠍥󠙗󠑲󠈇󠢪󠁾󠤠󠛷󠁵󠲒󠟨󠋠󠸊󠙔󠕶󠜦󠔠󠻌󠀇󠙲󠹯󠛦󠁡󠩒󠟪󠈀󠅭󠚶󠁥󠦂󠟩󠈀󠽹󠝖󠠠󠺌󠴇󠹿󠀇󠘲󠥲󠈇󠲭󠁾󠨬󠓠󠙥󠙗󠁲󠪲󠟭󠈀󠽧󠛦󠅮󠈆󠪥󠁾󠌠󠘗󠁹󠫂󠟩󠈀󠽧󠛶󠉤󠞖󠁥󠲲󠟨󠋀󠸊󠙔󠕶󠜦󠔠󠻌󠀇󠙲󠹯󠛦󠁡󠩒󠟪󠈀󠕴󠛆󠁬󠸲󠟫󠈀󠁡󠛂󠕩󠈆󠲤󠁾󠄠󠛦󠁤󠚂󠉵󠝇󠔠󠼩󠀇󠞒󠕯󠈇󠣈󠁾󠨮󠂠󠕎󠝦󠉥󠈇󠻌󠈄󠽧󠛦󠅮󠈆󠪥󠁾󠜠󠚖󠕶󠈆󠞁󠁾󠤠󠛷󠁵󠲂󠟨󠈀󠁵󠈇󠲝󠋄󠸊󠙔󠕶󠜦󠰠󠓬󠪚󠁾󠜠󠛶󠹮󠘖󠔠󠺪󠀇󠛂󠑥󠈇󠳆󠁾󠤠󠛷󠁵󠲂󠟨󠈀󠽤󠝶󠁮󠡲󠱖󠂢󠕎󠝦󠉥󠈇󠳅󠁾󠜠󠛶󠹮󠘖󠔠󠺪󠀇󠜢󠹵󠈆󠟃󠁾󠄠󠜦󠕯󠛧󠁤󠠲󠟪󠈀󠹡󠙆󠐠󠙖󠕳󠜦󠁴󠪲󠟨󠈀󠽹󠝖󠠠󠺌󠸇󠂢󠕎󠝦󠉥󠈇󠻌󠈄󠽧󠛦󠅮󠈆󠣉󠁾󠴠󠘖󠕫󠈆󠮕󠁾󠤠󠛷󠁵󠲒󠟨󠈀󠉣󠞗󠔠󠻊󠴇󠻊󠰇󠂢󠕎󠝦󠉥󠈇󠻌󠈄󠽧󠛦󠅮󠈆󠪥󠁾󠌠󠘗󠁹󠸲󠟫󠈀󠽧󠛶󠉤󠞖󠁥󠲲󠟨󠋀󠸊󠙔󠕶󠜦󠔠󠻌󠰇󠓬󠜠󠛶󠹮󠘖󠔠󠺪󠀇󠝂󠱥󠛆󠰠󠺚󠀇󠘒󠰠󠚖󠁥󠩂󠟬󠈀󠹡󠙆󠠠󠝖󠑲󠈇󠊕󠁿󠤠󠛷󠁵󠲒󠟨󠋠Y