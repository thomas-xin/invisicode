# invisicode

Encodes arbitrary byte-strings into strings that display invisibly on devices and platforms supporting unicode.

Operates in base 4096 with one additional padding character for specific input widths, meaning 1.5 bytes per character on average.

Originally a coding scheme designed for github.com/thomas-xin/Miza, as one of the methods to hide small amounts of persistent data in text messages to reduce visual clutter.

Usage
```python
encode(
    b: str | bytes | bytearray | memoryview | numpy.ndarray,
    padding: int = 119162
) -> str
decode(s: str, padding: int = 119162) -> bytes
```
```python
import invisicode
data = b"Hello World!"
encoded = invisicode.encode(data) # '\U000e0548\U000e06c6\U000e0f6c\U000e0206\U000e0f57\U000e0726\U000e046c\U000e0216'
assert invisicode.decode(encoded) == data # b"Hello World!"
```
```python
import invisicode
data = "Hello World! ❤️"
encoded = invisicode.encode(data) # '\U000e0548\U000e06c6\U000e0f6c\U000e0206\U000e0f57\U000e0726\U000e046c\U000e0216\U000e0220\U000e09de\U000e0fa4\U000e0b8e\U000e008f'
assert invisicode.decode(encoded) == data.encode("utf-8") # b'Hello World! \xe2\x9d\xa4\xef\xb8\x8f'
```
```python
import invisicode
import numpy as np
data = np.random.randint(0, 256, size=10 ** 8, dtype=np.uint8)
encoded = invisicode.encode(data) # '\U000e05b7\U000e0504\U000e02cc\U000e09a9\U000e0df5\U000e0066\U000e0d96󠅋\U000e0959\U000e0469...
assert invisicode.decode(encoded) == data.tobytes()
```