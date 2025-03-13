import numpy as np
import struct


FILE_COUNT = 50
FILE_SIZE_BYTES = 104_857_600  # 100MB per file
CHUNK_SIZE_BYTES = 64 * 1024  # 64KB
CHUNKS_PER_FILE = FILE_SIZE_BYTES // CHUNK_SIZE_BYTES
TOTAL_CHUNKS = FILE_COUNT * CHUNKS_PER_FILE

FILE_PATH = "randomness/ANU_13Oct2017_100MB_"


def read_chunk(idx):
    """Reads a 64 KB chunk from the correct file based on the given index.
    Max index is 80000. This index points to a 64KB block of random bits in
    the directory randomness/. Each chunk is used to generate one trial of the experiment.

    Each file has 838,860,800 bits (104,857,600 bytes). So there are exactly 1600 chunks of 64KB per file.
    """
    if idx < 0 or idx >= TOTAL_CHUNKS:
        raise ValueError(f"Index {idx} is out of range (0 to {TOTAL_CHUNKS - 1}).")

    # Determine which file contains the requested chunk
    file_idx = idx // CHUNKS_PER_FILE  # File number (0-based)
    chunk_offset = idx % CHUNKS_PER_FILE  # Chunk position within the file

    # Compute the byte offset inside the file
    byte_offset = chunk_offset * CHUNK_SIZE_BYTES

    # Construct filename (1-based numbering)
    filename = f"{FILE_PATH}{file_idx + 1}"

    # Read the required chunk from the file
    with open(filename, "rb") as file:
        file.seek(byte_offset)  # Move to the correct position
        chunk_data = file.read(CHUNK_SIZE_BYTES)  # Read 64 KB

    return chunk_data  # Returns the 64KB chunk as a bytes object


class TrueRandom:
    """Random number generator using our own true source of randomness."""

    def __init__(self, chunk):
        """
        Initialize with a fixed 64KB chunk of randomness.
        :param chunk: bytes object of length exactly 64KB (65536 bytes).
        """
        assert isinstance(chunk, bytes), "chunk must be a bytes object"
        assert len(chunk) == 64 * 1024, "chunk must be 64KB"

        self.chunk = chunk
        self.chunk_pos = 0  # Tracks how many bytes have been used
        self.bit_buffer = None  # Stores current byte for bit extraction
        self.bit_pos = 8  # Tracks how many bits are left in current byte

    def _consume_bytes(self, num_bytes):
        """Consume bytes from chunk and return as a bytes object."""
        if self.chunk_pos + num_bytes > len(self.chunk):
            raise RuntimeError("Out of randomness! No more bytes available.")

        data = self.chunk[self.chunk_pos : self.chunk_pos + num_bytes]
        self.chunk_pos += num_bytes
        return data

    def _random_uniform_0_1(self):
        """Generate a uniform random number in [0,1) using 4 bytes from chunk."""
        raw_bytes = self._consume_bytes(4)
        uint_value = struct.unpack(">I", raw_bytes)[0]  # Convert to 32-bit unsigned int
        return uint_value / 2**32  # Normalize to [0,1)

    def random(self):
        """Generate a uniform random number in [0,1)"""
        return self._random_uniform_0_1()

    def normal(self, size):
        """
        Generate standard normal (mean=0, std=1) values using Box-Muller transform.
        :param size: Tuple specifying the shape of the output array.
        :return: np.ndarray of normally distributed values.
        """
        total_values = np.prod(size)  # Total numbers required
        # Total number of bytes consumed:
        # (total_values + 1) // 2 iterations of generating 2 normals each
        # Generating each normal consumes 4 bytes
        if ((total_values + 1) // 2) * 2 * 4 > len(self.chunk) - self.chunk_pos:
            raise RuntimeError(
                "Not enough randomness to generate requested normal values."
            )

        # Generate pairs of uniform values and convert to normal
        normals = []
        for _ in range((total_values + 1) // 2):  # Generate in pairs
            u1, u2 = self._random_uniform_0_1(), self._random_uniform_0_1()
            r = np.sqrt(-2.0 * np.log(u1))
            theta = 2.0 * np.pi * u2
            z1, z2 = r * np.cos(theta), r * np.sin(theta)
            normals.extend([z1, z2])

        return np.array(normals[:total_values]).reshape(size)

    def _random_bit(self):
        """Extract a single bit from the chunk."""
        if self.bit_pos == 8:
            # Load a new byte when all bits are consumed
            self.bit_buffer = self._consume_bytes(1)[0]  # Get a new byte
            self.bit_pos = 0  # Reset bit position

        # Extract the bit from the current byte
        bit = (self.bit_buffer >> (7 - self.bit_pos)) & 1
        self.bit_pos += 1
        return bit

    def randint(self, high, size, dtype=bool):
        """
        Mimics np.random.randint(2, size=(rows, cols), dtype=bool).
        This generates a boolean array where each element is either True (1) or False (0).
        """
        if high != 2 or dtype != bool:
            raise ValueError(
                "Only np.random.randint(2, size=..., dtype=bool) is supported."
            )

        total_values = np.prod(size)
        if total_values > (len(self.chunk) - self.chunk_pos) * 8:
            raise RuntimeError(
                "Not enough randomness to generate requested boolean values."
            )

        # Generate boolean values directly from bits
        bool_array = np.array(
            [self._random_bit() for _ in range(total_values)], dtype=bool
        )
        return bool_array.reshape(size)


# A few ad hoc tests i ran...
# idx = 5000
# random_bits = read_chunk(idx)
# print(f"Read {len(random_bits)} bytes from chunk index {idx}.")
# rand = TrueRandom(random_bits)

# n = 2
# random_values = rand.normal(size=[2] * n)
# print("Normal values:\n", random_values)

# rows, cols = 4, 4
# bool_matrix = rand.randint(2, size=(rows, cols), dtype=bool)
# print("Boolean matrix:\n", bool_matrix)

# for _ in range(10):
#     print(rand.random())
