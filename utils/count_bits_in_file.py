import os


def count_random_bits(filename):
    try:
        with open(filename, "rb") as file:
            data = file.read()
        bit_count = len(data) * 8  # Each byte has 8 bits
        print(f"Total number of random bits in '{filename}': {bit_count}")
        return bit_count
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


filename = "randomness/ANU_13Oct2017_100MB_2"
count_random_bits(filename)


# print("Current working directory:", os.getcwd())
