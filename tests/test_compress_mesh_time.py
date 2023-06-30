import sys, os
import flow360 as fl
import time
import gzip
import pgzip
import zlib
import pigz
import tempfile
from shutil import copyfileobj
import bz2

fl.Env.dev.active()
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(here, "..")))


def print_file_sizes(input_file, output_file_path):
    input_file_size = os.path.getsize(input_file)
    output_file_size = os.path.getsize(output_file_path)

    print("File size before compression:", input_file_size, "bytes")
    print("File size after compression:", output_file_size, "bytes")
    return input_file_size


def compress_file_bz2(input_file, output_file_path=None):
    if output_file_path is None:
        output_file = tempfile.NamedTemporaryFile(delete=False)
        output_file_path = output_file.name + ".cgns.bz2"

    with open(input_file, "rb") as f_in:
        with bz2.BZ2File(output_file_path, "wb", compresslevel=9) as f_out:
            copyfileobj(f_in, f_out)

    input_file_size = print_file_sizes(input_file, output_file_path)
    return output_file_path, input_file_size


def compress_file_gzip(input_file):
    output_file = tempfile.NamedTemporaryFile(delete=False)
    output_file_path = output_file.name + ".gz"
    with open(input_file, "rb") as f_in, gzip.open(output_file_path, "wb") as f_out:
        f_out.write(f_in)

    input_file_size = print_file_sizes(input_file, output_file_path)
    return output_file_path, input_file_size


def compress_file_pgzip(input_file):
    output_file = tempfile.NamedTemporaryFile(delete=False)
    output_file_path = output_file.name + ".gz"
    with open(input_file, "rb") as f_in, pgzip.open(
        output_file_path, "wb", blocksize=2 * 10**8
    ) as f_out:
        f_out.write(f_in.read())
        # copyfileobj(f_in, f_out)

    input_file_size = print_file_sizes(input_file, output_file_path)
    return output_file_path, input_file_size


def compress_file_zlib(input_file):
    output_file = tempfile.NamedTemporaryFile(delete=False)
    output_file_path = output_file.name + ".gz"

    with open(input_file, "rb") as f_in, open(output_file_path, "wb") as f_out:
        compressor = zlib.compressobj(9, zlib.DEFLATED, zlib.MAX_WBITS | 16)
        for chunk in iter(lambda: f_in.read(1024), b""):
            compressed_chunk = compressor.compress(chunk)
            f_out.write(compressed_chunk)
        compressed_final_chunk = compressor.flush()
        f_out.write(compressed_final_chunk)

    input_file_size = print_file_sizes(input_file, output_file_path)
    return output_file_path, input_file_size


input_file_path = "/Users/linjin/Desktop/Flow360/tests/upload_test_files/wing_tetra.8M.lb8.ugrid"
# input_file_path = AirplaneTest.meshFilePath


# start = time.time()

# compressed_file_path, input_file_size = compress_file_bz2(
#     input_file_path, output_file_path=f"{input_file_path}.bz2"
# )
# print(f"Compressed file saved to: {compressed_file_path}")

# end = time.time()
# print(
#     f"compress with bz2 took: {end - start} seconds, {input_file_size/(1024**2)/(end - start)}MB/s"
# )


# start = time.time()

# compressed_file_path, input_file_size = compress_file_pgzip(input_file_path)
# print(f"Compressed file saved to: {compressed_file_path}")

# end = time.time()
# print(f"compress with pgzip took: {end - start}, {input_file_size/(1024**2)/(end - start)}MB/s")
# os.remove(compressed_file_path)


# start = time.time()

# compressed_file_path, input_file_size = compress_file_zlib(input_file_path)
# print(f"Compressed file saved to: {compressed_file_path}")

# end = time.time()
# print(f"compress with zlib took: {end - start}, {input_file_size/(1024**2)/(end - start)}MB/s")
# os.remove(compressed_file_path)


# start = time.time()

# compressed_file_path, input_file_size = compress_file_zlib(input_file_path)
# print(f"Compressed file saved to: {compressed_file_path}")

# end = time.time()
# print(f"compress with zlib took: {end - start}, {input_file_size/(1024**2)/(end - start)}MB/s")
# os.remove(compressed_file_path)


start = time.time()

vm = fl.VolumeMesh.from_file(input_file_path, name="test-upload-compressed-file").submit()


end = time.time()
print(f"upload took: {end - start} seconds")

# os.remove(compressed_file_path)
