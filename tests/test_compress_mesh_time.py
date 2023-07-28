import bz2
import gzip
import io
import lzma
import os
import sys
import tempfile
import time
from shutil import copyfileobj, rmtree

import zstandard as zstd

import flow360 as fl
from flow360.component.utils import zstd_compress
from flow360.component.volume_mesh import CompressionFormat


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
        f_out.write(f_in.read())

    input_file_size = print_file_sizes(input_file, output_file_path)
    return output_file_path, input_file_size


def compress_file_lzma(input_file, output_file_path=None):
    if output_file_path is None:
        output_file = tempfile.NamedTemporaryFile(delete=False)
        output_file_path = output_file.name + ".xz"
    with open(input_file, "rb") as f_in, lzma.open(output_file_path, "wb") as f_out:
        f_out.write(f_in.read())

    input_file_size = print_file_sizes(input_file, output_file_path)
    return output_file_path, input_file_size


def decompress_if_needed(file_path: str):
    # Check if the file is compressed with bz2
    if file_path.endswith(".bz2"):
        print("decompress bz2")
        start = time.time()
        with open(file_path, "rb") as file:
            content = bz2.decompress(file.read())
            end = time.time()
            print(f"Decompression with bz2 took {end - start} second")
            return io.BytesIO(content)

    # Check if the file is compressed with zstandard
    if file_path.endswith(".zst"):
        print("decompress zst")
        start = time.time()
        with open(file_path, "rb") as file:
            dctx = zstd.ZstdDecompressor()
            decompressor = dctx.decompressobj()
            content = decompressor.decompress(file.read())
            end = time.time()
            print(f"Decompression with zst took {end - start} second")
            return io.BytesIO(content)

    # Return the file content as is if it's not compressed
    return open(file_path, "rb")


def compare_ugrid_files(file_path1: str, file_path2: str) -> bool:
    print(f"Comparing {file_path1} and {file_path2}")
    with decompress_if_needed(file_path1) as file1, decompress_if_needed(file_path2) as file2:
        if file1.read() == file2.read():
            print("Equal")
        else:
            print("NOT equal")


input_file_path = os.path.join(os.getcwd(), "tests/upload_test_files/CRMHL_Wingbody_7v.cgns")
# output_file, tempfile = zstd_compress(input_file_path)
# compare_ugrid_files(
#     output_file,
#     input_file_path,
# )
# rmtree(tempfile)

# print("start bz2")
# start = time.time()

# compressed_file_path, input_file_size = compress_file_bz2(
#     input_file_path, output_file_path=f"{input_file_path}.bz2"
# )
# end = time.time()
# print(
#     f"compress with bz2 took: {end - start} seconds, {input_file_size/(1024**2)/(end - start)}MB/s"
# )


# print("start upload")
# vm = fl.VolumeMesh.from_file(input_file_path, name="test-upload-compressed-file")
# # vm.compress_method = CompressionFormat.BZ2
# vm.compress_method = CompressionFormat.ZST
# print("finish init")
# start = time.time()
# vm.submit()
# end = time.time()
# print(f"upload took: {end - start} seconds, {4143.68/(end - start)}MB/s")
# print(
#     compare_ugrid_files(
#         "/Users/linjin/Downloads/accae1ec-9650-4f1d-9a1c-1f4fa80a639b_mesh.lb8.ugrid.zst",
#         input_file_path,
#     )
# )
