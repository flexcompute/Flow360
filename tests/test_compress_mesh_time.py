import bz2
import gzip
import lzma
import os
import subprocess
import sys
import tempfile
import time
import zipfile
import zlib
import zstandard as zstd
from shutil import copyfileobj
from flow360.component.volume_mesh import CompressMethod
import py7zr

import flow360 as fl

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


def compress_file_zipfile(input_file, output_file_path=None):
    if output_file_path is None:
        output_file = tempfile.NamedTemporaryFile(delete=False)
        output_file_path = output_file.name + ".zip"
    with zipfile.ZipFile(output_file_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(input_file_path, arcname=input_file_path)

    input_file_size = print_file_sizes(input_file, output_file_path)
    return output_file_path, input_file_size


def compress_file_py7zr(input_file, output_file_path=None):
    if output_file_path is None:
        output_file = tempfile.NamedTemporaryFile(delete=False)
        output_file_path = output_file.name + ".7z"

    with py7zr.SevenZipFile(output_file_path, "w") as archive:
        archive.write(input_file_path)

    input_file_size = print_file_sizes(input_file, output_file_path)
    return output_file_path, input_file_size


def compress_file_gzip(input_file):
    output_file = tempfile.NamedTemporaryFile(delete=False)
    output_file_path = output_file.name + ".gz"
    with open(input_file, "rb") as f_in, gzip.open(output_file_path, "wb") as f_out:
        f_out.write(f_in.read())

    input_file_size = print_file_sizes(input_file, output_file_path)
    return output_file_path, input_file_size


def compress_file_pigz(input_file, output_file_path=None, num_threads=5):
    output_file = tempfile.NamedTemporaryFile(delete=False)
    if output_file_path is None:
        output_file = tempfile.NamedTemporaryFile(delete=False)
        output_file_path = output_file.name + ".gz"
        output_file.close()
    with open(output_file_path, "wb") as output_file:
        process = subprocess.run(
            ["pigz", "-8", "-p", str(num_threads), "-c", input_file_path], stdout=subprocess.PIPE
        )
        output_file.write(process.stdout)
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
            x = bz2.decompress(file.read())
            end = time.time()
            print(f"decompress with bz2 took {end - start} second")
            return x

    # Check if the file is compressed with zstandard
    if file_path.endswith(".zst"):
        print("decompress zst")
        start = time.time()
        with open(file_path, "rb") as file:
            dctx = zstd.ZstdDecompressor()
            x = dctx.decompress(file.read())
            end = time.time()
            print(f"decompress with zst took {end - start} second")
            return x

    # Return the file content as is if it's not compressed
    with open(file_path, "rb") as file:
        return file.read()


def compare_ugrid_files(file_path1, file_path2):
    content1 = decompress_if_needed(file_path1)
    with open(file_path2, "rb") as file2:
        content2 = file2.read()

    return content1 == content2


input_file_path = os.path.join(os.getcwd(), "tests/upload_test_files/wing_tetra.8M.lb8.ugrid")

# print("start py7zr")
# start = time.time()
# compressed_file_path, input_file_size = compress_file_py7zr(
#     input_file_path, output_file_path=f"{input_file_path}.7z"
# )
# end = time.time()
# print(
#     f"compress with py7zr took: {end - start} seconds, {input_file_size/(1024**2)/(end - start)}MB/s"
# )

# print("start zipfile")
# start = time.time()
# compressed_file_path, input_file_size = compress_file_zipfile(
#     input_file_path, output_file_path=f"{input_file_path}.zip"
# )
# end = time.time()
# print(
#     f"compress with zipfile took: {end - start} seconds, {input_file_size/(1024**2)/(end - start)}MB/s"
# )

# print("start pigz")
# start = time.time()
# compressed_file_path, input_file_size = compress_file_pigz(
#     input_file_path, output_file_path=f"{input_file_path}.gz"
# )
# end = time.time()
# print(
#     f"compress with zipfile took: {end - start} seconds, {input_file_size/(1024**2)/(end - start)}MB/s"
# )


# input_file_path = AirplaneTest.meshFilePath


# print("start bz2")
# start = time.time()

# compressed_file_path, input_file_size = compress_file_bz2(
#     input_file_path, output_file_path=f"{input_file_path}.bz2"
# )
# end = time.time()
# print(
#     f"compress with bz2 took: {end - start} seconds, {input_file_size/(1024**2)/(end - start)}MB/s"
# )

# print("start zlib")
# start = time.time()

# compressed_file_path, input_file_size = compress_file_zlib(input_file_path)
# end = time.time()
# print(f"compress with zlib took: {end - start}, {input_file_size/(1024**2)/(end - start)}MB/s")
# os.remove(compressed_file_path)


# print("start gzip")
# start = time.time()

# compressed_file_path, input_file_size = compress_file_gzip(input_file_path)
# end = time.time()
# print(f"compress with gzip took: {end - start}, {input_file_size/(1024**2)/(end - start)}MB/s")
# os.remove(compressed_file_path)

# print("start lzma")
# start = time.time()
# compressed_file_path, input_file_size = compress_file_lzma(
#     input_file_path, output_file_path=f"{input_file_path}.gz"
# )
# end = time.time()
# print(
#     f"compress with lzma took: {end - start} seconds, {input_file_size/(1024**2)/(end - start)}MB/s"
# )


# print("start upload")
# vm = fl.VolumeMesh.from_file(input_file_path, name="test-upload-compressed-file")
# # vm.compress_method = CompressMethod.BZ2
# vm.compress_method = CompressMethod.ZSTD
# print("finish init")
# start = time.time()
# vm.submit()
# end = time.time()
# print(f"upload took: {end - start} seconds")
# print(
#     compare_ugrid_files(
#         "/Users/linjin/Downloads/accae1ec-9650-4f1d-9a1c-1f4fa80a639b_mesh.lb8.ugrid.zst",
#         input_file_path,
#     )
# )
