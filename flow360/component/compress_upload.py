"""
Parallel Compress and Multiupload Flow360Resource to S3
"""
import bz2
import concurrent.futures
import os

from rich.progress import Progress

from flow360.component.resource_base import Flow360Resource


# pylint: disable=too-many-arguments, too-many-locals
def compress_and_upload_chunks(
    file_name: str,
    upload_id: str,
    remote_resource: Flow360Resource,
    remote_file_name: str,
    max_workers: int = 50,
    chunk_length: int = 25 * 1024 * 1024,
):
    """
    Compresses and uploads file chunks to a remote resource using Bzip2 compression.

    Args:
        file_name (str): The path to the input file that needs to be compressed
        and uploaded.
        upload_id (str): The ID of the multipart upload for the remote resource.
        remote_resource (Flow360Resource): The remote resource to which the chunks
        will be uploaded.
        remote_file_name (str): The name of the remote file on the remote resource.
        max_workers (int, optional): The maximum number of concurrent workers for
        the thread pool (default is 50).
        chunk_length (int, optional): The size (in bytes) of each chunk to be
        compressed and uploaded (default is 25 MB).

    Raises:
        AssertionError: If the input 'file_name' does not exist or is not a regular file.
    """

    def upload_and_update(part_number, chunk):
        future = executor.submit(
            remote_resource.upload_part,
            remote_file_name,
            upload_id,
            part_number,
            chunk,
        )
        # Wait for the upload to finish and update the progress
        result = future.result()
        progress.update(task_id1, advance=1)
        return result

    assert os.path.isfile(file_name)
    uploaded_parts = []  # Initialize an empty list to store the uploaded parts
    futures = []
    min_upload_size = 5 * 1024 * 1024

    with Progress() as progress:
        task_id = progress.add_task("[cyan]Compressing...", total=os.path.getsize(file_name))
        task_id1 = progress.add_task(
            "Uploading...", total=os.path.getsize(file_name) // chunk_length
        )
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        part_number = 1
        with open(file_name, "rb") as file:
            while True:
                chunk_data = file.read(chunk_length)
                progress.update(task_id, advance=chunk_length)
                if not chunk_data:
                    break
                compressed_chunk = bz2.compress(chunk_data)
                while len(compressed_chunk) < min_upload_size and chunk_data:
                    chunk_data = file.read(chunk_length)
                    compressed_chunk += bz2.compress(chunk_data)
                    progress.update(task_id, advance=chunk_length)
                # Call the upload function for each part without waiting for the result
                futures.append(executor.submit(upload_and_update, part_number, compressed_chunk))
                part_number += 1
        progress.update(task_id1, total=part_number - 1)
        concurrent.futures.wait(futures)
    uploaded_parts = [future.result() for future in futures]

    remote_resource.complete_multipart_upload(remote_file_name, upload_id, uploaded_parts)
