import threading
import time

from flow360.component.case import CaseList
from flow360 import VolumeMesh, Flow360MeshParams, ProgressCallbackInterface
from flow360.component.case import CaseDownloadable

from testcases import OM6test

OM6test.get_files()


my_cases = CaseList()
case1 = my_cases[0].to_case()
case2 = my_cases[1].to_case()

print(case1)
print(case2)


class ProgressCallback(ProgressCallbackInterface):
    def __init__(self, name):
        self.total = 0
        self.bytes_transferred = 0
        self.name = name

    def total(self, total: int):
        self.total = total

    def __call__(self, bytes_chunk_transferred):
        self.bytes_transferred += bytes_chunk_transferred
        print(f"progress {self.name}: {self.bytes_transferred / self.total * 100} %")


def thread_download1():
    thread = threading.Thread(
        target=case1.results.download_file,
        args=[CaseDownloadable.VOLUME],
        kwargs={"progress_callback": ProgressCallback("volume case1")},
    )
    thread.start()
    return thread


def thread_download2():
    thread = threading.Thread(
        target=case2.results.download_file,
        args=[CaseDownloadable.VOLUME],
        kwargs={"progress_callback": ProgressCallback("volume case2")},
    )
    thread.start()
    return thread


def thread_upload():
    meshParams = Flow360MeshParams.from_file(OM6test.mesh_json)
    thread = threading.Thread(
        target=VolumeMesh.from_file,
        args=[OM6test.mesh_filename, meshParams, "OM6wing-mesh"],
        kwargs={"progress_callback": ProgressCallback("mesh upload")},
    )
    thread.start()
    return thread


t1 = thread_download1()
t2 = thread_download2()
t3 = thread_upload()


for _ in range(10):
    print("This thread continues while upload/download progress")
    time.sleep(1)


# wait for thread to finish, this is not necessary if your main thread doesn't stop
t1.join()
t2.join()
t3.join()
