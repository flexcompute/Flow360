import asyncio
import threading


from flow360.component.case import CaseList
from flow360 import ProgressCallbackInterface
from flow360.component.case import CaseDownloadable

my_cases = CaseList()
case1 = my_cases[0].to_case()
case2 = my_cases[2].to_case()

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




async def download1():
    await case1.results.download_file_async(CaseDownloadable.VOLUME, progress_callback=ProgressCallback('volume case1'))

async def download2():
    await case1.results.download_file_async(CaseDownloadable.VOLUME, progress_callback=ProgressCallback('volume case2'))

async def other_function():
    for _ in range(5):
        print('this is concurrenctly running process')
        await asyncio.sleep(1)



async def test_async_download():
    await asyncio.gather(
        download1(),
        download2(),
        other_function())


asyncio.run(test_async_download())


def thread_download1():
    thread = threading.Thread(target=case1.results.download_file, args=[CaseDownloadable.VOLUME], kwargs={"progress_callback": ProgressCallback('volume case1')})
    thread.start()
    return thread

def thread_download2():
    thread = threading.Thread(target=case2.results.download_file, args=[CaseDownloadable.VOLUME], kwargs={"progress_callback": ProgressCallback('volume case2')})
    thread.start()
    return thread


t1 = thread_download1()
t2 = thread_download2()

# wait for thread to finish, this is not necessary if your main thread doesn't stop
t1.join()
t2.join()
