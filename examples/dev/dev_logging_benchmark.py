import os
import datetime
import flow360

from flow360.log import log
from flow360.file_path import flow360_dir

N_LOGS = 10000

print("Started logging")
# get the start datetime
st = datetime.datetime.now()

for i in range(N_LOGS):
    log.debug(f"Sample log {i}")

# get the end datetime
et = datetime.datetime.now()
print("Finished logging")

exec_time = et - st

print(f'Log time for {N_LOGS} calls:', exec_time, 'seconds')