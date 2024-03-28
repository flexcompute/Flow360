import datetime

from flow360.log import log

N_LOGS = 100000

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