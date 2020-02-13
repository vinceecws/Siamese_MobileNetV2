import concurrent.futures as futures
import time

def testSleep(duration):
    print('Sleeping for {}s'.format(duration))
    time.sleep(duration)
    print('Done sleeping')

print('Testing with loop')
t1 = time.perf_counter()
for i in range(5):
    testSleep(i)
t2 = time.perf_counter()

print('Time taken {}s'.format(t2 - t1))

print('Testing with Multi-processing')
t1 = time.perf_counter()
with futures.ProcessPoolExecutor() as executor:
    executor.map(testSleep, [i for i in range(5)])
t2 = time.perf_counter()

print('Time taken {}s'.format(t2 - t1))