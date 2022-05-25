import re
import time
from string import punctuation
from collections import Counter
from itertools import chain
import sys
import statistics


def read_file_in_chunks(file_object, chunk_size=2 << 13):
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data


def countInFile(filename):
    carry = ''

    def handle_chunk(chunk):
        nonlocal carry
        res = (carry + re.sub('[' + punctuation + ']', '', chunk).lower()).split()
        carry = '' if (str(chunk[-1]) in ' \n\r\t,\'\"-') else res.pop()
        return res

    with open(filename, encoding="latin-1") as f:
        return Counter(chain.from_iterable(map(handle_chunk, read_file_in_chunks(f))))


if __name__ == '__main__':

    for file in sys.argv[1:]:
        times = []
        for i in range(10):
            start = time.time()
            countInFile(file)
            end = time.time()
            times.append((end - start))

        avg = statistics.mean(times)
        stdev = statistics.stdev(times)
        print(f'{avg},{stdev}')
