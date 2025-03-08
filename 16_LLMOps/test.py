import time
from time import perf_counter

result = []

for _ in range(10):
    s = perf_counter()
    # result.append(hf_llm.invoke("What is DeepSeek-R1?"))
    time.sleep(1)
    e = perf_counter()
    print("\nTotal Runtime = {:.6f}".format(e - s))
