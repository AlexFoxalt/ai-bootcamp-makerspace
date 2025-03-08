##### ❓ Question #1:

What are some limitations you can see with this approach? When is this most/least useful. Discuss with your group!

1. Qdrant hosted in RAM: first limitation that has its own pros and cons. 
   1. \+ fast deploy in code with no additional DevOps (like docker)
   2. \+ latency (literally the fastest one)
   3. \- RAM storage of local PC is not infinite
   4. \- Database should be recreated on every run (which also means that we must embed our files on every run)
2. Hardcoded embedding dims num
   1. Not really a big issue, since we won't change the model RN. Just a possible one if we do.
3. Local files as cache storage
   1. \+ fast deploy / no need to deploy external DB using
   2. \+ local storage is much less expensive than RAM
   3. \- not very performative search
4. Low context recall `k=1`
   1. \+ small context window means less money for VM that hosts model or less money for tokens, if we use already deployed models like `gpt4` or `claude`.
   2. \- retrieval quality will decrease

##### 🏗️ Activity #1:

Create a simple experiment that tests the cache-backed embeddings.

```python
from time import perf_counter

for _ in range(10):
    s = perf_counter()
    retriever.invoke("What is DeepSeek-R1?")
    print(f"{perf_counter() - s:.6f}")
```

Output:
```text
1.442304
0.151021
0.187207
0.214860
0.196103
0.207202
...
```

##### ❓ Question #2:

What are some limitations you can see with this approach? When is this most/least useful. Discuss with your group!

1. Max new tokens `max_new_tokens=128`
   1. \+ pay less since output tokens are also billable
   2. \- it may cap output length, which might truncate longer responses
2. Memory cache:
   1. \+ fast and simple deploy / no additional servers
   2. \+ top (fastest) latency
   3. \- data loss on reload
   4. \- RAM storage is limited and expensive

##### 🏗️ Activity #2:

Create a simple experiment that tests the cache-backed embeddings.

Code is really quite the same
```python
from time import perf_counter

for _ in range(10):
    s = perf_counter()
    hf_llm.invoke("What is DeepSeek-R1?")
    print(f"{perf_counter() - s:.6f}")
```

Output:
```text
8.702055
0.000334
0.000082
0.000071
0.000069
...
```

##### 🏗️ Activity #3:

Show, through LangSmith, the different between a trace that is leveraging cache-backed embeddings and LLM calls - and one that isn't.

Post screenshots in the notebook!

![img](Screenshot%20Chain%20Cache%20Comparison.png)

Nothing really to add. The numbers speak for themselves. Perfect final