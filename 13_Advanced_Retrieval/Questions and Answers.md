# 🏗️ Activity + Bonus
Trace of RAG Fusion chain:

![img](media/RAG%20Fusion%20Chain.png)

## Setup
- Total documents: `100`
- Chains LLM: `gpt-3.5-turbo`
- Chains Embedding: `text-embedding-3-small`
- RAGAS generation LLM: `gpt-4o-mini`
- RAGAS generation Embedding: `text-embedding-3-small`
- RAGAS testset size: `10`
- Evaluation (judge) LLM: `gpt-4o-mini`

## Performance (RAGAS)
![img](media/Performance%20Comparison.png)
Summary:
- Performance itself depends on many factors (almost every step affects performance)
- `Answer relevancy`: Not sure why exactly so big difference, simple **Naive** chain just destroyed opponents. It seems like we have good chunks (1 csv row = 1 item), the simplest algorithm brings us the best result overall here.
- `Context Entity Recall`: All chains are quite the same. The leaders are **Contextual Compression** and **Fusion**, which is not surprising because the core of these approaches is Cohere reranking. So we can definitely say that reranking affects this metric the most.
- `Context Recall`: Big spread here. I would assume that the situation is similar to the first one, since we have good chunks we can easily extract correct vectors from DB using simple algorithms.
- `Factual Correctness`: Nothing special here, all chains are good. **Multi-Query** getting a little ahead.
- `Faithfulness`: Also nothing special, **BM25** slightly inferior to the others.
- `Noise Sensivity`: Absolute leader here is **Multi-Query**, because it generates many possible queries from initial one, so this approach affects metric the most.
- Overall, there is no obvious leader. Every retrieval has its own pros and cons. We should choose wisely one depending on our data and expected result.

## Latency (LangSmith)
_Because the metrics is latency, so lower value means better result_
![img](media/Latency%20Comparison.png)
Summary:
- Latency is affected by the number of additional requests to external sources (databases, LLMs, etc.) and final size of a prompt to LLM model (more tokens means latency increasing)
- **BM25's** leadership is not a surprise here. No requests, already pre-build and structured DB → the lowest possible latency.
- We can name three outsiders here. **Multi-Query** and **Fusion** using the same method for generating additional queries using LLM requests that affect latency a lot. **Multi-Query** is a part of the **Ensemble**, so **Ensemble** is also the outsider.

## Costs (LangSmith)
_Note: here we exclude billing for embedding models & Cohere because LangSmith doesn't provide such info_
![img](media/Costs%20Comparison.png)
Summary:
- Cost is affected by additional requests to LLMs or services (like Cohere) and a huge context window with many input tokens.
- Seems to be if you want to burn your money, you definitely should choose **Ensemble** because it contains all other retrievals. **Multi-Query** is costly because it's using additional generation. I'm not sure why we don't see **Fusion** here, because it's also generating additional queries. I double-checked values that we use for drawing this graph and they are absolutely correct.
- The least expensive methods are: **BM25** (no additional requests), **Parent** (using internal DB), **Contextual Compression** (internal DB)
- Also, we should remember that **Contextual Compression**, **Ensemble** and **Fusion** use Cohere reranking that is also not a free service
