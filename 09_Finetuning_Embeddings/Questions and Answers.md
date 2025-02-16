#### 🏗️ Activity #2:
Both of these losses sound "cool", but what are they - exactly - under the hood?

Why are these losses specifically doing? Please write a short summary of each loss.

NOTE: This is a course focused on AI Engineering and the application of AI - looking for a hint? Try pasting the code (linked above) into ChatGPT/Claude to write the summary!


`MultipleNegativesRankingLoss`

Loss function that accepts positive pairs as input, and generates negative pairs randomly 
(btw. we can also provide negative pairs if we want, we're able to do it just by passing `[initial_query, positive_value, negative_value]`).


`MatryoshkaLoss`

The core idea of this loss, is that it will modify provided loss function 
(in our case it's a MultipleNegativesRankingLoss) so it will be used for processing different embedding dimensions.
We will use this loss when we want to handle multiple dimensions of embedding, from little to bigger ones.

Some useful tables from internet + gpt

**Core Difference**

| Feature | MultipleNegativesRankingLoss (MNRL) | MatryoshkaLoss (ML) |
|---------|--------------------------------------|----------------------|
| **Purpose** | Optimizes embeddings for contrastive learning by ranking positives over negatives | Learns hierarchical structure in embeddings by ensuring finer details for small radii and generalization for larger radii |
| **Loss Computation** | Maximizes the similarity between positive pairs while minimizing similarity to all negatives | Enforces similarity at different levels of granularity, optimizing at multiple radii in embedding space |
| **Optimization Goal** | Improves hard negative separation for retrieval and contrastive learning | Produces structured embeddings with coarse-to-fine representation learning |
| **Embedding Structure** | Encourages high similarity between positive samples while penalizing negatives | Embeddings maintain different levels of detail, useful for multi-level retrieval |

**When and Why to Use Each Loss?**

| Scenario | MNRL | ML |
|----------|------|----|
| **Information Retrieval (Search Engines, Document Ranking, etc.)** | ✅ Strong choice for ranking-based retrieval tasks | ❌ Less relevant unless hierarchical retrieval is needed |
| **Contrastive Learning (Siamese Networks, Triplet Loss Alternative)** | ✅ Works well for learning discriminative representations | ❌ Not specifically designed for contrastive learning |
| **Hierarchical Embedding Learning (Generalization vs. Detail Tradeoff)** | ❌ Doesn’t explicitly enforce hierarchical learning | ✅ Ideal for applications where coarse-to-fine granularity is important |
| **Few-Shot Learning or Transfer Learning** | ✅ Helps create robust representations for retrieval tasks | ✅ Better at creating embeddings that generalize across multiple levels |
| **Clustering or Similarity Matching** | ✅ Suitable for discriminative tasks | ✅ Suitable for structured similarity learning |

**Which One Should You Use?**

| **Criterion** | **MultipleNegativesRankingLoss** | **MatryoshkaLoss** |
|--------------|----------------------------------|--------------------|
| **Best For** | Search, ranking, and retrieval | Multi-resolution learning, hierarchical embeddings |
| **Strength** | Discriminative contrastive learning with negatives | Coarse-to-fine embedding structure |
| **Weakness** | Can struggle with hierarchical learning | Not ideal for traditional contrastive learning |
| **Example Use Case** | Sentence embeddings for search engines | Semantic compression and hierarchical similarity retrieval |

#### ❓Question #2:

Which LCEL RAG Chain do you think answered the questions better, and why?

I really want to rephrase a question a little bit, because we know that we didn't modify our LLMs itself.
My answer to exact question is - they both answered correctly and good! They both followed our instructions successfully. 
We ask them to use provided context, and if there is no context - respond "don't know" and they did exact these things.

Instead, we will compare which chain `retrieval` was better. And here is another chain: better retrieval means better answer generation.
So we can definitely say that fine-tuned embedding model from `Snowflake` shows much better results vs initial and 
from-the-box OpenAI model simply because it extracts the context better.

And this is not surprising at all, because we fed specific and exact retrieval cases to our fine-tuned model. 
In other words: we show to our model how to extract the context correctly.
