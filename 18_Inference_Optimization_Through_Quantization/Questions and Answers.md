![img](Screenshot%20Evaluation.png)

## ❓ Question:

Describe the difference in performance profiles between the three solutions.

#### Time to First Token
So as we discussed previously, BNB can slightly affect the latency of generation due to computational overhead.
And to be honest, it's not the primary goal of this library, instead it's focusing on decreasing of memory usage of our models.
So here we can see the exact latency penalty that we mentioned during the class.

#### Total Generation Time
This one is a combined result of other metrics, the total difference that we get at the final. 
We can say that this one consists of our another three metrics.

#### Tokens per Second
Almost three times more tokens we can get using AWQ / GPTQ quantization, because these two focused of inference speed up.

#### Mean Inter-token lat
On first look, the real difference is not so huge, only 0.04 ms.
But we know that sometimes there are thousands and thousands of tokens that are generated one by one. 
A small difference of latency in a single case can grow to a much bigger one in the future.

#### Overall
Generally, I can't name this comparison a relevant or, let's call it an honest one.
The purposes of compared mechanisms are slightly different.
We can't really say that provided comparison of AWQ and GPTQ shows us a big difference between them, because both of them are focused on increasing the latency of our generation and quite similar algorithms.
Instead, BNB allows us to run really huge models on not very performative hardware.