#### ❓Question #1:

The default embedding dimension of `text-embedding-3-small` is 1536, as noted above. 

1. Is there any way to modify this dimension?
   - Yes, we can modify some of newest embedding models using `dimensions` field 
OR we can change the embedding dimension AFTER we generate it. 
But manual dimension change gives us greater freedom and imposes responsibility for the quality of such optimization
2. What technique does OpenAI use to achieve this?
   - OpenAI use [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) to implement this feature


#### ❓Question #2:

1. What are the benefits of using an `async` approach to collecting our embeddings?
   - There is no need to wait for all the I/O operations to complete, which takes up most of the CPU idle time


#### ❓ Question #3:

1. When calling the OpenAI API - are there any ways we can achieve more reproducible outputs?
    - We can play with temperature value. As lower temperature - as more predicted and reproducible will be output
    - Don't make your prompt very huge (like adding many RAG context items) - it can confuse model
    - Use strict and structured prompt templates every time
    - Use the same model version


#### ❓ Question #4:

1. What prompting strategies could you use to make the LLM have a more thoughtful, detailed response?
    - `Chain of thoughts` or `step-by-step` strategy where LLM will try to think about our request gradually, processing each step patiently
    - Specify desired format. It can be just response structure (f.e. "please start with `a`, then go to `b` and finish with `c`), or output format like table, markdown, flat text and etc.
    - Role playing. Frame the model as an expert, teacher, or other relevant role to encourage a more comprehensive response
2. What is that strategy called?
    - If we're speaking about RAG, we can call this strategy as `Guided Contextual Prompting` or `In-context Learning`