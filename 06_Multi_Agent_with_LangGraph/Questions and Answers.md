#### Activity #1 (Bonus Marks)
Allow the system to dynamically fetch Arxiv papers instead of hard coding them.

HINT: Tuesday's assignment will be very useful here.

In this block I just replace RAG via QDrant with Arxiv tool
```python
# Dynamic data retrieving from arxiv instead hardcoded one

from langchain_community.tools.arxiv.tool import ArxivQueryRun

tool = ArxivQueryRun()
openai_chat_model.bind_tools([tool])
rag_chain = (
    {"context": itemgetter("question") | tool, "question": itemgetter("question")}
    | rag_prompt | openai_chat_model | StrOutputParser()
)
rag_chain.invoke({"question" : "What does the 'context' in 'long context' refer to?"})
```
In the attached screenshot we can see that this tool was triggered

##### ❓ Question #1:

Why is a "powerful" LLM important for this use-case?

I believe that the main reason of selecting powerful model, 
is the highest responsibility across all other nodes that will just do their job.
On this step (`router` step) we need maximum analysis of provided data and smartness from our supervisor, 
because it will decide which path we will choose to achieve final goal. 

LLM model answers this question:
1. A weaker LLM might struggle with context tracking, decision-making, and error handling, leading to inefficiencies and failures in workflow execution. 
2. In contrast, a powerful LLM ensures robustness, adaptability, and scalability, making it ideal for a building supervisor role in LangGraph applications.

What tasks must our Agent perform that make it such that the LLM's reasoning capability is a potential limiter?

What about some abstract things that can be understood only by human? I'm not sure that I clearly get the question, 
but I can suggest that some complex math, physics, coding cases are limited by reasoning capability. 
Some things that have no direct answer like human being, self-consciousness, etc.


##### ❓ Question #2:

How could you make sure your Agent uses specific tools that you wish it to use? 
Based on prompt engineering, because the mechanism on choosing is LLM as it is. 
Since AI engineering is about not deterministic outputs, 
we can't be confident that the tool we want will be used in 100% cases.
But we can make this chance very high by improving our prompting, 
tool design itself (like description, info about input arguments and other things)

Are there any ways to concretely set a flow through tools?

We can build our chain of tools step by step, 
but this will no longer be associated with any “decision making”, 
but will simply be the execution of a scenario.
Example:
```python
graph.add_edge("A", "B")
graph.add_edge("B", "C")
graph.add_edge("C", "D")
```
or we can define some strict conditional edge (router) design
```python
def router(state):
    if "x" in state:
        return "A"
    if "y" is state:
        return "B"
    if "z" in state:
        return "C"
    return END
```
