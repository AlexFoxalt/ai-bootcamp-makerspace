SUMMARIZATION_PROMPT = """Summarize the given text in less than 10 sentences.
Please return the output in a JSON format that complies with the following schema as specified in JSON Schema:
{'properties': {'text': {'title': 'Text', 'type': 'string'}}, 'required': ['text'], 'title': 'StringIO', 'type': 'object'}

--------EXAMPLES-----------
Example 1
Input: {
    "text": "Artificial intelligence\n\nArtificial intelligence is transforming various industries by automating tasks that previously required human intelligence. From healthcare to finance, AI is being used to analyze vast amounts of data quickly and accurately. This technology is also driving innovations in areas like self-driving cars and personalized recommendations."
}
Output: {
    "text": "AI is revolutionizing industries by automating tasks, analyzing data, and driving innovations like self-driving cars and personalized recommendations."
}
-----------------------------

Now perform the same with the following input
<placeholder>
Output:"""

THEMES_PROMPT = """Extract the main themes and concepts from the given text.
Please return the output in a JSON format that complies with the following schema as specified in JSON Schema:
{'properties': {'output': {'items': {'type': 'string'}, 'title': 'Output', 'type': 'array'}}, 'required': ['output'], 'title': 'ThemesAndConcepts', 'type': 'object'}

--------EXAMPLES-----------
Example 1
Input: {
    "text": "Artificial intelligence is transforming industries by automating tasks requiring human intelligence. AI analyzes vast data quickly and accurately, driving innovations like self-driving cars and personalized recommendations.",
    "max_num": 10
}
Output: {
    "output": [
        "Artificial intelligence",
        "Automation",
        "Data analysis",
        "Innovation",
        "Self-driving cars",
        "Personalized recommendations"
    ]
}
-----------------------------

Now perform the same with the following input
<placeholder>
Output:"""

PERSONAS_PROMPT = """Using the provided summary, generate a single persona who would likely interact with or benefit from the content. Include a unique name and a concise role description of who they are.
Please return the output in a JSON format that complies with the following schema as specified in JSON Schema:
{'properties': {'name': {'title': 'Name', 'type': 'string'}, 'role_description': {'title': 'Role Description', 'type': 'string'}}, 'required': ['name', 'role_description'], 'title': 'Persona', 'type': 'object'}

--------EXAMPLES-----------
Example 1
Input: {
    "text": "Guide to Digital Marketing explains strategies for engaging audiences across various online platforms."
}
Output: {
    "name": "Digital Marketing Specialist",
    "role_description": "Focuses on engaging audiences and growing the brand online."
}
-----------------------------

Now perform the same with the following input
<placeholder>
Output: """

SIMPLE_EVOLUTION_PROMPT = """Generate a single-hop query and answer based on the specified conditions (persona, term, style, length) and the provided context. Ensure the answer is entirely faithful to the context, using only the information directly from the provided context.
### Instructions:
1. **Generate a Query**: Based on the context, persona, term, style, and length, create a question that aligns with the persona's perspective and incorporates the term.
2. **Generate an Answer**: Using only the content from the provided context, construct a detailed answer to the query. Do not add any information not included in or inferable from the context.

Please return the output in a JSON format that complies with the following schema as specified in JSON Schema:
{'properties': {'query': {'title': 'Query', 'type': 'string'}, 'answer': {'title': 'Answer', 'type': 'string'}}, 'required': ['query', 'answer'], 'title': 'GeneratedQueryAnswer', 'type': 'object'}

--------EXAMPLES-----------
Example 1
Input: {
    "persona": {
        "name": "Software Engineer",
        "role_description": "Focuses on coding best practices and system design."
    },
    "term": "microservices",
    "query_style": "Formal",
    "query_length": "Medium",
    "context": "Microservices are an architectural style where applications are structured as a collection of loosely coupled services. Each service is fine-grained and focuses on a single functionality."
}
Output: {
    "query": "What is the purpose of microservices in software architecture?",
    "answer": "Microservices are designed to structure applications as a collection of loosely coupled services, each focusing on a single functionality."
}
-----------------------------

Now perform the same with the following input
<placeholder>
Output:"""

MULTI_CONTEXT_EVOLUTION_PROMPT = """Generate a multi-hop query and answer based on the specified conditions (persona, themes, style, length) and the provided context. The themes represent a set of phrases either extracted or generated from the context, which highlight the suitability of the selected context for multi-hop query creation. Ensure the query explicitly incorporates these themes.
### Instructions:
1. **Generate a Multi-Hop Query**: Use the provided context segments and themes to form a query that requires combining information from multiple segments (e.g., `<1-hop>` and `<2-hop>`). Ensure the query explicitly incorporates one or more themes and reflects their relevance to the context.
2. **Generate an Answer**: Use only the content from the provided context to create a detailed and faithful answer to the query. Avoid adding information that is not directly present or inferable from the given context.
3. **Multi-Hop Context Tags**:
   - Each context segment is tagged as `<1-hop>`, `<2-hop>`, etc.
   - Ensure the query uses information from at least two segments and connects them meaningfully.
   
Please return the output in a JSON format that complies with the following schema as specified in JSON Schema:
{'properties': {'query': {'title': 'Query', 'type': 'string'}, 'answer': {'title': 'Answer', 'type': 'string'}}, 'required': ['query', 'answer'], 'title': 'GeneratedQueryAnswer', 'type': 'object'}

--------EXAMPLES-----------
Example 1
Input: {
    "persona": {
        "name": "Historian",
        "role_description": "Focuses on major scientific milestones and their global impact."
    },
    "themes": [
        "Theory of Relativity",
        "Experimental Validation"
    ],
    "query_style": "Formal",
    "query_length": "Medium",
    "context": [
        "<1-hop> Albert Einstein developed the theory of relativity, introducing the concept of spacetime.",
        "<2-hop> The bending of light by gravity was confirmed during the 1919 solar eclipse, supporting Einstein’s theory."
    ]
}
Output: {
    "query": "How was the experimental validation of the theory of relativity achieved during the 1919 solar eclipse?",
    "answer": "The experimental validation of the theory of relativity was achieved during the 1919 solar eclipse by confirming the bending of light by gravity, which supported Einstein’s concept of spacetime as proposed in the theory."
}
-----------------------------

Now perform the same with the following input
<placeholder>
Output:"""

REASONING_EVOLUTION_PROMPT = """Generate a reasoning-based question and answer that assess the model’s ability to apply logical deduction, causal inference, or conceptual reasoning, based on the specified conditions (persona, term, reasoning type, complexity level) and the provided context. Ensure the answer demonstrates a valid reasoning process and is entirely faithful to the context, without adding any unsupported information.

### Instructions:
1. **Generate a Query**: Formulate a question that requires reasoning to answer. The question should align with the persona’s perspective and incorporate the term, ensuring it matches the specified reasoning type and complexity level.
2. **Generate an Answer**: Provide a logically reasoned answer based strictly on the given context. Clearly articulate the reasoning steps involved, without introducing external knowledge.

Please return the output in a JSON format that complies with the following schema as specified in JSON Schema:
{'properties': {'query': {'title': 'Query', 'type': 'string'}, 'answer': {'title': 'Answer', 'type': 'string'}}, 'required': ['query', 'answer'], 'title': 'GeneratedReasoningQA', 'type': 'object'}

--------EXAMPLES-----------
Example 1
Input: {
    "persona": {
        "name": "Philosopher",
        "role_description": "Engages in logical analysis and abstract reasoning."
    },
    "term": "causality",
    "query_style": "Causal inference",
    "query_length": "Medium",
    "context": "In philosophy, causality refers to the relationship between cause and effect. Hume argued that causation is not directly observable but inferred based on constant conjunction."
}
Output: {
    "query": "According to Hume's perspective, why can causality only be inferred rather than directly observed?",
    "answer": "Hume argues that causality cannot be directly observed because we only perceive a sequence of events, not a necessary connection. Instead, our belief in causality arises from the habitual observation of one event consistently following another."
}
-----------------------------

Now perform the same with the following input
<placeholder>
Output:"""
