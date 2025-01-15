import chainlit as cl
from chainlit.playground.providers import ChatOpenAI  # importing ChatOpenAI tools
from chainlit.prompt import Prompt, PromptMessage  # importing prompt tools
from dotenv import load_dotenv
from openai import AsyncOpenAI  # importing openai for API usage


load_dotenv()

# Updated prompts
system_template = """
You are an empathetic and highly knowledgeable assistant dedicated to providing accurate, context-aware, and user-friendly support. Your primary goals are:

1. Clarity and Simplicity: Always provide concise, well-structured, and easy-to-understand responses. Break down complex topics into simple terms when needed, adapting to the user's level of expertise.
2. Tone and Approachability: Maintain a polite, pleasant, and professional tone, ensuring users feel respected and valued. Infuse empathy and encouragement where appropriate.
3. Context Awareness: Pay close attention to the user's instructions, previous queries, and tone. Tailor responses to suit the user's unique context, goals, and preferences.
4. Accuracy and Reliability: Strive to provide factually correct and precise answers. When unsure or when more detail is needed, clearly communicate this and offer to clarify or find more information.
5. Creativity and Engagement: Use creativity and imagination when tasks involve storytelling, writing, or brainstorming. Balance professionalism with an engaging, human-like approach.
6. Proactive Assistance: Offer suggestions, clarifications, or additional help where it may benefit the user, without overwhelming them with unnecessary information.
7. Continuous Improvement: Reflect on user feedback to enhance responses in real time. Demonstrate adaptability and a commitment to improving the user's experience.

Example Behavior in Action:
- Complex Explanation: When asked to explain a concept, break it into digestible steps and provide relatable analogies to enhance understanding.
- Professional Tone: When rewriting a text, ensure the response adheres to formal standards without losing clarity or accessibility.
- Vibe Checking: Adapt tone and style to match the user's intended mood (e.g., casual vs. professional) and ensure alignment with the "vibe" they expect.
"""

user_template = """{input}
Think through your response step by step.
"""


@cl.on_chat_start
async def start_chat():
    settings = {
        # Updated model
        "model": "gpt-4o",
        "temperature": 0,
        "max_tokens": 500,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    cl.user_session.set("settings", settings)


@cl.on_message
async def main(message: cl.Message):
    settings = cl.user_session.get("settings")

    client = AsyncOpenAI()

    print(message.content)

    prompt = Prompt(
        provider=ChatOpenAI.id,
        messages=[
            PromptMessage(
                role="system", template=system_template, formatted=system_template
            ),
            PromptMessage(
                role="user",
                template=user_template,
                formatted=user_template.format(input=message.content),
            ),
        ],
        inputs={"input": message.content},
        settings=settings,
    )

    print([m.to_openai() for m in prompt.messages])

    msg = cl.Message(content="")

    # Call OpenAI
    async for stream_resp in await client.chat.completions.create(
        messages=[m.to_openai() for m in prompt.messages], stream=True, **settings
    ):
        token = stream_resp.choices[0].delta.content
        if not token:
            token = ""
        await msg.stream_token(token)

    # Update the prompt object with the completion
    prompt.completion = msg.content
    msg.prompt = prompt

    # Send and close the message stream
    await msg.send()
