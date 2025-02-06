from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

EmbeddingLLM = OpenAIEmbeddings(model="text-embedding-3-small")
MiniLLM = ChatOpenAI(model="gpt-4o-mini")
TurboLLM = ChatOpenAI(model="gpt-4-turbo")
