from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import tiktoken_len


TextSplitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0,
    length_function=tiktoken_len,
)
