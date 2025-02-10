from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document


def load_files(path_to_data: str, extensions: list[str]) -> list[Document]:
    res = []
    for ext in extensions:
        res.extend(DirectoryLoader(path_to_data, glob=ext).load())
    return res
