import re

from langchain_core.documents import Document


class WordsTextSplitter:
    """
    Optimized text splitter by words
    """

    def split_documents(
        self, docs: list[Document], *, chunk_size=1000, overlap=50
    ) -> list[str]:
        result = []
        for doc in docs:
            words = re.split(" |\n\n", doc.page_content)
            # Remove empty
            words = [w for w in words if w]
            chunks = []
            for i in range(0, len(words), chunk_size - overlap):
                chunk = " ".join(words[i : i + chunk_size])  # Join words into a chunk
                chunks.append(chunk)
            result.extend(chunks)

        return result
