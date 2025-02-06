from abc import abstractmethod, ABC
from io import BytesIO

import PyPDF2


class BaseParser(ABC):
    @abstractmethod
    def load(self, data: bytes) -> str:
        ...


class TxtParser(BaseParser):
    def load(self, data: bytes) -> str:
        return data.decode("utf-8", errors="ignore")


class PdfParser(BaseParser):
    def load(self, data: bytes) -> str:
        pdf_stream = BytesIO(data)
        pdf_reader = PyPDF2.PdfReader(pdf_stream)
        extracted_text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            extracted_text += page.extract_text()
        return extracted_text


ParsersMap: dict[str, BaseParser] = {".txt": TxtParser(), ".pdf": PdfParser()}
