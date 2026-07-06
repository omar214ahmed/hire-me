from helpers.config import get_settings
from preprocessing.extractor import extract_pdf, extract_docx
from preprocessing.cleaner import clean_resume_text


class PreprocessingDispatcher:

    def __init__(self):
        self.settings = get_settings()

    async def process_file(self, file):

        file_bytes = await file.read()

        if file.content_type == "application/pdf":

            raw_text, parser, page_count = extract_pdf(file_bytes)

        elif file.content_type in (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword"
        ):

            raw_text, parser, page_count = extract_docx(file_bytes)

        else:
            raise ValueError(
                f"Unsupported file type: {file.content_type}"
            )

        cleaned_text = clean_resume_text(raw_text)

        return {
            "filename": file.filename,
            "parser": parser,
            "text": cleaned_text,
            "char_count": len(cleaned_text),
            "page_count": page_count,
        }