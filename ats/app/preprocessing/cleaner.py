from ftfy import fix_text
import re
import unicodedata

def clean_resume_text(text: str) -> str:

    text = fix_text(text)

    text = unicodedata.normalize("NFKC", text)

    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)

    text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text, flags=re.I)

    text = re.sub(r"[ ]{2,}", " ", text)

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()