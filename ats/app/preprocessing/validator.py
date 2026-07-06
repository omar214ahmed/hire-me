import os
import re
import uuid
from fastapi import UploadFile

from helpers.config import get_settings


class FileService:

    def __init__(self):
        self.app_settings = get_settings()
        self.size_scale = 1
        self.files_dir = "files"

    async def validate_uploaded_file(self, file: UploadFile):

        if file.content_type not in self.app_settings.FILE_ALLOWED_TYPES:
            return False, "FILE_TYPE_NOT_SUPPORTED"

        contents = await file.read()
        size = len(contents)

        await file.seek(0)

        if size > self.app_settings.FILE_MAX_SIZE * self.size_scale:
            return False, "FILE_SIZE_EXCEEDED"

        return True, "FILE_VALIDATED_SUCCESS"

    def validate_extracted_content(self, char_count: int, page_count):
        """
        Second validation gate, run *after* text extraction (pdfplumber/
        PyMuPDF/python-docx), on the resulting char/page counts —
        catches empty/scanned docs and unreasonably long ones that a
        MIME/size check alone can't see.
        """
        if char_count < self.app_settings.FILE_MIN_CHARS:
            return False, "FILE_CONTENT_TOO_SHORT"

        if char_count > self.app_settings.FILE_MAX_CHARS:
            return False, "FILE_CONTENT_TOO_LONG"

        if page_count is not None and page_count > self.app_settings.FILE_MAX_PAGES:
            return False, "FILE_TOO_MANY_PAGES"

        return True, "FILE_CONTENT_VALIDATED_SUCCESS"

    def get_clean_file_name(self, orig_file_name: str):
        base = os.path.basename(orig_file_name)

        base = re.sub(r'[^a-zA-Z0-9._-]', '_', base)
        base = re.sub(r'_+', '_', base)

        return base.strip("._")

    def generate_unique_filepath(self, orig_file_name: str, project_id: str):

        project_path = self.get_project_path(project_id)

        cleaned = self.get_clean_file_name(orig_file_name)

        unique_id = uuid.uuid4().hex[:12]

        filename = f"{unique_id}_{cleaned}"

        return os.path.join(project_path, filename), filename

    def get_project_path(self, project_id: str):
        project_dir = os.path.join(self.files_dir, project_id)
        os.makedirs(project_dir, exist_ok=True)
        return project_dir