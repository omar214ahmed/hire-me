from fastapi import (
    APIRouter,
    UploadFile,
    Request,
    Depends,
    status,
)
from fastapi.responses import JSONResponse
import aiofiles

from helpers.config import Settings, get_settings
from preprocessing.validator import FileService
from preprocessing.dispatcher import PreprocessingDispatcher
from pipeline import cv_processor, storage
from pipeline.models import ModelRegistry


candidate_router = APIRouter(
    prefix="/api/v1/candidates",
    tags=["api_v1", "candidates"],
)


@candidate_router.post("/upload")
async def upload_resume(
    request: Request,
    file: UploadFile,
    app_settings: Settings = Depends(get_settings),
):

    file_service = FileService()
    dispatcher = PreprocessingDispatcher()

    # -----------------------------
    # Validate uploaded file
    # -----------------------------
    is_valid, signal = await file_service.validate_uploaded_file(
        file=file
    )

    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": signal
            }
        )

    # -----------------------------
    # Save file
    # -----------------------------
    file_path, file_id = file_service.generate_unique_filepath(
        orig_file_name=file.filename,
        project_id="candidates"
    )

    try:

        async with aiofiles.open(file_path, "wb") as f:

            while chunk := await file.read(1024 * 1024):
                await f.write(chunk)

        await file.seek(0)

    except Exception as e:

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "signal": "FILE_UPLOAD_FAILED",
                "error": str(e)
            }
        )

    # -----------------------------
    # Extract + Clean
    # -----------------------------
    try:

        result = await dispatcher.process_file(file)

    except Exception as e:

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "signal": "PREPROCESSING_FAILED",
                "error": str(e)
            }
        )

    # -----------------------------
    # Check type: content-level checks (char count, page count)
    # -----------------------------
    content_valid, content_signal = file_service.validate_extracted_content(
        char_count=result["char_count"],
        page_count=result["page_count"],
    )

    if not content_valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": content_signal,
                "char_count": result["char_count"],
                "page_count": result["page_count"],
            }
        )

    # -----------------------------
    # Split sections + extract labels (regex)
    # -----------------------------
    try:

        cv_parsed = cv_processor.parse_cv(cv_id=file_id, text=result["text"])

    except Exception as e:

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "signal": "CV_LABELING_FAILED",
                "error": str(e)
            }
        )

    # -----------------------------
    # Embed CV text once and persist
    # -----------------------------
    try:

        embedder = ModelRegistry.get_embedder()
        cv_text = cv_parsed.get("full_text", result["text"])
        cv_embedding = embedder.encode(
            [cv_text],
            batch_size=1,
            max_length=512,
        )["dense_vecs"][0].tolist()

        await storage.save_candidate(
            file_id=file_id,
            filename=file.filename,
            cv_parsed=cv_parsed,
            cv_embedding=cv_embedding,
        )

    except Exception as e:

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "signal": "CV_EMBEDDING_FAILED",
                "error": str(e)
            }
        )

    # -----------------------------
    # Success
    # -----------------------------
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "signal": "CV_PREPROCESSED_SUCCESS",
            "file_id": file_id,
            "filename": file.filename,
            "data": result,
            "parsed": {
                "candidate_id": cv_parsed["id"],
                "email": cv_parsed["email"],
                "phone": cv_parsed["phone"],
                "years_of_experience": cv_parsed["years_of_experience"],
                "degrees": cv_parsed["degrees"],
                "skills": cv_parsed["skills"],
                "languages": cv_parsed["languages"],
            },
        }
    )