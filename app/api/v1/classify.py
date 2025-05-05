from typing import Annotated

from starlette.concurrency import run_in_threadpool
from fastapi import APIRouter, UploadFile, File, Body, HTTPException, Depends

from app.dependencies import get_classify_service
from app.services.classify import ClassifyService
from app.domain.entities import FileEntity, ReferenceEntity
from app.schemas.schemas import ReferenceSchema, ClassifySchema


router = APIRouter(
    prefix="/classify",
    tags=["classify"],
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"},
    },
)


@router.post("/", response_model=list[ClassifySchema])
async def classify(
    files: Annotated[list[UploadFile], File(...)],
    references_info: Annotated[list[ReferenceSchema], Body(...)],
    classify_service: Annotated[ClassifyService, Depends(get_classify_service)],
):
    try:
        contents = [await file.read() for file in files]
        file_entities = [
            FileEntity(
                file=content,
                references_info=ReferenceEntity(
                    id=reference.id,
                    target=reference.target,
                    text=reference.text,
                    pages=reference.pages,
                    width=reference.width,
                    height=reference.height,
                    template=reference.template,
                    example=reference.example,
                    qr=reference.qr,
                    signature=reference.signature,
                    render=reference.render,
                ),
            )
            for content, reference in zip(contents, references_info)
        ]
        # Run classification in threadpool to avoid blocking the event loop
        classify_entities = await run_in_threadpool(classify_service.classify, file_entities)
        return classify_entities
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
