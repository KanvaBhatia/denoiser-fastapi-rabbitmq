from fastapi import APIRouter
from starlette.responses import JSONResponse
from celery_tasks.tasks import process_vid
from config.celery_utils import get_task_info

router = APIRouter(prefix='/video', tags=['VideoProcessing'], responses={404: {"description": "Not found"}})


@router.post("/")
def helloapi() -> dict:
    return {"Hello": "API"}



@router.post("/process")
async def process(video_url: str):
    task = process_vid.apply_async(args=[video_url])
    return JSONResponse({"task_id": task.id})


@router.get("/task/{task_id}")
async def get_task_status(task_id: str) -> dict:
    return get_task_info(task_id)