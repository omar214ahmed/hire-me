from celery import Celery
from helpers.config import get_settings

settings = get_settings()

celery = Celery(
    "hireme_ats",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["pipeline.tasks"],
)

celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,  # results kept in Redis for 1 hour
)