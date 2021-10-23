import celery
from celery.signals import after_setup_task_logger, after_setup_logger

from config import vars, log


class Celery(celery.Celery):

    def gen_task_name(self, name: str, module: str) -> str:
        if module.startswith('src.'):
            module = module[4:]

        return super().gen_task_name(name, module)


celery_app = Celery('ZReader',
                    broker=vars.CELERY_BROKER,
                    backend=vars.CELERY_BACKEND,
                    include=['app.api.backend.tasks']
                    )

celery_app.conf.update({
    'task_serializer': 'json',
    'result_serializer': 'json',
    'accept_content': ['json'],
    'worker_prefetch_multiplier': 1,
    'task_acks_late': True,
    'task_track_started': True,
    'task_reject_on_worker_lost': True,
})


# Configure Celery logger
@after_setup_task_logger.connect
def setup_task_logger(logger, *args, **kwargs):
    logger.addHandler(hdlr=log.error_handler)
    logger.addHandler(hdlr=log.info_handler)


@after_setup_logger.connect
def setup_task_logger(logger, *args, **kwargs):
    logger.addHandler(hdlr=log.error_handler)
    logger.addHandler(hdlr=log.info_handler)
