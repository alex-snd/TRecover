from celery import Celery
from celery.signals import after_setup_task_logger, after_setup_logger

from trecover.config import var, log

celery_app = Celery('TRecover',
                    broker=var.CELERY_BROKER,
                    backend=var.CELERY_BACKEND,
                    include=['trecover.app.api.backend.tasks'])

celery_app.conf.update({
    'task_serializer': 'json',
    'result_serializer': 'json',
    'accept_content': ['json'],
    'worker_prefetch_multiplier': 1,
    'task_acks_late': True,
    'task_track_started': True,
    'task_reject_on_worker_lost': True,
})


@after_setup_task_logger.connect
def setup_task_logger(logger, *args, **kwargs):
    """ Customize celery task logger. """

    logger.addHandler(hdlr=log.error_handler)
    logger.addHandler(hdlr=log.info_handler)


@after_setup_logger.connect
def setup_task_logger(logger, *args, **kwargs):
    """ Customize celery logger. """

    logger.addHandler(hdlr=log.error_handler)
    logger.addHandler(hdlr=log.info_handler)
