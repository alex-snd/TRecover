import celery

import config


class Celery(celery.Celery):

    def gen_task_name(self, name: str, module: str) -> str:
        if module.startswith('ml.'):
            module = module[4:]

        return super().gen_task_name(name, module)


celery_app = Celery('ZReader',
                    broker=config.CELERY_BROKER,
                    backend=config.CELERY_BACKEND,
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
