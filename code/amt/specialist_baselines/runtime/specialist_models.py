"""Official-source architectures, with explicit offline AMT task adapters.

The names do not alias legacy patch_transformer. SAITS uses the shared
per-case missing-target MSE, not the original ORT+MIT training objective.
"""
from ..vendor.patchtst.adapter import PatchTSTAdapter
from ..vendor.saits.adapter import SAITSAdapter

CLEAN_TASKS = ('forecasting', 'interpolation', 'imputation')


def supported_models(task):
    if task not in CLEAN_TASKS:
        raise ValueError('The specialist supplement supports clean tasks only')
    return ('patchtst',) if task == 'forecasting' else ('saits',)


def build_model(name, task, **options):
    if name not in supported_models(task):
        raise ValueError('Unsupported specialist model/task: %s/%s' % (name, task))
    if name == 'patchtst':
        return PatchTSTAdapter(**options)
    return SAITSAdapter(**options)
