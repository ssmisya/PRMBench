from .qwq_inferencer_dataset import *
from .qwq_inferencer import *

inferencer_type_dict = dict(
    generate_prm=dict(model=QwQGeneratePRMInferencer, dataset=QwQGeneratePRMDataset),
    parallel_generate_prm=dict(model=QwQParallelGeneratePRMInferencer, dataset=QwQGeneratePRMDataset),
)