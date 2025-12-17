from model import *
import enum

class Ablations(enum.IntEnum):
    FULL = 0
    NO_UNDERSTAND = 1
    NO_SOCIAL_COST = 2
    NO_INFER = 3
    ONLY_UNDERSTAND = 4
    NO_PSICK_INTERP = 5
    
NUM_PARAMS = {
    0: 6, # number of free parameters in full model
    1: 5,
    2: 4,
    3: 5,
    4: 3, # if only understand: vary understanding and two cost terms 
    5: 6 # NOT run.... 
}

def run_model(params, ablation):
    # teacher[f, r, h, t, d, u, s](uc, urc, tc, tca, sc, scc, p_both, p_either)
    out = teacher(
        0. if ablation == Ablations.NO_UNDERSTAND else params['uc'],
        0. if ablation == Ablations.NO_INFER or ablation == Ablations.ONLY_UNDERSTAND else params['urc'],
        params['tc'],
        params['tca'],
        0. if ablation == Ablations.NO_SOCIAL_COST or ablation == Ablations.ONLY_UNDERSTAND else params['sc'],
        0. if ablation == Ablations.NO_SOCIAL_COST or ablation == Ablations.ONLY_UNDERSTAND else params['scc'],
        params.get('p_both', 0.5),
        params.get('p_either', 0.5),
        params.get('alphac', 0.5),
        params.get('alpha', 0.5),
    )
    # Slice S dimension as before: take FLUENT (index 0)
    return out[..., 0]
