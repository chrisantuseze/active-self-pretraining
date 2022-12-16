'''
@misc{azabou2021view,
      title={Mine Your Own vieW: Self-Supervised Learning Through Across-Sample Prediction}, 
      author={Mehdi Azabou and Mohammad Gheshlaghi Azar and Ran Liu and Chi-Heng Lin and Erik C. Johnson 
              and Kiran Bhaskaran-Nair and Max Dabagia and Keith B. Hengen and William Gray-Roncal 
              and Michal Valko and Eva L. Dyer},
      year={2021},
      eprint={2102.10106},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
'''

import random

import torch
import numpy as np


def set_random_seeds(random_seed=0):
    r"""Sets the seed for generating random numbers.

    Args:
        random_seed: Desired random seed.
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
