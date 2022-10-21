
# import torch
# from numba import cuda

# def free_gpu_cache():
#     print("Initial GPU Usage")

#     torch.cuda.empty_cache()

#     cuda.select_device(0)
#     cuda.close()
#     cuda.select_device(0)

#     print("GPU Usage after emptying the cache")

# free_gpu_cache()  
# 
import numpy as np                         

my_list = [
    {
        "name": "Osita",
        "position": 1
    },
    {
        "name": "Charles",
        "position": 2
    },
    {
        "name": "Ezinne",
        "position": 3
    },
    {
        "name": "Nna",
        "position": 4
    },
    {
        "name": "Uche",
        "position": 5
    },
    {
        "name": "Chinonso",
        "position": 6
    },
]
idx = np.argsort(my_list)

