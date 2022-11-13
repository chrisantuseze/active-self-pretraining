
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

import subprocess
subprocess.Popen(["python","main.py"])
