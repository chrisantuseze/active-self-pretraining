import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

def acc_pretrain_data_plots():
    # plot the accuracy against the number of pretrain data for generalist and specialist pretraining
    # use 25, 50, 75, and 100% of the target dataset
    # one plot for each of the datasets: 
    # 1. imagenet-sketch-quickdraw
    ## 2. imagenet-clipart-quickdraw (Depends on the availability of resources)
    # 3. cifar10-sketch-quickdraw
    ## 4. cifar10-clipart-quickdraw (Depends on the availability of resources)

    # do this for SimCLR, DCL, and MYOW/SwAV
    # so we have 4 * 2(or 4) * 3 = Total of 24 cycle of pretrainings
    # do these using the best AL method

    # also vary the AL method using DCL and cifar10 as the base dataset. This informs the AL method to be used for the previous steps
    # so we have:
    # 1. cifar10(ssl = DCL)-sketch(al = LC)-quickdraw
    # 2. cifar10(ssl = DCL)-sketch(al = EN)-quickdraw
    # 3. cifar10(ssl = DCL)-sketch(al = BOTH)-quickdraw
    None