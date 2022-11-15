# import pandas as pd
# # import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt

# from models.utils.commons import accuracy 

# # import seaborn as sns

# def plot_compare_al_methods_with_pretext():
#     # plot the accuracy against the al methods for sketch and imagenet using the pretext task
#     # one plot for each dataset type

#     epochs = 50
#     least_confidence_acc = []
#     entropy_acc = []
#     both = []

#     x_axis = [i for i in range(epochs)]
#     plt.plot(x_axis, least_confidence_acc, color ='red')
#     plt.plot(x_axis, entropy_acc, color ='blue')
#     plt.plot(x_axis, both, color ='orange')

#     # plt.plot(x2, y2, 'o-')

#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy")
#     plt.title("Clipart")
#     plt.legend(['Least Confidence', 'Entropy', 'Both'], loc='lower right')

#     filename = "save/clipart_all_methods.png"
#     plt.savefig(filename)
#     plt.show()

# def plot_compare_al_methods_without_pretext():
#     # plot the accuracy against the al methods for sketch and imagenet without using the pretext task
#     # one plot for each dataset type
#     None

# def plot_without_al():
#     epochs = 50
#     accuracies = []

#     x_axis = [i for i in range(epochs)]
#     plt.plot(x_axis, accuracies, color ='red')

#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy")
#     plt.title("Clipart")

#     filename = "save/clipart_none_method.png"
#     plt.savefig(filename)
#     plt.show()