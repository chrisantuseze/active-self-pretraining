import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# import seaborn as sns

def plot_classifier_with_and_without_pretext_weights():
    accuracies_with = [
            0.2568, 0.2733, 0.2864, 0.2926, 0.3048, 0.3025, 0.2973, 0.3245, 0.3267, 0.3179, 0.3134, 0.3222, 0.3283, 0.3249, 0.3258, 0.3251, 0.3281, 
            0.3399, 0.3236, 0.3385, 0.3402, 0.3317, 0.3415, 0.3394, 0.3423, 0.3431, 0.3354, 0.3429, 0.3543, 0.3415, 0.3514, 0.3449, 0.3524, 0.3446, 
            0.3541, 0.3424, 0.3561, 0.3569, 0.3475, 0.3528, 0.3499, 0.3545, 0.3593, 0.3535, 0.3524, 0.3548, 0.3557, 0.3544, 0.3596, 0.3575, 0.3678, 
            0.3613, 0.3541, 0.3590, 0.3580, 0.3606, 0.3621, 0.3664, 0.3535, 0.3602, 0.3700, 0.3666, 0.3650, 0.3622, 0.3612, 0.3693, 0.3624, 0.3641, 
            0.3589, 0.3771, 0.3617, 0.3735, 0.3720, 0.3658, 0.3728, 0.3647, 0.3678, 0.3680, 0.3697, 0.3711, 0.3718, 0.3744, 0.3642, 0.3724, 0.3725, 
            0.3713, 0.3700, 0.3833, 0.3692, 0.3759, 0.3717, 0.3748, 0.3659, 0.3765, 0.3781, 0.3814, 0.3765, 0.3811, 0.3729, 0.3724
        ]
    accuracies_without = [
            0.2784, 0.3030, 0.3153, 0.3451, 0.3318, 0.3300, 0.3383, 0.3597, 0.3472, 0.3611, 0.3615, 0.3550, 0.3640, 0.3743, 0.3779, 0.3804, 0.3796, 
            0.3767, 0.3845, 0.3860, 0.3945, 0.3853, 0.3970, 0.3892, 0.3881, 0.3891, 0.3985, 0.3879, 0.3964, 0.3978, 0.4031, 0.3990, 0.3914, 0.3935, 
            0.3833, 0.4029, 0.3887, 0.4024, 0.3966, 0.3906, 0.3958, 0.3830, 0.3974, 0.4080, 0.4044, 0.3920, 0.4134, 0.4094, 0.4014, 0.3972, 0.4088, 
            0.4029, 0.4011, 0.4099, 0.4001, 0.4191, 0.4008, 0.3954, 0.3936, 0.3966, 0.4062, 0.4041, 0.3873, 0.4168, 0.4037, 0.4123, 0.4028, 0.4072, 
            0.4085, 0.4105, 0.4098, 0.4008, 0.3913, 0.4066, 0.4037, 0.3986, 0.4076, 0.3993, 0.4035, 0.4096, 0.4055, 0.4043, 0.4057, 0.4131, 0.4033, 
            0.4133, 0.4058, 0.4143, 0.3941, 0.4065, 0.4150, 0.4035, 0.4121, 0.3982, 0.4051, 0.4124, 0.4069, 0.4131, 0.4140, 0.4045 
        ]

    x_axis = np.arange(len(accuracies_with))
    plt.plot(x_axis, accuracies_with, color ='red')
    plt.plot(x_axis, accuracies_without, color ='orange')

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Classifier fine-tuned on pretext task weights")
    plt.legend(['Finetuned', 'Not Finetuned'], loc='lower right')

    filename = "classifier_w_o_finetuning.png"
    plt.savefig(filename)
    plt.grid()

def plot_multi_data_regime():
    both_500 = 72.67 #[42.22, 61.11, 20.89, 23.11, 59.11, 58.44, 52.22, 42.22, 61.33, 38.67, 37.11, 45.33, 24.22, 37.33, 24.67, 54.67, 58.44, 58.22, 71.11, 44.44]
    both_5000 = 82.8 #[60.78, 65.58, 45.29, 47.71, 58.71, 58.84, 47.87, 67.84, 64.96, 60.16, 62.18, 67.91, 54.11, 51.16, 78.36, 49.0, 62.64, 50.93, 48.2, 54.6]
    both_15000 = 86.33 #[59.84, 44.296, 59.17, 61.50, 62.21, 55.94, 60.91, 72.82, 70.33, 72.69, 71.71, 54.71, 63.86, 44.57, 69.11, 72.93, 69.02, 56.73, 75.13, 77.97]
    both_30000 = 81.16 #[68.71, 67.54, 66.24, 77.015, 75.56, 74.49, 73.77, 75.59, 75.50, 74.61, 75.40, 70.6, 77.26, 76.79, 80.46, 72.35, 71.98, 67.21, 70.66, 73.84]
    both = [both_500, both_5000, both_15000, both_30000]

    entropy_500 = 76.33 #[60.0, 35.33, 69.78, 60.89, 66.0, 70.67, 43.56, 51.11, 36.67, 27.56, 39.33, 40.0, 53.56, 50.67, 18.89, 38.67, 57.33, 35.11, 30.44, 60.67]
    entropy_5000 = 78.8 #[30.82, 61.04, 52.04, 51.53, 48.13, 53.33, 60.0, 56.27, 49.36, 62.82, 50.64, 33.088, 78.8, 66.87, 60.47, 40.31, 44.18, 61.42, 73.42, 64.24]
    entropy_15000 = 80.93 #[57.77, 65.44, 39.19, 48.99, 66.51, 64.24, 70.88, 74.01, 68.12, 66.79, 74.51, 65.81, 70.067, 56.42, 71.83, 64.05, 59.02, 69.05, 76.67, 74.67]
    entropy_30000 = 82.61 #[77.57, 73.63, 80.64, 71.52, 72.47, 80.13, 80.75, 67.57, 77.19, 77.19, 75.21, 74.52, 80.037, 77.701, 77.84, 73.51, 74.17, 69.10, 80.09, 74.23]
    entropy = [entropy_500, entropy_5000, entropy_15000, entropy_30000]

    lc_500 = 80.89 #[21.11, 20.89, 33.11, 19.33, 16.22, 53.11, 67.56, 12.89, 30.44, 24.44, 49.78, 40.22, 34.0, 29.56, 45.11, 36.44]
    lc_5000 = 82.85 #[52.47, 59.44, 50.27, 74.47, 51.36, 35.76, 63.2, 45.088, 73.73, 49.67, 50.18, 52.76, 56.0, 68.76, 63.29, 80.58, 56.84, 61.53, 47.067, 41.02]
    lc_15000 = 81.88 #[69.35, 64.26, 71.59, 68.06, 76.59, 74.29, 78.02, 74.41, 60.24, 80.87, 81.88, 61.31, 73.49, 75.48, 74.71, 73.49, 60.24, 75.66, 73.67, 70.57]
    lc_30000 = 84.55 #[57.7, 70.56, 64.49, 70.83, 77.32, 69.9, 74.76, 74.21, 65.64, 66.029, 73.54, 77.04, 76.03, 74.63, 74.93, 75.56, 74.27, 67.38, 77.37, 80.41]
    lc = [lc_500, lc_5000, lc_15000, lc_30000]

    full_data = [42.23, 42.23, 42.23, 42.23]

    x_axis = [1, 10, 30, 60]
    plt.plot(x_axis, both, color ='red', marker='o', markersize=4)
    plt.plot(x_axis, entropy, color ='orange', marker='o', markersize=4)
    plt.plot(x_axis, lc, color ='blue', marker='o', markersize=4)
    plt.plot(x_axis, full_data, color ='red', linestyle='dashed', marker='o', markersize=4)

    plt.xlabel("Percentage of training data")
    plt.ylabel("Accuracy")
    plt.title("Main task on varying data regimes")
    plt.legend(['Both', 'Entropy', 'Least Confidence', 'Classfier'], loc='lower right')

    filename = "multi_data_regime.png"
    plt.savefig(filename)
    plt.grid()
    # plt.show()

def plot_main_task_wo_pretrained_weights():
    both_wo = 81.42 #[68.49, 73.76, 73.97, 74.33, 71.08, 71.03, 68.67, 73.35, 73.03, 76.29, 74.47, 75.23, 71.72, 76.83, 74.74, 74.60, 66.75, 74.77, 77.02, 74.61]
    both_w = 80.16 #[70.47, 75.15, 77.004, 76.64, 65.47, 75.62, 76.84, 75.42, 72.48, 78.93, 74.96, 69.067, 73.11, 73.96, 68.9, 73.12, 72.51, 74.32, 67.86, 76.04]

    entropy_wo = 80.84 #[62.03, 66.4, 70.25, 73.75, 72.02, 71.37, 71.89, 65.99, 74.78, 70.2, 72.59, 76.59, 74.44, 75.64, 70.26, 70.97, 72.17, 74.07, 70.51, 74.7]
    entropy_w = 80.93 #[65.67, 66.31, 64.16, 65.61, 73.71, 63.79, 75.29, 69.57, 66.71, 69.11, 72.70, 71.15, 75.98, 69.76, 77.63, 80.39, 72.71, 74.69, 71.47, 76.93]

    lc_wo = 81.29 #[76.98, 75.16, 77.86, 76.21, 66.37, 76.48, 78.21, 72.02, 75.79, 72.79, 79.31, 74.45, 71.39, 74.53, 76.06, 77.89, 77.34, 79.13, 75.34, 78.24]
    lc_w = 84.48 #[68.10, 77.05, 77.11, 75.89, 78.09, 71.99, 71.34, 73.96, 77.11, 77.46, 78.47, 74.62, 82.25, 82.62, 79.44, 79.32, 80.77, 80.97, 75.22, 79.57]

    x_axis = np.arange(2)
    width = 0.15

    plt.figure(figsize=(5, 5))

    lc = [lc_wo, lc_w]
    entropy = [entropy_wo, entropy_w]
    both = [both_wo, both_w]

    labels = ['Without weights', 'With weights']
    plt.xticks(x_axis + (width * 1), labels)
    
    plt.bar(x_axis, 
          lc, 
          width = width, 
          label = 'Least Confidence', 
          color = '#ff8e63')

    plt.bar(x_axis + width, 
            entropy, 
            width = width, 
            label = 'Entropy', 
            color = '#154360')

    plt.bar(x_axis + (width * 2), 
          both, 
          width = width, 
          label = 'Both', 
          color = '#0b8089')

    # plt.xlabel('Participants')
    plt.ylabel('Accuracy')

    plt.title("Training main task with and without pretrained weights")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06),
            fancybox=False, shadow=False, ncol=5)
    
    plt.grid(axis='y')
    filename = "wo_pretrained_weights.png"
    plt.savefig(filename)

def plot_compare_al_methods_with_pretext_cifar10():
    # plot the accuracy against the al methods for sketch and imagenet using the pretext task
    # one plot for each dataset type

    lc = [84.48, 84.55]
    entropy = [80.93, 82.61]
    both = [80.16, 81.16]
    no_al = [41.91, 42.23]

    x_axis = np.arange(2)
    width = 0.15

    plt.figure(figsize=(6, 6))

    labels = ['DCL', 'SimCLR']
    plt.xticks(x_axis + (width * 1.5), labels)
    
    plt.bar(x_axis, 
          lc, 
          width = width, 
          label = 'Least Confidence', 
          color = '#ff8e63')

    plt.bar(x_axis + width, 
            entropy, 
            width = width, 
            label = 'Entropy', 
            color = '#154360')

    plt.bar(x_axis + (width * 2), 
          both, 
          width = width, 
          label = 'Both', 
          color = '#0b8089')

    plt.bar(x_axis + (width * 3), 
          no_al, 
          width = width, 
          label = 'Baseline', 
          color = '#af3667')

    plt.xlabel('SSL Method')
    plt.ylabel('Accuracy')

    plt.title("Comparison between SSL Methods on Cifar10")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
            fancybox=False, shadow=False, ncol=5)
    
    plt.grid(axis='y')
    filename = "cifar_al_comparison.png"
    plt.savefig(filename)

def plot_compare_al_methods_with_pretext_imagenet():
    # plot the accuracy against the al methods for sketch and imagenet using the pretext task
    # one plot for each dataset type

    lc = [99.87, 97.2]
    entropy = [99.86, 98.2]
    both = [99.58, 81.0]
    no_al = [46.57, 42.85]

    x_axis = np.arange(2)
    width = 0.15

    plt.figure(figsize=(6, 6))

    labels = ['DCL', 'SimCLR']
    plt.xticks(x_axis + (width * 1.5), labels)
    
    plt.bar(x_axis, 
          lc, 
          width = width, 
          label = 'Least Confidence', 
          color = '#ff8e63')

    plt.bar(x_axis + width, 
            entropy, 
            width = width, 
            label = 'Entropy', 
            color = '#154360')

    plt.bar(x_axis + (width * 2), 
          both, 
          width = width, 
          label = 'Both', 
          color = '#0b8089')

    plt.bar(x_axis + (width * 3), 
          no_al, 
          width = width, 
          label = 'Baseline', 
          color = '#af3667')

    plt.xlabel('SSL Method')
    plt.ylabel('Accuracy')

    plt.title("Comparison between SSL Methods on ImageNet")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
            fancybox=False, shadow=False, ncol=5)
    
    plt.grid(axis='y')
    filename = "imagenet_al_comparison.png"
    plt.savefig(filename)

if __name__ == "__main__":
    plot_compare_al_methods_with_pretext_imagenet()