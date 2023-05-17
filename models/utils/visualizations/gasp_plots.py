import matplotlib.pyplot as plt
import numpy as np
import utils.logger as logging

dir = "model/utils/visualizations/plots"

def separability():
    # accuracy = [89.7, 89.51, 90.7, 91.53, 90.96] #chest
    # accuracy = [91.81, 93.07, 94.80, 93.41] #eurosat
    # accuracy = [27.0, 53.34, 54.98, 59.57, 55.32] #clipart
    accuracy = [24.0, 46.28, 49.84, 54.55, 50.61] #sketch
    # accuracy = [28.0, 33.33, 42.06, 42.99, 42.65] #quickdraw
    # accuracy = [93.0, 92.01, 91.9, 92.78, 91.20] #flowers
    # accuracy = [94.68, 95.65, 96.77, 94.68] #office
    # accuracy = [64.31, 62.81, 63.95, 63.21] #ham10000

    # x_axis = ['S', 'S-T', 'GASP', 'GASP+T']
    x_axis = ['Baseline', 'S', 'S-T', 'GASP', 'GASP+T']

    # colors = ['#698e77', '#ed9a68', '#5f81c2', '#c26b67']  # Colors for each bar
    colors = ['#e2d4b6', '#698e77', '#ed9a68', '#5f81c2', '#c26b67']  # Colors for each bar

    plt.figure(figsize=(5, 3))

    # Create the bar chart
    plt.bar(x_axis, accuracy, color=colors)

    plt.bar(x_axis[:], accuracy[:], color=colors[:], hatch='', edgecolor='black', linewidth=1)

    # Add hatching (strips) to the last two bars
    plt.bar(x_axis[-2:], accuracy[-2:], color=colors[-2:], hatch='/', edgecolor='black', linewidth=1)

    # Add labels and title
    plt.xlabel('Training Strategy')
    # plt.ylabel('Accuracy')
    # plt.title('Chest-Xray')
    # plt.title('EuroSAT')
    # plt.title('Clipart')
    plt.title('Sketch')
    # plt.title('Quickdraw')
    # plt.title('Flowers')
    # plt.title('Office-31')
    # plt.title('Ham10000')

    # Add text labels for accuracy values
    for i in range(len(x_axis)):
        plt.text(x_axis[i], accuracy[i] + 0.2, f'{accuracy[i]}%', ha='center', color='black')

    # Add arrows to show difference between accuracy values
    for i in range(len(x_axis)-1):
        plt.annotate('', xy=(x_axis[i], accuracy[i]), xycoords='data',
                    xytext=(x_axis[i+1], accuracy[i+1]), textcoords='data',
                    arrowprops=dict(arrowstyle='->', color='black'))

    # Set y-axis limit to show bar lengths clearly
    plt.ylim(min(accuracy) - 1, max(accuracy) + 0.5)

    plt.savefig(f'{dir}/separability.png')
    logging.info("Plot saved.")

def data_regime():
    #30%, 60%, 90%

    office = [95.32, 95.87, 96.77]
    office_t = [94.84, 95.16, 95.32]

    eurosat = [94.31, 94.59, 94.80]
    eurosat_t = [93.67, 92.91, 93.41]

    ham10000 = [62.21, 62.21, 63.95]
    ham10000_t = [62.16, 61.81, 63.16]

    gasp = ham10000
    max_gasp = max(gasp)

    gasp_t = ham10000_t
    max_gasp_t = max(gasp_t)

    x_axis = [30, 60, 90]

    max_gasp_x_axis = x_axis[gasp.index(max_gasp)]
    max_gasp_t_x_axis = x_axis[gasp_t.index(max_gasp_t)]

    plt.figure(figsize=(5, 3))

    plt.plot(x_axis, gasp, color ='#5f81c2', marker='o', markersize=4)
    plt.plot(x_axis, gasp_t, color ='#c26b67', marker='*', markersize=4)
    # plt.plot(x_axis, bt, color ='#ed9a68', linestyle='dashed', marker='*', markersize=4)

    plt.xlabel("Percentage of Pretrain Data")
    # plt.ylabel("Accuracy")
    # plt.title("Office-31")
    # plt.title("EuroSAT")
    plt.title("Ham10000")
    plt.legend(['GASP', 'GASP+T'], loc='center right')

    # Mark the highest point on the y-axis
    plt.plot(max_gasp_x_axis, max_gasp, marker='o', markersize=8, color='#5f81c2')
    plt.annotate(f'{max_gasp}', xy=(max_gasp_x_axis, max_gasp), xytext=(max_gasp_x_axis+2, max_gasp), color='black')

    plt.plot(max_gasp_t_x_axis, max_gasp_t, marker='*', markersize=8, color='#c26b67')
    plt.annotate(f'{max_gasp_t}', xy=(max_gasp_t_x_axis, max_gasp_t), xytext=(max_gasp_t_x_axis, max_gasp_t+0.1), color='black')

    plt.savefig(f'{dir}/data_regime.png')
    logging.info("Plot saved.")

def replace_gan_with_proxy():
    # replace GAN with proxy for gradual pretraining

    # accuracy = [88.82, 89.51, 89.42, 91.53] #chest
    # accuracy = [50.00, 91.81, 93.72, 94.80] #eurosat
    # accuracy = [59.46, 64.31, 62.41, 63.95] #ham10000
    # accuracy = [53.13, 92.01, 90.86, 92.78] #flowers
    accuracy = [66.76, 94.68, 95.13, 96.77] #office

    x_axis = ['Random Init.', 'S', 'GASP - P', 'GASP - G']

    colors = ['#a7a7a4', '#698f78', '#e2d4b6', '#5f81c2']  # Colors for each bar

    plt.figure(figsize=(4, 2))

    # Create the bar chart
    plt.bar(x_axis, accuracy, color=colors)

    plt.bar(x_axis[:], accuracy[:], color=colors[:], hatch='', edgecolor='black', linewidth=1)

    # Add hatching (strips) to the last two bars
    plt.bar(x_axis[-1:], accuracy[-1:], color=colors[-1:], hatch='/', edgecolor='black', linewidth=1)

    # Add labels and title
    plt.xlabel('Training Strategy')
    # plt.ylabel('Accuracy')
    # plt.title('Chest-Xray')
    # plt.title('EuroSAT')
    # plt.title('Ham10000')
    # plt.title('Flowers')
    plt.title('Office-31')

    # Add text labels for accuracy values
    for i in range(len(x_axis)):
        plt.text(x_axis[i], accuracy[i] + 0.2, f'{accuracy[i]}%', ha='center', color='black')

    # Add arrows to show difference between accuracy values
    for i in range(len(x_axis)-1):
        plt.annotate('', xy=(x_axis[i], accuracy[i]), xycoords='data',
                    xytext=(x_axis[i+1], accuracy[i+1]), textcoords='data',
                    arrowprops=dict(arrowstyle='->', color='black'))

    # Set y-axis limit to show bar lengths clearly
    plt.ylim(min(accuracy) - 2.5, max(accuracy) + 2.5) #_proxy_source_for_gan

    plt.savefig(f'{dir}/replace_gan_with_proxy.png')
    logging.info("Plot saved.")

