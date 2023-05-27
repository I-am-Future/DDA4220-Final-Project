import re
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Garamond", "Palatino"]
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "Calibri"]
plt.rcParams["font.size"] = 18


def extract_critical_role(file_name, pattern):
    with open(file_name, 'r') as file:
        content = file.readlines()
    print('total lines in logging:', len(content))
    extracted_numbers = []
    for line in content:
        match = re.search(pattern, line)
        if match:
            extracted_numbers.append(float(match.group(1)))
    
    ndarray = np.array(extracted_numbers)
    return ndarray

pattern = r'\sAverage Precision  \(AP\) @\[ IoU=0\.50\s*\|\s*area=\s*all\s*\|\s*maxDets=100 \] = (\d+\.\d+)'


def plot_multiple_val_acc(val_acc_lr_decay_list, fname: str='exps/val.png', legends=None, show_best_in_fig=False):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot each validation accuracy curve and corresponding dashed lines
    for idx, (val_acc, lr_decay_epochs) in enumerate(val_acc_lr_decay_list):
        # Plot the validation accuracy curve
        epochs = np.arange(1, len(val_acc) + 1)
        ax.plot(epochs, val_acc, alpha=0.8, linewidth=2, label=f"Validation Accuracy {idx+1}")

        # Find the largest accuracy from the last several epochs
        last_several_epochs = 5
        max_acc = np.max(val_acc[-last_several_epochs:])
        max_acc_epoch = np.argmax(val_acc[-last_several_epochs:]) + len(val_acc) - last_several_epochs + 1

        text = f"{max_acc:.3f}"
        if show_best_in_fig:
            # Add a textbox with the largest accuracy
            # There is two experience value, to get modified with.
            ax.text(max_acc_epoch - last_several_epochs - 8, max_acc, text, fontsize=12, verticalalignment='top', backgroundcolor='none')
        elif legends:
            # legends[idx] = legends[idx] + f' ({text})'
            ...
    # Set axis labels and title
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Validation Accuracy vs Epochs")

    # Add a legend
    if legends:
        ax.legend(legends)
    ax.set_ylim([0, 0.7])

    for idx, (val_acc, lr_decay_epochs) in enumerate(val_acc_lr_decay_list):
        # Draw vertical dashed lines at learning rate decay epochs
        for lr_decay_epoch in lr_decay_epochs:
            y_value = val_acc[lr_decay_epoch - 1]
            ax.plot([lr_decay_epoch, lr_decay_epoch], [0, y_value], linestyle='--', color='red')

    # Display the plot
    plt.savefig(fname)


def plot_multiple_gradnorm(gradnorm_list, fname: str='exps/gradnorm.png', legends=None, show_best_in_fig=False):
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot each validation accuracy curve and corresponding dashed lines
    for idx, gradnorm in enumerate(gradnorm_list):
        # Plot the validation accuracy curve
        Batches = np.arange(1, len(gradnorm) + 1)
        ax.plot(Batches, gradnorm, label=f"Validation Accuracy {idx+1}")

    # Set axis labels and title
    ax.set_xlabel("Batches")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Validation Accuracy vs Batches")

    # Add a legend
    if legends:
        ax.legend(legends)

    # Display the plot
    plt.savefig(fname)