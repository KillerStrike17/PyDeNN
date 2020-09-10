import matplotlib.pyplot as plt
import math

def plot_metrics(train_acc, test_acc,train_loss,test_loss):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Metrics')
    axs[0, 0].plot(train_loss)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_loss)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
    plt.show()

def plot_curves_for_multi_experiments(*experiments):
    train_losses = [exp.history.train_losses for exp in experiments]
    train_accs = [exp.history.train_accs for exp in experiments]
    test_losses = [exp.history.test_losses for exp in experiments]
    test_accs = [exp.history.test_accs for exp in experiments]
    data = [train_losses, train_accs, test_losses, test_accs]
    titles = ["Train loss", "Train accuracy", "Test loss", "Test accuracy"]
    legends = [exp.name for exp in experiments]


    nrows = 2
    ncols = 2
    fig, ax = plt.subplots(nrows, ncols, figsize=(18, 13))

    for i in range(nrows):
        for j in range(ncols):
            index = i * ncols + j
            ax[i, j].set_title(titles[index])

            for k, legend in enumerate(legends):
                ax[i, j].plot(data[index][k], label=legend)

            ax[i, j].legend()
    plt.show()

def plot_misclassified(number, experiment, test_loader, device, save_path=None):
    image_data, predicted, actual = experiment.solver.get_misclassified(test_loader, device)
    nrows = math.floor(math.sqrt(number))
    ncols = math.ceil(math.sqrt(number))

    fig, ax = plt.subplots(nrows, ncols, figsize=(10, 15))

    for i in range(nrows):
        for j in range(ncols):
            index = i * ncols + j
            ax[i, j].axis("off")
            ax[i, j].set_title("Predicted: %d\nActual: %d" % (predicted[index], actual[index]))
            ax[i, j].imshow(image_data[index].cpu().numpy(), cmap="gray_r")

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0.25)