import matplotlib.pyplot as plt

def plot_training_history(history):
    plt.plot(history.history['accuracy'], label='train corr')
    plt.plot(history.history['val_accuracy'], label='acc corr')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('training_plot.png')