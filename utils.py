import os
import time
from datetime import timedelta

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np


class History:
    """This class represents the training history of a model. It can load the
       prior history when training continues, keeps track of the training and
       validation error, and finally plots them as a curve after each epoch.
    """

    def __init__(self, n_train_batches, n_valid_batches,
                 dataset, path, device):
        self.train_history = []
        self.valid_history = []

        self._prior_epochs = 0

        self._train_error = 0
        self._valid_error = 0

        self._n_train_batches = n_train_batches
        self._n_valid_batches = n_valid_batches

        self._path = path
        self._id = (dataset, device)

        self._get_prior_history()

    def _get_prior_history(self):
        if os.path.isfile(self._path + "train_%s_%s.txt" % self._id):
            with open(self._path + "train_%s_%s.txt" % self._id, "r") as file:
                for line in file.readlines():
                    self.train_history.append(float(line))

        if os.path.isfile(self._path + "valid_%s_%s.txt" % self._id):
            with open(self._path + "valid_%s_%s.txt" % self._id, "r") as file:
                for line in file.readlines():
                    self.valid_history.append(float(line))

        self.prior_epochs = len(self.train_history)

    def update_train_step(self, train_error):
        self._train_error += train_error

    def update_valid_step(self, valid_error):
        self._valid_error += valid_error

    def get_mean_train_error(self, reset=True):
        mean_train_error = self._train_error / self._n_train_batches

        if reset:
            self._train_error = 0

        return mean_train_error

    def get_mean_valid_error(self, reset=True):
        mean_valid_error = self._valid_error / self._n_valid_batches

        if reset:
            self._valid_error = 0

        return mean_valid_error

    def save_history(self):
        mean_train_loss = self.get_mean_train_error(False)
        mean_valid_loss = self.get_mean_valid_error(False)

        self.train_history.append(mean_train_loss)
        self.valid_history.append(mean_valid_loss)

        os.makedirs(self._path, exist_ok=True)

        with open(self._path + "train_%s_%s.txt" % self._id, "a") as file:
            file.write("%f\n" % self.train_history[-1])

        with open(self._path + "valid_%s_%s.txt" % self._id, "a") as file:
            file.write("%f\n" % self.valid_history[-1])

        if len(self.train_history) > 1:
            axes = plt.figure().gca()

            x_range = np.arange(1, len(self.train_history) + 1)

            plt.plot(x_range, self.train_history, label="train", linewidth=2)
            plt.plot(x_range, self.valid_history, label="valid", linewidth=2)

            plt.legend()
            plt.xlabel("epochs")
            plt.ylabel("error")

            locations = plticker.MultipleLocator(base=1.0)
            axes.xaxis.set_major_locator(locations)

            plt.savefig(self._path + "curve_%s_%s.png" % self._id)
            plt.close()


class Progbar:
    """This class represents a progress bar for the terminal that visualizes
       the training progress for each epoch, estimated time of accomplishment,
       and then summarizes the training and validation loss together with the
       elapsed time.
    """

    def __init__(self, n_train_data, n_train_batches,
                 batch_size, n_epochs, prior_epochs):
        self._train_time = 0
        self._valid_time = 0

        self._start_time = time.time()

        self._batch_size = batch_size

        self._n_train_data = n_train_data
        self._n_train_batches = n_train_batches

        self._target_epoch = str(n_epochs + prior_epochs).zfill(2)
        self._current_epoch = str(prior_epochs + 1).zfill(2)

    def _flush(self):
        self._train_time = 0
        self._valid_time = 0

        self._start_time = time.time()

        current_epoch_int = int(self._current_epoch) + 1
        self._current_epoch = str(current_epoch_int).zfill(2)

    def update_train_step(self, current_batch):
        current_batch += 1

        self._train_time = time.time() - self._start_time
        batch_train_time = self._train_time / current_batch

        eta = (self._n_train_batches - current_batch) * batch_train_time
        eta = str(timedelta(seconds=np.ceil(eta)))

        progress_line = "=" * (20 * current_batch // self._n_train_batches)

        current_instance = current_batch * self._batch_size
        current_instance = np.clip(current_instance, 0, self._n_train_data)

        progress_frac = "%i/%i" % (current_instance, self._n_train_data)

        information = (self._current_epoch, self._target_epoch,
                       progress_line, progress_frac, eta)

        progbar_output = "Epoch %s/%s [%-20s] %s (ETA: %s)" % information

        print(progbar_output, end="\r", flush=True)

    def update_valid_step(self):
        self._valid_time = time.time() - self._start_time - self._train_time

    def write_summary(self, mean_train_loss, mean_valid_loss):
        train_time = str(timedelta(seconds=np.ceil(self._train_time)))
        valid_time = str(timedelta(seconds=np.ceil(self._valid_time)))

        train_information = (mean_train_loss, train_time)
        valid_information = (mean_valid_loss, valid_time)

        train_output = "\n\tTrain loss: %.6f (%s)" % train_information
        valid_output = "\tValid loss: %.6f (%s)" % valid_information

        print(train_output, flush=True)
        print(valid_output, flush=True)

        self._flush()
