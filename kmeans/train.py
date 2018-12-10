import os

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plot

import constants
from cluster.kmeans import KMeans
from data_loader import DataLoader
from utils import file_utils
from utils.log_utils import LOGGER
from utils.shell_args import SHELL_ARGS


class Train(object):
    def __init__(self, prefix, initializer, epochs):
        self.prefix = prefix
        self.epochs = epochs
        self.initializer = initializer
        self.data_loader = DataLoader(self.prefix)
        self.kmeans = KMeans(self.data_loader.n_clusters, init=initializer)

    def visualize_result(self, predicted_labels, epoch_i):
        plot.clf()

        pca = PCA(n_components=2)
        reduced_source_data = pca.fit_transform(self.data_loader.source_data)
        centers = pca.transform(self.kmeans.centers)

        colors = ['blue', 'red', 'green']
        markers = ['o', 'x', '^']
        target_data = self.data_loader.target_data.reshape(-1)
        for label_id in range(self.data_loader.n_clusters):
            condition = target_data == label_id

            source_data = reduced_source_data[condition]
            labeld_predicted_labels = predicted_labels[condition]

            for predicted_label_id in range(self.data_loader.n_clusters):
                predicted_source_data = source_data[labeld_predicted_labels == predicted_label_id]
                plot.scatter(predicted_source_data[:, 0],
                             predicted_source_data[:, 1],
                             c=colors[predicted_label_id],
                             marker=markers[label_id],
                             label=label_id)

        plot.scatter(centers[:, 0], centers[:, 1], linewidths=6, c='black')
        plot.savefig(os.path.join(constants.OUTPUT_DIR, self.prefix, epoch_i, 'result.png'))

    def visualize_loss(self, sse_list, epoch_i):
        plot.clf()

        plot.plot(range(len(sse_list)), sse_list)
        plot.savefig(os.path.join(constants.OUTPUT_DIR, self.prefix, epoch_i, 'loss.png'))

    def train(self):
        LOGGER.info('TRAINING START, prefix: {}, initializer: {}, epochs: {}'.format(self.prefix,
                                                                                     self.initializer,
                                                                                     self.epochs))
        for epoch_i in range(self.epochs):
            LOGGER.info('Training epoch: {}'.format(epoch_i))
            predicted_labels, sse_list = self.kmeans.fit(self.data_loader.source_data)

            file_utils.safe_mkdirs(os.path.join(constants.OUTPUT_DIR, self.prefix, str(epoch_i)))
            self.visualize_result(predicted_labels, str(epoch_i))
            self.visualize_loss(sse_list, str(epoch_i))
        LOGGER.info('TRAINING END\n')


def main():
    train = Train(SHELL_ARGS.prefix, SHELL_ARGS.initializer, SHELL_ARGS.epochs)
    train.train()


if __name__ == '__main__':
    main()
