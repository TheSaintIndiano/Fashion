__author__ = 'indiano'

import itertools

import matplotlib.pyplot as plt
import numpy as np

# seaborn for beautiful plots.
plt.style.use('seaborn-bright')


class Plot:

    def __init__(self):
        pass

    def plotly(self, cnf_matrix, classification_report, save_dir, model_name):
        """
        plotly function plots confusion matrix (normalized also)

        :param cnf_matrix:np.array
        :param save_dir:str
        :param cm_folder:str
        :param model_name:str
        :return:
        """

        # Plot non-normalized confusion matrix
        self.plot_confusion_matrix(cnf_matrix, target_names=['NadaSportswear', 'Sportswear'],
                                   title='Confusion matrix, without normalization',
                                   save_dir=save_dir, plt_name=model_name)

        # Plot normalized confusion matrix
        self.plot_confusion_matrix(cnf_matrix, target_names=['NadaSportswear', 'Sportswear'], normalize=True,
                                   title='Normalized confusion matrix', save_dir=save_dir,
                                   plt_name=model_name)

        # Plot classification report containing precision, recall, f1 score
        self.plot_classification_report(classification_report, save_dir=save_dir,
                                        plt_name=model_name)

    def show_values(self, pc, fmt="%.2f", **kw):
        """
        :param pc:collection
        :param fmt:str
        :param kw:
        :return:

        show_values function displays heatmap with text in each cell.
        """

        pc.update_scalarmappable()
        ax = pc.axes
        for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
            x, y = p.vertices[:-2, :].mean(0)
            if np.all(color[:3] > 0.5):
                color = (0.0, 0.0, 0.0)
            else:
                color = (1.0, 1.0, 1.0)
            ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)

    def cm2inch(self, *tupl):
        """
        :param tupl:
        :return:

        cm2inch function is used to specify figure size in centimeter in matplotlib.
        """

        inch = 2.54
        if type(tupl[0]) == tuple:
            return tuple(i / inch for i in tupl[0])
        else:
            return tuple(i / inch for i in tupl)

    def heatmap(self, AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=20, figure_height=20,
                correct_orientation=False, cmap=None):
        """
        :param AUC:
        :param title:
        :param xlabel:
        :param ylabel:
        :param xticklabels:
        :param yticklabels:
        :param figure_width:int
        :param figure_height:int
        :param correct_orientation:
        :param cmap:
        :return:

        heatmap function plots the heatmap of the classification report.
        """

        # Plot it out
        fig, ax = plt.subplots()
        c = ax.pcolor(AUC, edgecolors='k', linestyle='dashed', linewidths=0.2, cmap=cmap)

        # put the major ticks at the middle of each cell
        ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
        ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

        # set tick labels
        ax.set_xticklabels(xticklabels, minor=False)
        ax.set_yticklabels(yticklabels, minor=False)

        # set title and x/y labels
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # Remove last blank column
        plt.xlim((0, AUC.shape[1]))

        # Turn off all the ticks
        ax = plt.gca()
        for t in ax.xaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        for t in ax.yaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False

        # Add color bar
        plt.colorbar(c)

        # Add text in each cell
        self.show_values(c)

        # Proper orientation (origin at the top left instead of bottom left)
        if correct_orientation:
            ax.invert_yaxis()
            ax.xaxis.tick_top()

        # resize
        fig = plt.gcf()
        fig.set_size_inches(self.cm2inch(figure_width, figure_height))

    def plot_classification_report(self, classification_report,
                                   plt_name,
                                   rootdir='./',
                                   save_dir='save/',
                                   title='Classification report ', cmap=None):
        """
        plot_classification_report function plots classification report.

        :param classification_report:np.array
        :param title:str
        :param cmap:
        :return:
        """

        if cmap is None:
            cmap = plt.get_cmap('Oranges')

        lines = classification_report.split('\n')

        classes = []
        plotMat = []
        support = []
        class_names = []
        for line in lines[2: (len(lines) - 2)]:
            t = line.strip().split()
            if len(t) < 2: continue
            classes.append(t[0])
            v = [float(x) for x in t[1: len(t) - 1]]
            support.append(int(t[-1]))
            class_names.append(t[0])
            print(v)
            plotMat.append(v)

        print('Plot Matrix: {0}'.format(plotMat))
        print('Support: {0}'.format(support))

        xlabel = 'Metrics'
        ylabel = 'Classes'
        xticklabels = ['Precision', 'Recall', 'F1-score']
        yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup in enumerate(support)]
        figure_width = 15
        figure_height = len(class_names) + 7
        correct_orientation = False
        self.heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height,
                     correct_orientation, cmap=cmap)

        print('\nSaving Classification Report in the {} directory'.format(rootdir + save_dir))
        plt.savefig(rootdir + save_dir + '/{}_ClassificationReport.png'.format(plt_name), dpi=200, format='png',
                    bbox_inches='tight')

        plt.show()
        plt.close()

    def plot_confusion_matrix(self, cm,
                              target_names,
                              plt_name,
                              rootdir='./',
                              save_dir='save/',
                              title='Confusion matrix',
                              cmap=None,
                              normalize=False):
        """
        plot_confusion_matrix function prints & plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.

        :param cm:confusion matrix from sklearn.metrics.confusion_matrix
        :param target_names:classification classes list eg. [0, 1] ['high', 'medium', 'low']
        :param rootdir:str
        :param save_dir:str
        :param plt_name:str
        :param title:str
        :param cmap:color map list
        :param normalize:bool
        :return:
        """

        plt_name += '_ConfusionMatrix'
        if normalize:
            plt_name = '{}_Normalized'.format(plt_name)

        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\n\nAccuracy={:0.4f}; Misclassified={:0.4f}'.format(accuracy, misclass))

        print('Saving Confusion Matrices in the {} directory'.format(rootdir + save_dir))
        plt.savefig(rootdir + save_dir + '/{}.png'.format(plt_name), dpi=200, format='png', bbox_inches='tight')

        plt.show()
        plt.close()
