import matplotlib.pyplot as plt
import numpngw


class GIFVisualizer(object):
    def __init__(self):
        self.frames = []

    def set_data(self, img):
        self.frames.append(img)

    def reset(self):
        self.frames = []

    def get_gif(self):
        # generate the gif
        filename = 'pushing_visualization.gif'
        print("Creating animated gif, please wait about 10 seconds")
        numpngw.write_apng(filename, self.frames, delay=10)
        return filename


class NotebookVisualizer(object):
    def __init__(self, fig, hfig, filename = 'pushing_visualization.gif'):
        self.fig = fig
        self.hfig = hfig
        self.frames = []
        self.filename = filename

    def set_data(self, img):
        self.frames.append(img)
        plt.clf()
        plt.imshow(img)
        plt.axis('off')
        self.fig.canvas.draw()
        self.hfig.update(self.fig)

    def reset(self):
        self.frames = []

    def get_gif(self):
        # generate the gif
        print("Creating animated gif, please wait about 10 seconds")
        numpngw.write_apng(self.filename, self.frames, delay=10)
        return self.filename