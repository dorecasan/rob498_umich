import matplotlib.pyplot as plt
import numpngw
import cv2
import numpy as np

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

class ExtraNotebookVisualizer(object):
    def __init__(self, fig, hfig, filename = 'pushing_visualization.gif'):
        self.fig = fig
        self.hfig = hfig
        self.frames = []
        self.filename = filename

    def set_data(self, img, states):
        img = draw_boxes_with_lines(img,states)
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

def draw_boxes_with_lines(image, box_states):
    # Define box color and line color
    box_color = (0, 255, 0)  # Green color for boxes
    line_color = (0, 0, 255)  # Red color for lines

    # Iterate over box states
    for state in box_states:
        # Unpack box state
        x, y, theta, box_width = state

        # Calculate box corners
        half_width = box_width / 2
        corners = np.array([[-half_width, -half_width],
                            [-half_width, half_width],
                            [half_width, half_width],
                            [half_width, -half_width]])

        # Create rotation matrix
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotation_matrix = np.array([[cos_theta, -sin_theta],
                                    [sin_theta, cos_theta]])

        # Rotate and translate box corners
        rotated_corners = np.dot(rotation_matrix, corners.T).T
        translated_corners = rotated_corners + np.array([x, y])

        # Draw box
        cv2.polylines(image, [translated_corners.astype(np.int32)], True, box_color, 2)

        # Calculate center of the box
        center = np.mean(translated_corners, axis=0).astype(np.int32)

        # Draw line connecting the box center to a reference point (e.g., image center)
        reference_point = np.array([image.shape[1] // 2, image.shape[0] // 2])  # Use image center as reference
        cv2.line(image, tuple(center), tuple(reference_point), line_color, 2)

    return image



    