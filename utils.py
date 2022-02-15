""" This module contains helper classes, functions, and variables.

--- Variables ---
    COLORS

--- Classes ---
    LossAverager

--- Functions ---
    change_bbox_format
    collate_fn
    create_model_legacy
    get_bbox_from_poly
    load_settings
    show_item
"""
import json
import typing

import PIL
import torch
import torchvision
import torchvision.models.detection as detection


# Variables
COLORS = [(0, 0, 0),
          (225, 225, 0),
          (0, 225, 0)]


# Classes
class LossAverager:
    """ A helper class.

    Keeps track of the training and validation loss values and also
    averages them for each epoch.

    ...

    Attributes
    ----------
    total_loss: float
        The sum of losses for all passed iterations.
    iterations: int
        The number of iterations passed.

    Methods
    -------
    update(value: float)
        Updates the attributes.
    average()
        Returns the average loss.
    reset()
        Resets the attributes.
    """
    def __init__(self):
        self.total_loss = 0.0
        self.iterations = 0

    @property
    def average(self) -> float:
        """ Returns the average loss. """
        if self.iterations == 0:
            return 0
        else:
            return self.total_loss / self.iterations

    def reset(self) -> None:
        """ Resets the attributes. """
        self.total_loss = 0.0
        self.iterations = 0.0

    def update(self, value: float):
        """ Updates the attributes.

        Parameters
        ----------
        value: float
            The loss value for the current iteration.
        """
        self.total_loss += value
        self.iterations += 1


# Functions
def change_bbox_format(x: int, y: int, width: int, height: int) -> list:
    """ Changes the format of a bounding box.

    Converts a bounding box in the format [x1, y1, width, height] into
    a bounding box in the format [x1, y1, x2, y2].

    Parameters
    ----------
    x: int
        The X coordinate of the top-left corner of a bounding box.
    y: int
        The Y coordinate of the top-left corner of a bounding box.
    width: int
        The width of a bounding box.
    height: int
        The height of a bounding box.

    Returns
    -------
    list
        A bounding box in the format [x1, y1, x2, y2].
    """
    return [x, y, x+width, y+height]


def collate_fn(batch) -> tuple:
    """ Collates images with corresponding annotations. """
    return tuple(zip(*batch))


def create_model_legacy(num_classes: int) -> detection.faster_rcnn.FasterRCNN:
    """ Creates a Faster R-CNN model.

        Parameters
        ----------
        num_classes: int
            The number of classes.

        Returns
        -------
        model: torchvision.models.detectio.faster_rcnn.FasterRCNN
            The created model.
    """
    # loading a Faster RCNN pre-trained model
    model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # Get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(
                                    in_features, num_classes)
    print('Model is created!')
    return model


def create_model(num_classes: int) -> detection.faster_rcnn.FasterRCNN:
    """ Creates a Faster R-CNN model.

        Parameters
        ----------
        num_classes: int
            The number of classes.

        Returns
        -------
        model: torchvision.models.detectio.faster_rcnn.FasterRCNN
            The created model.
    """
    backbone = detection.backbone_utils.resnet_fpn_backbone('resnet101',
                                                            pretrained=True)
    
    anchor_sizes = ((128,), (256,), (512,), (1024,), (2048,), ) 
    aspect_ratios = ((1.0, 1.25, 1.5, 1.75, 2.0, 2.5),) * len(anchor_sizes) 
    
    anchor_generator = detection.rpn.AnchorGenerator(
                                   sizes=anchor_sizes,
                                   aspect_ratios=aspect_ratios)
    model = detection.FasterRCNN(backbone, 
                                 num_classes=num_classes,
                                 rpn_anchor_generator=anchor_generator)
    print('Model is created!')
    return model


def get_bbox_from_poly(all_points_x: list, all_points_y: list) -> list:
    """ Converts a polygon mask into a bounding box.

    Parameters
    ----------
    all_points_x: list
        The X coordinates of all the vertices of the polygon.
    all_points_y: list
        The Y coordinates of all the vertices of the polygon.

    Returns
    -------
    list
        A bounding box in the format [x1, y1, x2, y2].
    """
    return [min(all_points_x), min(all_points_y),
            max(all_points_x), max(all_points_y)]


def load_settings(settings_path: str) -> dict:
    """ Loads settings from the specified file.

    Parameters
    ----------
    settings_path: str
        The path to the file.

    Returns
    -------
    settings: dict
        The loaded settings.
    """
    with open(settings_path, 'r') as read_file:
        settings = json.load(read_file)

    settings['num_classes'] = len(settings['classes'])

    return settings


def show_item(image: torch.Tensor, target: dict,
              colors: typing.Iterable) -> None:
    """ Saves the selected dataset image to a file. 
    
    Saves the selected image with drawn annotations to the file `test.jpg`.
    
    Parameters
    ----------
    image: torch.Tensor
        The selected image.
    target: dict
        The corresponding annotations.
    colors: typing.Iterable
        The list of colors.
    """
    image = torchvision.transforms.ToPILImage()(image)
    
    image_draw = PIL.ImageDraw.Draw(image)
    
    # Drawing bounding boxes in the image
    # Different colors for different classes
    boxes = target["boxes"].numpy()
    labels = target["labels"].numpy()
    
    for i in range(len(boxes)):
        image_draw.rectangle(boxes[i], outline=colors[labels[i]], width=1)
    
    image.save("test.jpg")
