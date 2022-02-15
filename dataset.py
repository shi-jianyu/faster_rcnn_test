""" This module contains the implementation of a custom dataset. """
import json
import os

import PIL
import torch
import torchvision

import utils


# Class implementation
class AlliumDataset(torch.utils.data.Dataset):
    """
    A custom dataset. The constructor gets a file with VIA-annotations and
    creates a dataset appropriate for the PyTorch Faster R-CNN implementation.

    ...

    Attributes
    ----------
    img_path: str
        Path to a folder with images.
    width: int
        The width of the input images in pixels.
    height: int
        The height of the input images in pixels.
    transform:
        A transfrom from the torchvision.transforms.transforms module.
    all_images: list
        The name of all images in the `img_path` folder.
    all_target: dict
        The targets (bounding boxes, labels, and parameters) for all images
        in the `img_path` folder.

    Overloaded methods
    ------------------
    __len__
    __getitem__
    """
    def __init__(self,
                 settings: dict,
                 mode: str,
                 transform=torchvision.transforms.ToTensor()):
        if mode == 'train':
            self.anns_path = settings['train_annotations']
            self.img_path = settings['train_images']
        elif mode == 'val':
            self.anns_path = settings['val_annotations']
            self.img_path = settings['val_images']
        else:
            print('Error: invalid mode')
            return None

        self.width = settings['width']
        self.height = settings['height']
        self.transform = transform

        self.all_images = [file for file in os.listdir(self.img_path)
                           if os.path.splitext(file)[-1] == '.jpg']

        # Loading VIA annotations from a file
        with open(self.anns_path, 'r') as read_file:
            original_data = json.load(read_file)

        # Creating an appropriate dataset containing only the necessary info
        self.all_targets = {}

        for image in original_data.keys():
            # Skip empty images
            if len(original_data[image]['regions']) == 0:
                self.all_images.remove(original_data[image]['filename'])
                print('There is the empty image %s. Skipping...' % 
                      original_data[image]['filename'])
                continue

            boxes = []
            labels = []

            for region in original_data[image]['regions']:
                # Add a `polygon` region
                if region['shape_attributes']['name'] == 'polygon':
                    # Converting a polygon into a bounding box
                    box = utils.get_bbox_from_poly(
                        region['shape_attributes']['all_points_x'],
                        region['shape_attributes']['all_points_y'])

                # Add a `rect` region
                elif region['shape_attributes']['name'] == 'rect':
                    box = utils.change_bbox_format(
                        region['shape_attributes']['x'],
                        region['shape_attributes']['y'],
                        region['shape_attributes']['width'],
                        region['shape_attributes']['height'])

                # Add the label of the region as an integer
                # In our case: "region_attribures": {"cell_type": "dividing"}
                label = settings['classes'].index(
                                list(region['region_attributes'].values())[0])

                # Attaching the bounding box
                if box in boxes:
                    print('DUBLICATE BOX in ',
                          original_data[image]['filename'],
                          '. It wasn\'t attached!')
                else:
                    boxes.append(box)
                    labels.append(label)

            # Convert the data to tensors
            self.all_targets[original_data[image]['filename']] = {
                'boxes': torch.as_tensor(boxes, dtype=torch.float32),
                'labels': torch.as_tensor(labels, dtype=torch.int64),
                'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)}

        print('Dataset is created!')

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index: int):
        """ Returns an image and corresponding annotations. """
        # Loading an image
        image_name = self.all_images[index]
        image = PIL.Image.open(
            os.path.join(self.img_path, image_name)).convert('RGB')
        image_width, image_height = image.size

        # Resizing the image
        image = image.resize((self.width, self.height),
                             resample=PIL.Image.BILINEAR)

        # Loading annotations
        target = self.all_targets[image_name].copy()
        new_boxes = []
        areas = []

        # Resizing a bounding box and calculating its area
        for box in target['boxes']:
            new_box = [box[0] / image_width * self.width,
                       box[1] / image_height * self.height,
                       box[2] / image_width * self.width,
                       box[3] / image_height * self.height]
            new_boxes.append(new_box)
            areas.append((new_box[2]-new_box[0]) * (new_box[3]-new_box[1]))

        target['image_id'] = torch.tensor([index], dtype=torch.int64)
        target['area'] = torch.as_tensor(areas, dtype=torch.float32)
        target['boxes'] = torch.as_tensor(new_boxes, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, target


# Main
if __name__ == '__main__':
    settings = utils.load_settings('settings.json')
    
    dataset = AlliumDataset(settings, 'train')
    
    utils.show_item(*dataset[4], utils.COLORS)
