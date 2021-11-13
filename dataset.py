import json
import os
from PIL import Image
import torch, torchvision

import utils


class AlliumDataset(torch.utils.data.Dataset):
	def __init__(self, ann_path, img_path, width, height, 
		settings, transform=None):
		self.img_path  = img_path
		self.width     = width
		self.height    = height
		self.transform = transform
		# list of all image names
		self.all_images = []
		for file in os.listdir(self.img_path):
			if os.path.splitext(file)[-1] == ".jpg":
				self.all_images.append(file)
		# dict with targets for all images
		with open(ann_path, "r") as read_file:
			original_data = json.load(read_file)
		self.all_targets = {}
		for image in original_data.keys():
			boxes = []
			labels = []
			for region in original_data[image]["regions"]:
				box = utils.get_bbox(region["shape_attributes"]["all_points_x"],
					region["shape_attributes"]["all_points_y"])
				label = list(region["region_attributes"].values())[0]
				label = settings["classes"].index(label)

				boxes.append(box)
				labels.append(label)
			self.all_targets[original_data[image]["filename"]] = {
				"boxes": torch.as_tensor(boxes, dtype=torch.float32),
				"labels": torch.as_tensor(labels, dtype=torch.int64),
			    "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64)}
		print("Dataset is created!")

	def __len__(self):
		return len(self.all_images)

	def __getitem__(self, index):
		image_name = self.all_images[index]
		image_path = os.path.join(self.img_path, image_name)
		image = Image.open(image_path).convert("RGB")
		image_width, image_height = image.size
		image = image.resize((self.width, self.height), 
			resample=Image.BILINEAR)

		target = self.all_targets[image_name].copy()
		areas = []
		for box in target["boxes"]:
			box[0] = box[0]/image_width*self.width
			box[1] = box[1]/image_height*self.height
			box[2] = box[2]/image_width*self.width
			box[3] = box[3]/image_height*self.height
			area = (box[2]-box[0]) * (box[3]-box[1])
			areas.append(area)
		image_id = torch.tensor([index], dtype=torch.int64)
		target["image_id"] = image_id
		target["area"] = torch.as_tensor(areas, dtype=torch.float32)

		if self.transform:
			image = self.transform(image)

		return image, target


if __name__ == "__main__":
	settings = utils.parse_ini("settings.ini")
	dataset = AlliumDataset(settings["train_annotations"], 
		settings["train_images"], settings["width"], settings["height"],
		settings, torchvision.transforms.ToTensor())
	image, target = dataset[120]
	utils.show_item(image, target, utils.COLORS)
