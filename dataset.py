import json
import os
from PIL import Image
import torch, torchvision

import utils


class AlliumDataset(torch.utils.data.Dataset):
	""" """
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
		
		# adding images
		for image in original_data.keys():
			boxes = []
			labels = []
			
			if len(original_data[image]["regions"]) == 0:
				self.all_images.remove(original_data[image]["filename"])
				print("Empty image. Skip.")
				continue
			
			for region in original_data[image]["regions"]:
				# adding polygon regions
				if region["shape_attributes"]["name"] == "polygon":
					box = utils.get_bbox(region["shape_attributes"]["all_points_x"],
						region["shape_attributes"]["all_points_y"])
					if len(box) < 4:
						print("ERROR: " + original_data[image]["filename"] + " small bbox!")
						continue
				# adding rect regions
				elif region["shape_attributes"]["name"] == "rect":
					box = utils.parse_bbox(
						region["shape_attributes"]["x"],
						region["shape_attributes"]["y"],
						region["shape_attributes"]["width"],
						region["shape_attributes"]["height"]
						)

				# adding labels
				label = list(region["region_attributes"].values())[0]
				label = settings["classes"].index(label)

				# attaching bbox
				if box in boxes:
					print("DUBLICATE BOX in " + original_data[image]["filename"] + ". Don't attach")
				else:
					boxes.append(box)
					labels.append(label)

			boxes = torch.as_tensor(boxes, dtype=torch.float32)
			# for box in boxes:
			# 	if len(box) < 4:
			# 		print("ERROR: " + original_data[image]["filename"] + " small bbox! (1)")
			
			self.all_targets[original_data[image]["filename"]] = {
				"boxes": boxes,
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
		new_boxes = []
		areas = []
		for box in target["boxes"]:
			new_box = []
			new_box.append(box[0]/image_width*self.width)
			new_box.append(box[1]/image_height*self.height)
			new_box.append(box[2]/image_width*self.width)
			new_box.append(box[3]/image_height*self.height)
			area = (new_box[2]-new_box[0]) * (new_box[3]-new_box[1])
			if len(new_box) < 4:
				print("ERROR: " + image_name + " small bbox! (2)")
			new_boxes.append(new_box)
			areas.append(area)
		image_id = torch.tensor([index], dtype=torch.int64)
		target["image_id"] = image_id
		target["area"] = torch.as_tensor(areas, dtype=torch.float32)
		target["boxes"] = torch.as_tensor(new_boxes, dtype=torch.float32)


		if self.transform:
			image = self.transform(image)

		return image, target


if __name__ == "__main__":
	settings = utils.parse_json("settings.json")
	dataset = AlliumDataset(settings["train_annotations"], 
		settings["train_images"], settings["width"], settings["height"],
		settings, torchvision.transforms.ToTensor())
	
	# for i in range(len(dataset)):
	# 	image, target = dataset[i]
	# 	for box in target["boxes"]:
	# 		if len(box) < 4:
	# 			print("ERROR: " + original_data[image]["filename"] + "маленький bbox! (4)")
	# 	print(f"Проверено {i+1} / {len(dataset)}", end="\r")

	image, target = dataset[667]
	utils.show_item(image, target, utils.COLORS)
