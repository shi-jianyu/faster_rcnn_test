import json
from configparser import ConfigParser
from PIL import Image, ImageDraw
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import tqdm
import random


COLORS = [
	(0, 0, 0),
	(225, 225, 0),
	(0, 225, 0),
]

class Averager:
	""" 
	Keep track of the training and validation loss values
	and also average them for each epoch.
	""" 
	def __init__(self):
		self.current_total = 0.0
		self.iterations = 0.0

	def send(self, value):
		self.current_total += value
		self.iterations += 1

	@property
	def value(self):
		if self.iterations == 0:
			return 0
		else:
			return 1.0 * self.current_total / self.iterations

	def reset(self):
		self.current_total = 0.0
		self.iterations = 0.0


def parse_ini(ini_path: str):
	""" """
	parser = ConfigParser()
	parser.read(ini_path)

	data = {}
	# parse the settings section
	data["width"] = parser.getint("settings", "width")
	data["height"] = parser.getint("settings", "height")
	data["batch_size"] = parser.getint("settings", "batch_size")
	data["num_epochs"] = parser.getint("settings", "num_epochs")
	# parse the classes section
	data["classes"] = parser.get("classes", "classes").split()
	data["num_classes"] = len(data["classes"])
	# parse the files section
	data["train_annotations"] = parser.get("files", "train_annotations")
	data["train_images"] = parser.get("files", "train_images")
	data["val_annotations"] = parser.get("files", "val_annotations")
	data["val_images"] = parser.get("files", "val_images")
	# parse the results section
	data["output_folder"] = parser.get("results", "output_folder")
	data["model_name"] = parser.get("results", "model_name")
	data["save_model_epochs"] = parser.getint("results", "save_model_epochs")
	data["save_plot_epochs"] = parser.getint("results", "save_plot_epochs")

	return data


def parse_json(json_path: str):
	""" """
	with open(json_path, "r") as read_file:
		data = json.load(read_file)

	data["num_classes"] = len(data["classes"])

	return data


def get_bbox(all_points_x: list, all_points_y: list):
	""" """
	x_min = min(all_points_x)
	y_min = min(all_points_y)
	x_max = max(all_points_x)
	y_max = max(all_points_y)

	return [x_min, y_min, x_max, y_max]


def parse_bbox(x: int, y: int, width: int, height: int):
	""" """
	return [x, y, x+width, y+height]


def collate_fn(batch):
	""" """
	return tuple(zip(*batch))


def show_item(image, target, colors):
	""" """
	image = torchvision.transforms.ToPILImage()(image)
	image_draw = ImageDraw.Draw(image)
	boxes = target["boxes"].numpy()
	labels = target["labels"].numpy()
	for i in range(len(boxes)):
		image_draw.rectangle(boxes[i], outline=colors[labels[i]], width=1)
	image.save("test.jpg")


def generate_colors(settings):
	""" """
	colors = []
	for i in range(settings["num_classes"]):
		colors.append(tuple(random.sample(range(0, 255), 3)))
	return colors


def create_model(num_classes):
	""" """
	# load Faster RCNN pre-trained model
	model = fasterrcnn_resnet50_fpn(pretrained=True)
	# get the number of input features
	in_features = model.roi_heads.box_predictor.cls_score.in_features
	# define a new head for the detector with required number of classes
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features,
			num_classes)
	print("Model is created!")

	return model


if __name__ == "__main__":
	settings = parse_ini("settings.ini")
	print(settings)
