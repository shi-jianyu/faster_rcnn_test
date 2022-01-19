import matplotlib.pyplot as plt
import time
import torch, torchvision
import tqdm

import utils
import dataset

settings = utils.parse_json("settings.json")
device = torch.device('cuda:0')


def train(device, train_data_loader, model, optimizer):
	""" function for running training iterations """
	print("Training")
	global train_itr
	global train_loss_list

	# initiate tqdm progress bar
	prog_bar = tqdm.tqdm(train_data_loader, total=len(train_data_loader))

	for i, data in enumerate(prog_bar):
		optimizer.zero_grad()
		images, targets = data

		images = list(image.to(device) for image in images)
		targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

		loss_dict = model(images, targets)

		losses = sum(loss for loss in loss_dict.values())
		loss_value = losses.item()
		train_loss_list.append(loss_value)

		train_loss_hist.send(loss_value)

		losses.backward()
		optimizer.step()

		train_itr += 1

		# update the loss value beside the progress bar for each iteration
		prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

	return train_loss_list


def validate(device, valid_data_loader, model):
	""" function for running validation iterations """

	print("Validating")
	global val_itr
	global val_loss_list

	# initialize tqdm progress bar
	prog_bar = tqdm.tqdm(valid_data_loader, total=len(valid_data_loader))

	for i, data in enumerate(prog_bar):
		images, targets = data

		images = list(image.to(device) for image in images)
		targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

		with torch.no_grad():
			loss_dict = model(images, targets)

		losses = sum(loss for loss in loss_dict.values())
		loss_value = losses.item()
		val_loss_list.append(loss_value)

		val_loss_hist.send(loss_value)

		val_itr += 1

		# update the loss value beside the progress bar for each iteration
		prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

	return val_loss_list


train_dataset = dataset.AlliumDataset(settings["train_annotations"], 
		settings["train_images"], settings["width"], settings["height"],
		settings, torchvision.transforms.ToTensor())

val_dataset = dataset.AlliumDataset(settings["val_annotations"], 
		settings["val_images"], settings["width"], settings["height"],
		settings, torchvision.transforms.ToTensor())

train_dataloader = torch.torch.utils.data.DataLoader(train_dataset,
	batch_size=settings["batch_size"], shuffle=True, num_workers=0,
	collate_fn=utils.collate_fn)
val_dataloader = torch.torch.utils.data.DataLoader(val_dataset,
	batch_size=settings["batch_size"], shuffle=False, num_workers=0,
	collate_fn=utils.collate_fn)


# initialize the model and move to the computation device
model = utils.create_model(settings["num_classes"])
model = model.to(device)
if settings["weights"] is not None:
	model.load_state_dict(torch.load(
	settings["weights"], map_location=device))
	print("Loaded weights from " + settings["weights"])
# get the model parameters
params = [p for p in model.parameters() if p.requires_grad]

# define the optimizer
if settings["optimizer"] == "SGD":
	optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9,
			weight_decay=0.0001)
else:
	optimizer = torch.optim.Adam(params, lr=0.001)


#initialize the Averager class
train_loss_hist = utils.Averager()
val_loss_hist = utils.Averager()
train_itr = 1
val_itr = 1
train_loss_list = []
val_loss_list = []

# name to save the trained model with
MODEL_NAME = settings["model_name"]


# start the training epochs
for epoch in range(settings["num_epochs"]):
	print(f"EPOCH {epoch+1} of " + str(settings["num_epochs"]))
	# reset the training and validation loss histories for 
	# the current epoch
	train_loss_hist.reset()
	val_loss_hist.reset()
	# create two subplots, one for each training and validation
	figure_1, train_ax = plt.subplots()
	figure_2, valid_ax = plt.subplots()
	# start timer and carry out training and validation
	start = time.time()
	train_loss = train(device, train_dataloader, model, optimizer)
	val_loss = validate(device, val_dataloader, model)
	print(f"Epoch #{epoch} train loss: {train_loss_hist.value:.3f}")
	print(f"Epoch #{epoch} validation loss: {val_loss_hist.value:.3f}")
	end = time.time()
	print(f"Took {((end-start)/60):.3f} minutes for epoch {epoch}")


	# save model after every n epochs
	if (epoch+1) % settings["save_model_epochs"] == 0:
		torch.save(model.state_dict(), settings["output_folder"]+f"/model{epoch+1}.pth")
		print("SAVING MODEL COMPLETE...")

	# save loss plots after n epochs
	if (epoch+1) % settings["save_plot_epochs"] == 0:
		train_ax.plot(train_loss, color="blue")
		train_ax.set_xlabel("iterations")
		train_ax.set_ylabel("train loss")
		valid_ax.plot(val_loss, color="red")
		valid_ax.set_xlabel("iterations")
		valid_ax.set_ylabel("validation loss")
		figure_1.savefig(settings["output_folder"]+f"/train_loss_{epoch+1}.png")
		figure_2.savefig(settings["output_folder"]+f"/valid_loss_{epoch+1}.png")
		print("SAVING PLOTS COMPLETE...")

	# save loss plots and model once at the end
	if (epoch+1) == settings["num_epochs"]:
		train_ax.plot(train_loss, color="blue")
		train_ax.set_xlabel("iterations")
		train_ax.set_ylabel("train loss")
		valid_ax.plot(val_loss, color="red")
		valid_ax.set_xlabel("iterations")
		valid_ax.set_ylabel("validation loss")
		figure_1.savefig(settings["output_folder"]+f"/train_loss_end.png")
		figure_2.savefig(settings["output_folder"]+f"/valid_loss_end.png")

		torch.save(model.state_dict(), settings["output_folder"]+f"/model_end.pth")

	plt.close("all")
