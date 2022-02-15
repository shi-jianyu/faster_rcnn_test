import time

import matplotlib.pyplot as plt
import torch
import torchvision
import tqdm

import utils
import dataset


# Function implementations
def train(device: torch.device,
          dataloader: torch.torch.utils.data.DataLoader,
          model: torchvision.models.detection.faster_rcnn.FasterRCNN,
          optimizer: torch.optim.Optimizer,
          train_loss: utils.LossAverager) -> list:
    """ Runs training iterations.
    
    Parameters
    ----------


    Returns
    -------
    """
    print('Training')
    global train_itr
    global train_loss_list

    # Initialize a tqdm progress bar
    prog_bar = tqdm.tqdm(dataloader, total=len(dataloader))

    # Learning through the entire training dataset
    for _, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item() 
        train_loss_list.append(loss_value)
        train_loss.update(loss_value)

        losses.backward()
        optimizer.step()

        train_itr += 1

        # Update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f'Loss: {loss_value:.4f}')

    return train_loss_list


def validate(device: torch.device,
          dataloader: torch.torch.utils.data.DataLoader,
          model: torchvision.models.detection.faster_rcnn.FasterRCNN,
          val_loss: utils.LossAverager) -> list:
    """ Runs validation iterations.
    
    Parameters
    ----------

    Returns
    -------
    """
    print('Validating')
    global val_itr
    global val_loss_list

    # Initialize a tqdm progress bar
    prog_bar = tqdm.tqdm(dataloader, total=len(dataloader))

    # Validating through the entire validation dataset
    for _, data in enumerate(prog_bar):
        images, targets = data

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)
        val_loss.update(loss_value)

        val_itr += 1

        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f'Loss: {loss_value:.4f}')

    return val_loss_list


# Main
if __name__ == '__main__':
    settings = utils.load_settings('settings.json')
    device = torch.device('cuda:0')

    # Loading data
    train_dataset = dataset.AlliumDataset(settings, 'train')
    val_dataset = dataset.AlliumDataset(settings, 'val')

    # Creating dataloaders
    train_dataloader = torch.torch.utils.data.DataLoader(train_dataset,
        batch_size=settings["batch_size"], shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)
    val_dataloader = torch.torch.utils.data.DataLoader(val_dataset,
        batch_size=settings["batch_size"], shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    # Initializing the model and moving it to the computation device
    model = utils.create_model(settings['num_classes'])
    model = model.to(device)
    
    if settings['weights'] is not None:
        model.load_state_dict(torch.load(
                              settings["weights"], map_location=device))
        print('Loaded weights from %s' % settings['weights'])

    # get the model parameters
    params = [p for p in model.parameters() if p.requires_grad]

    # Defining the optimizer
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9,
                weight_decay=0.0005)

    # Initialize the LossAverager class
    train_loss_hist = utils.LossAverager()
    val_loss_hist = utils.LossAverager()

    train_itr = 1
    val_itr = 1

    train_loss_list = []
    val_loss_list = []

    # Starting the learning process
    for epoch in range(settings['num_epochs']):
        print(f'EPOCH {epoch+1} of ' + str(settings['num_epochs']))
        
        # Reset the training and validation loss
        train_loss_hist.reset()
        val_loss_hist.reset()
        
        # Create two subplots, one for each training and validation
        figure_1, train_ax = plt.subplots()
        figure_2, valid_ax = plt.subplots()
        
        # Starting timer and carry out training and validation
        start = time.time()
        
        train_loss = train(device, train_dataloader, model, optimizer, 
                           train_loss_hist)
        val_loss = validate(device, val_dataloader, model, val_loss_hist)
        print(f'Epoch #{epoch} train loss: {train_loss_hist.average:.3f}')
        print(f'Epoch #{epoch} validation loss: {val_loss_hist.average:.3f}')
        
        end = time.time()
        
        print(f'Took {((end-start)/60):.3f} minutes for epoch {epoch}')

        # Save model after every n epochs
        if (epoch + 1) % settings['save_model_epochs'] == 0:
            torch.save(model.state_dict(), settings['output_folder']+
                       f'/model{epoch+1}.pth')
            print('Saving the model complete...')

        # Save loss plots after n epochs
        if (epoch + 1) % settings['save_plot_epochs'] == 0:
            train_ax.plot(train_loss, color='blue')
            train_ax.set_xlabel('iterations')
            train_ax.set_ylabel('train loss')
            valid_ax.plot(val_loss, color='red')
            valid_ax.set_xlabel('iterations')
            valid_ax.set_ylabel('validation loss')
            figure_1.savefig(settings['output_folder'] +
                             f'/train_loss_{epoch+1}.png')
            figure_2.savefig(settings["output_folder"] +
                             f"/valid_loss_{epoch+1}.png")
            print('Saving the plots complete...')

        # save loss plots and model once at the end
        if (epoch + 1) == settings['num_epochs']:
            train_ax.plot(train_loss, color='blue')
            train_ax.set_xlabel('iterations')
            train_ax.set_ylabel('train loss')
            valid_ax.plot(val_loss, color='red')
            valid_ax.set_xlabel('iterations')
            valid_ax.set_ylabel('validation loss')
            figure_1.savefig(settings['output_folder'] +
                             f"/train_loss_end.png")
            figure_2.savefig(settings['output_folder'] +
                             f"/valid_loss_end.png")

            torch.save(model.state_dict(),
                       settings['output_folder']+ f'/model_end.pth')

        plt.close('all')
