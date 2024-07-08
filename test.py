import torch
from torchmdnet.models.model import create_model
from torchmdnet.models.model import load_model as original_load_model
from torchmdnet.datasets import QM9
from torchmdnet.utils import make_splits
import re

# get data
data_path = "qm9data"
dataset = QM9(data_path, label="energy_U0")

# # this should be the data they used for testing, the last 10,831 examples
# test_dataset = dataset[120000:]
# dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False) 
# # they used batch size of 128 here, but I get memory issues if I go above 32


from torchmdnet.data import DataModule
import yaml

with open('new.yaml', 'r') as file:
    hparams = yaml.safe_load(file)

module = DataModule(hparams=hparams, dataset=dataset)
module.setup(stage="test")
dataloader = module.test_dataloader()




# a custom version of load_model that changes some keys to prevent errors later on
def custom_load_model(filepath, args=None, device="cpu", return_std=False, **kwargs):
    # Load the model using the original load_model function to get the checkpoint
    ckpt = torch.load(filepath, map_location="cpu")

    if args is None:
        args = ckpt["hyper_parameters"]

    for key, value in kwargs.items():
        if key not in args:
            warnings.warn(f"Unknown hyperparameter: {key}={value}")
        args[key] = value

    # Create the model using the same arguments
    model = create_model(args)

    # Extract the state dictionary from the checkpoint
    state_dict = ckpt["state_dict"]

    # Adjust the state dictionary keys
    state_dict = {re.sub(r"^model\.", "", k): v for k, v in state_dict.items()}

    # Add specific adjustments for the key changes
    state_dict = {k.replace("prior_model.initial_atomref", "prior_model.0.initial_atomref"): v for k, v in state_dict.items()}
    state_dict = {k.replace("prior_model.atomref.weight", "prior_model.0.atomref.weight"): v for k, v in state_dict.items()}

    # Apply additional pattern replacements if necessary
    patterns = [
        (r"output_model.output_network.(\d+).update_net.(\d+).", r"output_model.output_network.\1.update_net.layers.\2."),
        (r"output_model.output_network.([02]).(weight|bias)", r"output_model.output_network.layers.\1.\2"),
    ]
    for p in patterns:
        state_dict = {re.sub(p[0], p[1], k): v for k, v in state_dict.items()}

    # Load the adjusted state dictionary into the model
    model.load_state_dict(state_dict)
    return model.to(device)

# Use the custom load model function
model = custom_load_model("et-qm9/epoch=649-val_loss=0.0003-test_loss=0.0059.ckpt", derivative=True)



# start evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

total_absolute_error = 0
total_samples = 0

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

for batch in dataloader:
    # Unpack batch data
    z = batch.z  # Node features (atomic numbers)
    pos = batch.pos  # Positions
    pos.requires_grad = True
    
    # Transfer data to device
    z = z.to(device)
    pos = pos.to(device)
    batch_indices = batch.batch.to(device)
    y = batch.y.to(device)
    
    # Run inference on the model
    energies, forces = model(z, pos, batch_indices)
    
    absolute_error = torch.abs(energies - y).sum().item()
    # print(str(absolute_error))
    total_absolute_error += absolute_error
    total_samples += energies.size(0)

mae = total_absolute_error / total_samples
print(f"Mean Absolute Error (MAE): {mae}")

#Mean Absolute Error (MAE): 0.005939372699023636
