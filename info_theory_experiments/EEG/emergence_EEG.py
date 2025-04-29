# %% initialization
import wandb
import tqdm
import torch

from info_theory_experiments.models import (
    DecoupledSmileMIEstimator,
    DownwardSmileMIEstimator,
    SkipConnectionSupervenientFeatureNetwork,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
print(device)

torch.manual_seed(0)

# %% define feature network
def train_feature_network(config, trainloader, feature_network):

    wandb.init(project="ecog-dataset-neurips", config=config)
    # init weights to zero of the feature network

    decoupled_MI_estimator = DecoupledSmileMIEstimator(
        feature_size=config["feature_size"],
        critic_output_size=config["critic_output_size"],
        hidden_sizes_1=config["decoupled_critic_hidden_sizes_1"],
        hidden_sizes_2=config["decoupled_critic_hidden_sizes_2"],
        clip=config["clip"],
        include_bias=config["bias"],
    ).to(device)
    downward_MI_estimators = [
        DownwardSmileMIEstimator(
            feature_size=config["feature_size"],
            critic_output_size=config["critic_output_size"],
            hidden_sizes_v_critic=config["downward_hidden_sizes_v_critic"],
            hidden_sizes_xi_critic=config["downward_hidden_sizes_xi_critic"],
            clip=config["clip"],
            include_bias=config["bias"],
        ).to(device)
        for _ in range(config["num_atoms"])
    ]

    feature_optimizer = torch.optim.Adam(
        feature_network.parameters(),
        lr=config["feature_lr"],
        weight_decay=config["weight_decay"],
    )
    decoupled_optimizer = torch.optim.Adam(
        decoupled_MI_estimator.parameters(),
        lr=config["decoupled_critic_lr"],
        weight_decay=config["weight_decay"],
    )
    downward_optims = [
        torch.optim.Adam(
            dc.parameters(),
            lr=config["downward_lr"],
            weight_decay=config["weight_decay"],
        )
        for dc in downward_MI_estimators
    ]

    wandb.watch(feature_network, log="all")
    wandb.watch(decoupled_MI_estimator, log="all")
    for dc in downward_MI_estimators:
        wandb.watch(dc, log="all")

    ##
    ## TRAIN FEATURE NETWORK
    ##

    epochs = config["epochs"]

    step = 0

    for _ in tqdm.tqdm(range(epochs), desc="Training"):
        for batch_num, batch in enumerate(trainloader):
            x0 = batch[:, 0].to(device).float()
            x1 = batch[:, 1].to(device).float()
            v0 = feature_network(x0).detach()
            v1 = feature_network(x1).detach()

            # update decoupled critic
            decoupled_optimizer.zero_grad()
            decoupled_MI = decoupled_MI_estimator(v0, v1)
            decoupled_loss = -decoupled_MI
            decoupled_loss.backward(retain_graph=True)
            decoupled_optimizer.step()

            # update each downward critic
            for i in range(config["num_atoms"]):
                downward_optims[i].zero_grad()
                channel_i = x0[:, i].unsqueeze(1).detach()
                downward_MI_i = downward_MI_estimators[i](v1, channel_i)
                downward_loss = -downward_MI_i
                downward_loss.backward(retain_graph=True)
                downward_optims[i].step()
                wandb.log({f"downward_MI_{i}": downward_MI_i}, step=step)

            # update feature network
            feature_optimizer.zero_grad()
            channel_MIs = []

            MIs = []
            v0 = feature_network(x0)
            v1 = feature_network(x1)

            for i in range(config["num_atoms"]):
                channel_i = x0[:, i].unsqueeze(1)
                channel_i_MI = downward_MI_estimators[i](v1, channel_i)
                channel_MIs.append(channel_i_MI)
                MIs.append(channel_i_MI)

            sum_downward_MI = sum(channel_MIs)

            decoupled_MI1 = decoupled_MI_estimator(v0, v1)

            clipped_min_MIs = max(0, min(MIs))

            Psi = (
                decoupled_MI1
                - sum_downward_MI
                + (config["num_atoms"] - 1) * clipped_min_MIs
            )

            # NOTE an experiment
            feature_loss = -Psi

            if config["start_updating_f_after"] < step:
                if batch_num % config["update_f_every_N_steps"] == 0:
                    feature_loss.backward(retain_graph=True)
                    feature_optimizer.step()

            wandb.log(
                {
                    "decoupled_MI": decoupled_MI1,
                    "sum_downward_MI": sum_downward_MI,
                    "Psi": Psi,
                },
                step=step,
            )

            step += 1

    torch.save(
        feature_network.state_dict(),
        f"/Users/saharsattari/Library/CloudStorage/OneDrive-UBC/Documents/Sahar/Brain states/causally-emergent-representations/models/eeg_feature_network_{wandb.run.name}.pth",
    )

    return feature_network


# %% configuration of networks
config = {
    "batch_size": 1000,
    "num_atoms": 10,
    "feature_size": 1,
    "clip": 5,
    "epochs": 50,
    "critic_output_size": 5,
    "downward_hidden_sizes_v_critic": [512, 512, 512, 512],
    "downward_hidden_sizes_xi_critic": [512, 512, 512],
    "feature_hidden_sizes": [256, 256, 256, 256, 256],
    "decoupled_critic_hidden_sizes_1": [512, 512, 512],
    "decoupled_critic_hidden_sizes_2": [512, 512, 512],
    "feature_lr": 1e-4,
    "decoupled_critic_lr": 1e-3,
    "downward_lr": 1e-3,
    "bias": True,
    "update_f_every_N_steps": 5,
    "weight_decay": 0,
    "start_updating_f_after": 300,
    "add_spec_norm_downward": False,
    "add_spec_norm_decoupled": False,
}

# %% create dataset
from info_theory_experiments.custom_datasets import EEGDataset

dataset = EEGDataset(
    file_path="/Users/saharsattari/Library/CloudStorage/OneDrive-UBC/Documents/Sahar/Brain states/causally-emergent-representations/eeg_data_sleep.npy",
)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)


feature_newtork = SkipConnectionSupervenientFeatureNetwork(
    num_atoms=config['num_atoms'],
    feature_size=config['feature_size'],
    hidden_sizes=config['feature_hidden_sizes'],
    include_bias=config['bias']
).to(device)

feature_network = train_feature_network(config, train_loader, feature_newtork)

# %% Looking at feature in time
# Rebuild the feature network
feature_network = SkipConnectionSupervenientFeatureNetwork(
    num_atoms=config['num_atoms'],         # 32
    feature_size=config['feature_size'],   # 3
    hidden_sizes=config['feature_hidden_sizes'],  # [256, 256, 256, 256, 256]
    include_bias=config['bias']             # True
).to(device)

# Load trained weights
feature_network.load_state_dict(
    torch.load(
        '/Users/saharsattari/Library/CloudStorage/OneDrive-UBC/Documents/Sahar/Brain states/causally-emergent-representations/models/eeg_feature_network_earthy-fog-2.pth',
        map_location=device   # ensures it loads to CPU or GPU correctly
    )
)
feature_network.eval()  # Important: turn off dropout etc.

import numpy as np
import torch

# Load your EEG .npy file
eeg_data = np.load('/Users/saharsattari/Library/CloudStorage/OneDrive-UBC/Documents/Sahar/Brain states/causally-emergent-representations/eeg_data_ec.npy')

# Convert to PyTorch tensor
eeg_data_tensor = torch.tensor(eeg_data, dtype=torch.float32).to(device)  # Shape: (channels, time)

# Be careful about dimension: you need (time, channels)
eeg_data_tensor = eeg_data_tensor.T  # (time, channels)

# Extract features V
with torch.no_grad():
    V = feature_network(eeg_data_tensor)  # (time, feature_size)

import matplotlib.pyplot as plt

V_np = V.cpu().numpy()

plt.figure(figsize=(15,5))
for i in range(V_np.shape[1]):  # feature_size = 3
    plt.plot(V_np[:, i], label=f'Feature {i+1}')
plt.legend()
plt.title('Learned Feature Trajectories Over Time')
plt.xlabel('Time')
plt.ylabel('Feature Value')
plt.show()
# %%
