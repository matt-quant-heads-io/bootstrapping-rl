import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import numpy as np
import torch
from tqdm import tqdm
from einops import rearrange
import torch.nn as nn
import json
import torch as th

import utils


from stable_baselines3 import PPO
import torch.nn.functional as F


# from policy import TrainerCombinedCustomFeatureExtractorCustomPolicy, CustomCNNPolicy, CustomFeatureExtractorCNN, LegacyCombineActorCriticNetwork, CustomActorCriticPolicy



class CNNLarger(nn.Module):
    def __init__(self, device):
        super(CNNLarger, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=8, out_channels=64, kernel_size=(3,3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), padding=1)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 8)
        self.pool2d = nn.MaxPool2d(kernel_size=(2, 2))
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten(start_dim=1)
        self.device = device

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool2d(x)
        x = self.relu(self.conv2(x))
        x = self.pool2d(x)
        x = self.relu(self.conv3(x))
        x = self.pool2d(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x






BATCH_SIZE = 32
EPOCHS = 1000
EPISODES_FOR_PPO_TRAINING = 1_000_000

def train_combined_feature_extractor_and_policy(experiment):
    kwargs = {
        "change_percentage": 1.0,
        "trials": 1000,
        "verbose": False,
        "experiment": experiment
    }
    env = utils.make_vec_envs("zelda-narrow-v0", "narrow", None, 1, **kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # checkpoint = torch.load(f"/home/jupyter-msiper/bootstrapping_rl/cnn_larger_simple_model.pt", weights_only=True)
    net = CNNLarger(device)
    net.to(device)
    # net.load_state_dict(checkpoint['policy_state_dict'])
    # optimizer = optim.Adam(net.parameters(), lr=0.0003)
    optimizer = optim.SGD(net.parameters(), lr=0.0003, momentum=0.9)
    # optimizer.load_state_dict(checkpoint['optimizer'])
    print(f"Using device: {device}")
    
    # acc_metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=8).to(device)
    loss_function_critic = nn.MSELoss()
    loss_function_actor = nn.CrossEntropyLoss() 

    # data = np.load("/home/jupyter-msiper/bootstrapping_rl/small_expert_dataset.npz")
    data = np.load("/home/jupyter-msiper/bootstrapping-rl/goal_maps/expert_dataset.npz")
    
    
    # data = np.load("/home/jupyter-msiper/arl/expert_data.npz")
    obss_array = data["expert_observations"].astype(np.float32)
    target_array_critic_head = data["expert_rewards"].astype(np.float32)
    target_array_actor_head = data["expert_actions"].astype(np.float32)
    # import pdb 
    # pdb.set_trace()

    X = torch.Tensor([i for i in obss_array]).to(device).view(-1,11,11,8).to(device)
    y_critic = torch.Tensor([i for i in target_array_critic_head]).to(device)
    y_actor = torch.Tensor([i for i in target_array_actor_head]).to(device)
    
    X = X.to(device)
    y_critic = y_critic.to(device)
    y_actor = y_actor.to(device)
    
    TRAIN_PCT = 0.9
    val_size = int(len(X)*TRAIN_PCT)
    
    train_X = X[:val_size].to(device)
    train_y_critic = y_critic[:val_size].to(device)
    train_y_actor = y_actor[:val_size].to(device)
    
    test_X = X[val_size:].to(device)
    test_y_critic = y_critic[val_size:].to(device)
    test_y_actor = y_actor[val_size:].to(device)

    print(len(train_X), len(test_X))
    best_loss = np.inf
    ebootstrapping_rly_stop = False

    H = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    # best_acc = 0.9207287 #-np.inf
    best_acc = 0.93 #-np.inf
    
    for epoch in range(EPOCHS):
        net.train()

        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0
        # initialize the number of correct predictions in the training
        # and validation step
        trainCorrect = 0
        valCorrect = 0
        steps_per_epoch = 0
        
        for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
            batch_X = train_X[i:i+BATCH_SIZE].view(-1, 11, 11, 8).to(device)
            batch_y_actor = train_y_actor[i:i+BATCH_SIZE].to(device)


            optimizer = optim.SGD(net.parameters(), lr=0.0003, momentum=0.9)
            batch_X = rearrange(batch_X.to(device), 'b w h c -> b c w h').to(device)
            
            outputs_actor = net(batch_X.contiguous().view(-1, 8, 11, 11).to(device)).to(device)
            loss = loss_function_actor(outputs_actor.to(device), batch_y_actor.to(device)).to(device)
            
            # loss = -(acc_metric(outputs_actor.softmax(dim=1), torch.argmax(batch_y_actor, dim=1)))
            # loss = (outputs_actor.argmax(1) == batch_y_actor.argmax(1)).type(torch.float).sum() 
            # loss.requires_grad = True
            # print(f'loss: {loss}')
            import pdb
            # pdb.set_trace()

            net.zero_grad()
            loss.backward()
            optimizer.step()   
            
            totalTrainLoss += loss
            trainCorrect += (outputs_actor.argmax(1) == batch_y_actor.argmax(1)).type(torch.float).sum().item()
            steps_per_epoch += len(batch_y_actor)

        curr_acc = trainCorrect/steps_per_epoch
    
        print(f"Epoch: {epoch}. acc %: {curr_acc} Loss: {totalTrainLoss/steps_per_epoch}")
        if curr_acc > best_acc:
            print(f"Saving new best ActorCritic!")
            torch.save({
                'policy_state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            },f"/home/jupyter-msiper/bootstrapping-rl/cnn_larger_simple_model2.pt")
            # torch.save({
            #     'policy_state_dict': net.state_dict(),
            #     'optimizer': optimizer.state_dict(),
            # },f"/home/jupyter-msiper/bootstrapping_rl/simple_model.pt")
            
            best_acc = curr_acc
            if best_acc >= 0.9999:
                print(f"best loss hit threshold!")
                ebootstrapping_rly_stop = True
                break
        if ebootstrapping_rly_stop:
            break
    
        correct = 0
        total = 0
        errors = [] 
        
        with torch.no_grad():
            net.eval()
            steps_per_epoch = 0
            for i in tqdm(range(0, len(test_X), BATCH_SIZE)):
                tbatch_X = test_X[i:i+BATCH_SIZE].view(-1, 11, 11, 8).to(device)
                # batch_X = rearrange(batch_X, 'b c h w -> b w h c')
                tbatch_y_critic = test_y_critic[i:i+BATCH_SIZE].to(device)
                tbatch_y_actor = test_y_actor[i:i+BATCH_SIZE].to(device)
                # real_class = torch.argmax(test_y_actor[i])
                tbatch_X = rearrange(tbatch_X, 'b w h c -> b c w h').to(device)

                # toutputs_actor, toutputs_critic = net(tbatch_X)
                toutputs_actor = net(tbatch_X).to(device)
                # tloss = torch.stack([loss_function_critic(toutputs_critic, tbatch_y_critic), loss_function_actor(toutputs_actor, tbatch_y_actor)]).sum()
                tloss = loss_function_actor(toutputs_actor.to(device), tbatch_y_actor.to(device)).to(device)
                
                totalValLoss += tloss
                # calculate the number of correct predictions
                valCorrect += (toutputs_actor.argmax(1) == tbatch_y_actor.argmax(1)).type(torch.float).sum().item()
                steps_per_epoch += len(tbatch_y_actor)
                
        print(f"test acc %: {valCorrect/steps_per_epoch} test loss: {totalValLoss}")


experiment = "experiment_1"
train_combined_feature_extractor_and_policy(experiment)