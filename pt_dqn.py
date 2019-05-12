# PyTorch DQN for Flappy Bird

import random
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2

sys.path.append('game/')
from wrapped_flappy_bird import GameState

class NeuralNetwork(nn.Module):
    """Neural network doc string"""
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # Hyperparameters
        self.num_actions = 2
        self.gamma = 0.99
        self.epsilon_final = 0.0001
        self.epsilon_initial = 0.1
        self.iterations = 100000
        self.memory_size = 10000
        self.minibatch = 32

        # Network layers: 3 convolution, 2 fully connected
        # Note: Output = (Input - Filter + 2*Padding)/Stride + 1
        # Channels are like number of filters
        self.conv1 = nn.Conv2d(in_channels=4,out_channels=32,kernel_size=8,stride=4)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=4,stride=2)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1)
        self.fc4 = nn.Linear(3136, 512)
        self.fc5 = nn.Linear(512, self.num_actions)

        # Forward pass
        # Relu activation
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        # Flatten tensor for pooling layer
        # -1 allows the size to be informed by the other arguments
        out = out.view(out.size()[0], -1)
        out = F.relu(self.fc4(out))
        out = self.fc5(out)
        return out

# Initialize (neuron) weights and introduce bias (independent neuron; no input from prev layer)
def init_weights(layer):
    if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
        # Use following sytax: torch.nn.init.distribution_(Tensor, args*)
        # Initialize weights randomly from uniform (-0.01, 0.01)
        torch.nn.init.uniform_(layer.weight, -0.01, 0.01)
        # Initialize bias as constant 0.01
        torch.nn.init.constant_(layer.bias, 0.01)

# Take image of game state, convert to grayscale and resize to 84x84
def resize_and_bgr2gray(image):
    image = image[0:288, 0:404]
    image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
    image_data[image_data > 0] = 255
    image_data = np.reshape(image_data, (84, 84, 1))
    return image_data

def image_to_tensor(image):
    image_tensor = image.transpose(2, 0, 1) # Shape 1x84x84
    image_tensor = image_tensor.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)
    if torch.cuda.is_available():  # put on GPU if CUDA is available
        image_tensor = image_tensor.cuda()
    return image_tensor

def train(model, start):
    # Optimizer, learning rate 1e-6
    optimizer = optim.Adam(model.parameters(), lr = 1e-6)

    # Use MSE loss
    criterion = nn.MSELoss()

    # Start the game
    game = GameState()

    # Replay memory, minibatching speeds training and prevents overfitting
    memory = []

    # Initial game state, initial action is do nothing
    # Action = [1,0] is do nothing, Action = [0,1] is flap
    action = torch.zeros([model.num_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal = game.frame_step(action)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)

    # Concatenate last 4 images together
    # Unsqueeze adds a dimension of size 1 to the specific location (dim)
    state = torch.unsqueeze(torch.cat((image_data, image_data, image_data, image_data)), dim=0)

    # Initialize epsilon, then anneal linearly over number of iterations
    epsilon = model.epsilon_initial
    epsilons = np.linspace(model.epsilon_initial, model.epsilon_final, model.iterations)

    # MAIN LOOP
    iteration = 0
    while iteration < model.iterations:
        # Simply feed the game state (past 4 frames) into model and get output
        output = model(state)[0] # Index 0 to get the action

        # Initialize action
        action = torch.zeros([model.num_actions], dtype=torch.float32)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action = action.cuda()

        # Exploration. Instead of best action, perform random action at rate epsilon
        action_index = [torch.randint(model.num_actions, torch.Size([]), dtype=torch.int)
                        if random.random() < epsilon
                        else torch.argmax(output)][0]

        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action_index = action_index.cuda()

        action[action_index] = 1

        # Get next state, reward, and terminal (game over or not)
        image_data1, reward, terminal = game.frame_step(action)
        image_data1 = resize_and_bgr2gray(image_data1)
        image_data1 = image_to_tensor(image_data1)
        # Next state by dropping oldest frame then concatenating newest frame
        state1 = torch.unsqueeze(torch.cat((torch.squeeze(state, 0)[1:, :, :], image_data1)), 0)
        # Unsqueeze action to same dimension
        action = torch.unsqueeze(action, 0)
        # Reward as tensor
        reward = torch.unsqueeze(torch.from_numpy(np.array([reward], dtype = np.float32)), 0)

        # Add transition (state, action, reward, next state, terminal)
        memory.append((state, action, reward, state1, terminal))

        # Only save 10000 states in replay memory
        if len(memory) > model.memory_size:
            del memory[0]

        # Anneal epsilon
        epsilon = epsilons[iteration]

        # Sample minibatch from memory
        minibatch = random.sample(memory, min(len(memory), model.minibatch)) # Min if memory doesn't have enough states yet
        # Unpack minibatch
        # for s in minibatch: # This was stupid. Use list comprehension instead.
        #     state_batch = torch.cat(tuple(s[0]))
        #     action_batch = torch.cat(tuple(s[1]))
        #     reward_batch = torch.cat(tuple(s[2]))
        #     state1_batch = torch.cat(tuple(s[3]))
        state_batch = torch.cat(tuple(d[0] for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        state1_batch = torch.cat(tuple(d[3] for d in minibatch))
        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            state1_batch = state1_batch.cuda()

        # Output of next state
        output1_batch = model(state1_batch)

        # If j is terminal, set y_j = reward_j
        # Else y_j = r_j + gamma*max(Q) discounted future rewards
        y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                  else reward_batch[i] + model.gamma * torch.max(output1_batch[i])
                                  for i in range(len(minibatch))))
        # Get the expected reward for the action that provides max reward

        # Find Q-value (maximum discounted future reward, when action a is performed in state s)
        q_value = torch.sum(model(state_batch) * action_batch, dim = 1)

        # Reset gradients before doing back propogation
        optimizer.zero_grad()

        # Detach tensor, will never require gradient
        y_batch.detach_() # Equivalent below
        # y_batch = y_batch.detach()

        # Calculate loss, which we then do back propogation on
        loss = criterion(q_value, y_batch)

        # Backwards pass
        loss.backward()
        optimizer.step()

        # Update state to be next state, increase iteration
        state = state1
        iteration += 1

        # Some convenience prints, save model every 25000 iterations
        if iteration % 25000 == 0:
            torch.save(model, "pretrained_model/current_model_" + str(iteration) + ".pth")

        print("iteration:", iteration, "elapsed time:", time.time() - start, "epsilon:", epsilon, "action:",
              action_index.cpu().detach().numpy(), "reward:", reward.numpy()[0][0], "Q max:",
              np.max(output.cpu().detach().numpy()))

def test(model):
    game_state = GameState()

    # initial action is do nothing
    action = torch.zeros([model.num_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal = game_state.frame_step(action)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    while True:
        # get output from the neural network
        output = model(state)[0]

        action = torch.zeros([model.num_actions], dtype=torch.float32)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action = action.cuda()

        # get action
        action_index = torch.argmax(output)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action_index = action_index.cuda()
        action[action_index] = 1

        # get next state
        image_data_1, reward, terminal = game_state.frame_step(action)
        image_data_1 = resize_and_bgr2gray(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        # set state to be state_1
        state = state_1

def main(mode):
    cuda_is_available = torch.cuda.is_available()

    if mode == 'test':
        model = torch.load(
            'pretrained_model/current_model_2000000.pth',
            map_location='cpu' if not cuda_is_available else None
        ).eval()

        if cuda_is_available:  # put on GPU if CUDA is available
            model = model.cuda()

        test(model)

    elif mode == 'train':
        if not os.path.exists('pretrained_model/'):
            os.mkdir('pretrained_model/')

        model = NeuralNetwork()

        if cuda_is_available:  # put on GPU if CUDA is available
            model = model.cuda()

        model.apply(init_weights)
        start = time.time()

        train(model, start)


if __name__ == "__main__":
    main(sys.argv[1])

