import random
import numpy as np
import warnings
import pandas as pd
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import collections
import copy
warnings.filterwarnings("ignore")

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Market(object):
    def __init__(self):
        # Set constant parameters
        random.seed(41)
        self.bid_ask_spread = 1/12
        self.number_of_trade = 100
        self.liquidation_time = 150
        self.tau = self.liquidation_time / self.number_of_trade
        self.total_shares = 1000000 # 1,000,000
        self.stock_price = 100
        self.eta = self.bid_ask_spread/ 5000
        self.epsilon = self.bid_ask_spread / 2
        self.llambda = 0.00001
        self.single_step_variance = 0.4
        self.gamma = self.bid_ask_spread / (500000)
        self.eta_est = self.eta - (0.5 * self.gamma * self.tau)
        self.kappa_est = np.sqrt(self.llambda * self.single_step_variance / self.eta_est)
        # print(self.llambda * self.single_step_variance / self.eta_est)
        self.kappa = np.arccosh((((self.kappa_est ** 2) * (self.tau ** 2)) / 2) + 1) / self.tau
        # print(self.kappa)
        self.transacting = False

        # Paramters that would be change in the recursive process
        self.log_returns = collections.deque(np.zeros(1))
        self.time_horizon = self.number_of_trade
        self.shares_remaining = self.total_shares
        self.price = self.stock_price
        self.ind = 0

    def reset(self):
        """
        :return: initial_state
        """
        self.__init__()
        # self.initial_state = np.(list(self.))
        self.initial_state = np.array(list(self.log_returns) + [self.time_horizon / self.number_of_trade, \
                                                               self.shares_remaining / self.total_shares])
        # print(self.initial_state)
        return self.initial_state

    def start_transaction(self):
        """
        This is to initialize a transaction indicator to control when to start and stop trading
        :return:
        """
        self.transacting = True
        self.capture = 0
        self.previous_price = self.stock_price

    def step(self, action):
        """
        Classic step function for Environment.
        :param action:
        :return:
        """
        class Temp(object):
            pass
        tmp = Temp()
        tmp.done = False
        if self.transacting and (self.time_horizon == 0 or abs(self.shares_remaining) < 1):
            self.transacting = False
            tmp.done = True
            tmp.implementation_shortfall = self.total_shares * self.stock_price - self.capture

        if self.ind == 0:
            tmp.price = self.price
        else:
            tmp.price = self.price + np.sqrt(self.single_step_variance * self.tau) * random.normalvariate(0, 1)

        if self.transacting:

            tmp.share_to_sell = self.shares_remaining * action
            if self.time_horizon == 1:
                tmp.share_to_sell = self.shares_remaining
            tmp.temporary_impact = self.epsilon * np.sign(tmp.share_to_sell) + (self.eta/self.tau) * tmp.share_to_sell

            tmp.execution_price = tmp.price - tmp.temporary_impact
            self.capture += tmp.share_to_sell * tmp.execution_price
            self.log_returns.append(np.log(tmp.price/self.previous_price))
            self.log_returns.popleft()

            self.shares_remaining -= tmp.share_to_sell
            if self.shares_remaining <= 0:
                tmp.implementation_shortfall = self.total_shares * self.stock_price - self.capture
                tmp.done = True
            # reward = tmp.share_to_sell * tmp.execution_price
            reward = - (self.total_shares * self.stock_price - self.capture)
        else:
            reward = 0

        self.ind += 1
        state = np.array(list(self.log_returns) + [self.time_horizon / self.number_of_trade, self.shares_remaining / self.total_shares])
        return (state, np.array([reward]), tmp.done, tmp)


    def obtain_trade_trajectory(self):
        """
        Obtain ACF trade trajectory as a benchmark.
        :return:
        """
        trade_trajectory = np.zeros(self.number_of_trade) # initial trade_trajectory as a numpy array
        for i in range(1, self.number_of_trade + 1):
            trade_trajectory[i - 1] = np.cosh(self.kappa * (self.liquidation_time - (i - 0.5) * self.tau))
        trade_trajectory *= (2 * np.sinh(0.5 * self.kappa * self.tau)) / (np.sinh(self.kappa * self.liquidation_time)) \
                            * self.total_shares
        return trade_trajectory

    def plot_trade_trajectory(self):
        """
        Plot the ACF trade trajectory.
        :return:
        """
        trade_trajecotry = self.obtain_trade_trajectory()
        trade_trajecotry = np.insert(trade_trajecotry, 0, 0)
        # print(trade_trajecotry)

        df = pd.DataFrame(data=list(range(self.number_of_trade + 1)), columns=['Trade Number'])
        df['Stocks Sold'] = trade_trajecotry
        df['Stocks Remaining'] = (np.ones(self.number_of_trade + 1) * env.total_shares) - np.cumsum(trade_trajecotry)
        plt.figure(figsize=(20,10))
        df.plot.scatter(x='Trade Number', y='Stocks Remaining')

        plt.show()

env = Market()
env.reset()
env.plot_trade_trajectory()

# Get the ACF results first
trading_list = env.obtain_trade_trajectory()
price_list = np.array([])
env.start_transaction()
for trade in trading_list:
    action = trade/ env.shares_remaining
    _, _, _, tmp = env.step(action)
    if tmp.done:
        print('Implementation Shortfall:', tmp.implementation_shortfall)
        acf_is = tmp.implementation_shortfall
        break
    price_list = np.append(price_list, tmp.execution_price)

df = pd.DataFrame(price_list, columns = ['Stock'])
print(df)
ax = df.plot()
ax.set_title('Simulated Stock Price following the Random Walk Process' )
plt.plot(price_list, 'o')
plt.ylabel('Stock Price')
plt.xlabel('Trade Number')
plt.show()
# Implementation shortfall calculation

# def hidden_state(layer):
#     # variable of size [num_layers*num_of_trades, b_sz, hidden_sz]
#     return Variable(torch.zeros(num_of_trades, layer, idden_size)).cuda()

# Hidden state initialization
def hidden_state(layer):
    # print(np.sqrt(layer.weight.data.size()[0]))
    return (-1./np.sqrt(layer.weight.data.size()[0]), 1./np.sqrt(layer.weight.data.size()[0]))

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(41)
        self.hidden1 = nn.Linear(state_dim, 32)
        self.hidden2 = nn.Linear(32, 64)
        self.hidden3 = nn.Linear(64, action_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.hidden1.weight.data.uniform_(*hidden_state(self.hidden1))
        self.hidden2.weight.data.uniform_(*hidden_state(self.hidden2))
        self.hidden3.weight.data.uniform_(-0.003, 0.003)

    def forward(self, state):
        fl = F.relu(self.hidden1(state))
        fl = F.relu(self.hidden2(fl))
        return F.tanh(self.hidden3(fl))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(41)
        self.hidden = nn.Linear(state_dim, 32)
        self.hidden2 = nn.Linear(33, 64)
        self.hidden3 = nn.Linear(64, 1)
        self.reset_parameter()

    def reset_parameter(self):
        self.hidden.weight.data.uniform_(*hidden_state(self.hidden))
        self.hidden2.weight.data.uniform_(*hidden_state(self.hidden2))
        self.hidden3.weight.data.uniform_(-0.003, 0.003)

    def forward(self, state, action):
        f_layer = F.relu(self.hidden(state))
        fl = torch.cat((f_layer, action), dim=1)
        fl = F.relu(self.hidden2(fl))
        return self.hidden3(fl)

class MemoryBuffer:

    def __init__(self, action_size, batch_size, seed):

        self.action_size = action_size
        self.memory = collections.deque(maxlen=10000)
        self.batch_size = batch_size
        self.experience = collections.namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        self.memory.append(self.experience(state, action, reward, next_state, done))

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

class Noise:
    """
    Assume the noise item follow zero mean normal distribution.
    """
    def __init__(self, size, seed):
        self.mu = 0 * np.ones(size)
        self.theta = 0.15
        self.sigma = 0.2
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        """
        :return: a sample state of noise
        """
        self.state += self.theta * (self.mu - self.state) + self.sigma * np.array([random.random() for i in range(len(self.state))])
        return self.state

class DRL():
    def __init__(self):
        self.state_dim = 3
        self.action_dim = 1
        self.seed = random.seed(41)
        self.actor_local = Actor(self.state_dim, self.action_dim).to(device)
        self.actor_target = Actor(self.state_dim, self.action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr = 0.0001)

        self.critic_local = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr = 0.001)

        self.noise = Noise(self.action_dim, 41)
        self.memory = MemoryBuffer(self.action_dim, 128, 41)

    def learn(self, experiences, gamma):
        """
        DRL learn from the Environment, perceive states.
        :param experiences:
        :param gamma:
        :return:
        """
        states, actions, rewards, next_states, dones = experiences
        action_next = self.actor_target(next_states)
        Q_objective_next = self.critic_target(next_states, action_next)
        Q_objective = rewards + (gamma * Q_objective_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_objective)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        actions_predition = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_predition).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.critic_local, self.critic_target, 1e-3)
        self.soft_update(self.actor_local, self.actor_target, 1e-3)

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update for the DRL model.
        :param local_model:
        :param target_model:
        :param tau:
        :return:
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > 128:
            experiences = self.memory.sample()
            self.learn(experiences, 0.9)

    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()

        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        action = (action + 1.0) / 2.0
        return np.clip(action, 0, 1)


if __name__ == '__main__':
    start_time = time.time()
    drl = DRL()
    episodes = 300
    steps = 50

    shortfall_hist = np.array([])
    shortfall_deque = collections.deque(maxlen=100)
    shortfall_lst = []

    for episode in range(episodes):
        current_state = env.reset()
        env.start_transaction()
        for step in range(steps):
            action = drl.act(current_state)
            new_state, reward, done, tmp = env.step(action)
            drl.step(current_state, action, reward, new_state, done)
            cur_state = new_state
            if tmp.done:
                # print(tmp.implementation_shortfall)
                shortfall_hist = np.append(shortfall_hist, tmp.implementation_shortfall)
                shortfall_deque.append(tmp.implementation_shortfall)
                break

        if (episode + 1) % 10 == 0:
            print(episode + 1, np.mean(shortfall_deque))
            shortfall_lst.append([episode + 1,np.mean(shortfall_deque)])
            # print(shortfall_lst)

    shortfall_df = pd.DataFrame(shortfall_lst, columns = ['Episode', 'Implementation Shortfall'])
    # shortfall_df.plot(x='Episode', y='Implementation Shortfall')
    plt.plot(shortfall_df['Episode'], shortfall_df['Implementation Shortfall'])
    # print(tmp.implementation_shortfall)
    plt.plot([0, episodes], [acf_is, acf_is], color='g', linestyle=':')
    plt.legend(['DRL Implementation Shortfall', 'ACF Implementation Shortfall'])
    plt.show()

    print("Model Training Time:", (time.time()-start_time)/60, 'mins.')





