import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from faiss import StandardGpuResources
from faiss import METRIC_L2
from faiss.contrib.torch_utils import torch_replacement_knn_gpu



# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action

	
	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)


	def forward(self, state, action):
		q = F.relu(self.l1(torch.cat([state, action], 1)))
		q = F.relu(self.l2(q))
		return self.l3(q),q


class IRA_DDPG(object):
	def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005 ,
		alpha=5e-4,
		mu=1.0,
		k=10,
		device='cuda:0',
		warmup_timestamps=10000,
		action_buffer_size=3000):
		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
		self.discount = discount
		self.tau = tau
		self.total_it = 0
		self.mu=mu
		self.k = k
		self.warmup_timestamps=warmup_timestamps
		self.action_buffer_size=action_buffer_size
		self.alpha=alpha
		self.index = 0
		self.buff_actions = torch.zeros((int(self.action_buffer_size),action_dim),device=device) 
		self.buff_actions.requires_grad=False
		torch.cuda.set_device(int(str(device).replace('cuda:','')))
		self.resource = StandardGpuResources()
		self.device=device


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		return self.actor(state).cpu().data.numpy().flatten()
	def get_K_nn(self, x,k,memory):
		with torch.no_grad():
			distance, k_index = torch_replacement_knn_gpu(self.resource, x, memory, k, metric=METRIC_L2)
			return k_index
	def adjust_mu(self): 
		self.mu = self.mu-(self.mu-0.1)/100
		assert self.mu>0

	def train(self, replay_buffer, batch_size=256):
		if self.total_it%10000==0:
			self.adjust_mu()
		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
		with torch.no_grad():
			 #store action buffer \A
			if self.index+batch_size>self.action_buffer_size:
				left_o =  int(self.index+batch_size-self.action_buffer_size )
				self.buff_actions[self.index:,:] = action[:batch_size - left_o,:]
				self.buff_actions[:left_o,:] = action[batch_size - left_o:,:]
				self.index = left_o
			else:
				self.buff_actions[self.index:self.index+batch_size,:] = action
				self.index = (self.index+batch_size) % int(self.action_buffer_size)
			# Compute the target Q value
			target_Q = self.critic_target(next_state, self.actor_target(next_state))[0]
			target_Q = reward + (not_done * self.discount * target_Q).detach()
		# Get current Q estimate
		current_Q,fea_1 = self.critic(state, action)  
		# Compute critic loss
		critic_loss = F.mse_loss(current_Q, target_Q)
		a= self.actor(state)
		with torch.no_grad():
			k_neighbor_action = torch.unique(self.buff_actions,dim=0) 
			newactions =self.get_K_nn(a, self.k,k_neighbor_action)
			qsa = torch.cat(
				[self.critic_target(state, self.buff_actions[newactions[:, i].detach().cpu().numpy()])[0] for i in range(self.k)],
				dim=1) 
			_, new_sort_index = torch.sort(qsa, descending=False, dim=-1) #Sorted
			optimal_action = torch.concatenate([self.buff_actions[new_sort_index[ind,-1]].unsqueeze(0) for  ind in range(batch_size)],dim=0)  
			sub_optimal_action = torch.concatenate([self.buff_actions[new_sort_index[ind,-2]].unsqueeze(0) for  ind in range(batch_size)],dim=0)
			_,fea_sub_1 = self.critic_target(state,sub_optimal_action) 
			 
		peer_loss = torch.einsum('ij,ij->i',[fea_1,fea_sub_1]).mean()*self.alpha
		critic_loss+=peer_loss
		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Compute actor loss
		actor_loss = -self.critic(state,a)[0].mean()
		actor_loss += ((a - optimal_action).pow(2)).mean() * self.mu
		# Optimize the actor 
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# Update the frozen target models
		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		self.total_it+=1
		return current_Q.mean().detach().cpu().numpy()


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
		