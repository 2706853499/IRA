import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



from faiss import StandardGpuResources
from faiss import METRIC_L2
from faiss import METRIC_Linf
from faiss.contrib.torch_utils import torch_replacement_knn_gpu
# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


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

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1_f = F.relu(self.l2(q1))
		q1 = self.l3(q1_f)

		q2 = F.relu(self.l4(sa))
		q2_f = F.relu(self.l5(q2))
		q2 = self.l6(q2_f)
		return q1, q2,q1_f,q2_f


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)
		q1 = F.relu(self.l1(sa))
		q1_f = F.relu(self.l2(q1))
		q1 = self.l3(q1_f)
		return q1,q1_f
	def Q2(self, state, action):
		sa = torch.cat([state, action], 1)
		q2 = F.relu(self.l4(sa))
		q2_f = F.relu(self.l5(q2))
		q2 = self.l6(q2_f)
		return  q2,q2_f


class IRA(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		alpha=5e-4,
		mu=1.0,
		k=10,
		device='cuda:0',
		warmup_timestamps=4000,
		action_buffer_size=1e5
	):
		self.device=device
		self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
		self.critic = Critic(state_dim, action_dim).to(self.device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.total_it = 0
		self.alpha=alpha
		self.mu=mu
		self.k = k
		self.warmup_timestamps=warmup_timestamps
		self.action_buffer_size=action_buffer_size
		self.index = 0
		self.index = self.index% int(self.action_buffer_size)
		self.buff_actions = torch.zeros((int(self.action_buffer_size),action_dim),device=self.device) 
		self.buff_actions.requires_grad=False
		torch.cuda.set_device(int(str(self.device).replace('cuda:','')))
		self.resource = StandardGpuResources()
		 
		


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		return self.actor(state).cpu().data.numpy().flatten()

	def get_K_nn(self, x,k,memory):
		with torch.no_grad():
			distance, k_index = torch_replacement_knn_gpu(self.resource, x, memory, k, metric=METRIC_Linf)
			return k_index
	def adjust_mu(self): 
		self.mu = self.mu-(self.mu-0.1)/100
		assert self.mu>0

	def train(self, replay_buffer, batch_size=256):
		if self.total_it%10000==0:
			self.adjust_mu()
		self.total_it += 1
		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
		with torch.no_grad():
			#  strore Action Buffer \A
			if self.index+batch_size>self.action_buffer_size:
				left_o =  int(self.index+batch_size-self.action_buffer_size )
				self.buff_actions[self.index:,:] = action[:batch_size - left_o,:]
				self.buff_actions[:left_o,:] = action[batch_size - left_o:,:]
				self.index = left_o
			else:
				self.buff_actions[self.index:self.index+batch_size,:] = action
				self.index = (self.index+batch_size) % int(self.action_buffer_size)
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				 self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)
     		# Compute the target Q value
			target_Q1, target_Q2,_,_ = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2) 
			target_Q = reward + not_done * self.discount * target_Q
			
		predict_action = self.actor(state)
		if self.total_it>self.warmup_timestamps:
			with torch.no_grad():
				
				k_neighbor_action = torch.unique(self.buff_actions,dim=0) 
				newactions =self.get_K_nn(predict_action, self.k,k_neighbor_action)
				q1 = torch.cat(
					[self.critic_target.Q1(state, self.buff_actions[newactions[:, i].detach().cpu().numpy()])[0] for i in range(self.k)],
					dim=1) 
				q2 = torch.cat(
					[self.critic_target.Q2(state, self.buff_actions[newactions[:, i].detach().cpu().numpy()])[0] for i in range(self.k)],
					dim=1) 
				 
				qsa = torch.min(q1,q2) 
				_, sorted_index = torch.sort(qsa, descending=False, dim=-1) #Sort
				optimal_action = torch.concatenate([self.buff_actions[sorted_index[ind,-1]].unsqueeze(0) for  ind in range(batch_size)],dim=0)  # guid the actor
				sub_optimal_action = torch.concatenate([self.buff_actions[sorted_index[ind,-2]].unsqueeze(0) for  ind in range(batch_size)],dim=0)
				_,fea_sub_1 = self.critic_target.Q1(state,sub_optimal_action) 
				_,fea_sub_2 = self.critic_target.Q2(state,sub_optimal_action) 
				 
		# Get current Q estimates
		current_Q1, current_Q2,fea_1,fea_2  = self.critic(state, action)
						
		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
		#warmup
		if self.total_it>self.warmup_timestamps: #4000
			peer_loss = torch.einsum('ij,ij->i',[fea_1,fea_sub_1]).mean()*self.alpha+torch.einsum('ij,ij->i',[fea_2,fea_sub_2]).mean()*self.alpha
			critic_loss+=peer_loss

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()
		
		# Delayed policy updates  (IAR policy_freq=1)
		if self.total_it%self.policy_freq==0:
			# Compute actor losse
			actor_loss = -self.critic.Q1(state, predict_action)[0].mean()
			if self.total_it>self.warmup_timestamps:  
				actor_loss += ((predict_action - optimal_action).pow(2)).mean() * self.mu
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()   
			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			
			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

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
 