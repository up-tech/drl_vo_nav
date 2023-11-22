import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# import modules:
import os
import random

from srnn_model import *

class SpatialEdgeSelfAttn(nn.Module):
    """
    Class for the human-human attention,
    uses a multi-head self attention proposed by https://arxiv.org/abs/1706.03762
    """
    def __init__(self):
        super(SpatialEdgeSelfAttn, self).__init__()

        self.input_size = 12
        self.num_attn_heads=8
        self.attn_size=512

        self.device = torch.device("cuda" if torch.cuda.is_available else "cpu")

        # Linear layer to embed input
        self.embedding_layer = nn.Sequential(nn.Linear(self.input_size, 128), nn.ReLU(),
                                             nn.Linear(128, self.attn_size), nn.ReLU()
                                             )

        self.q_linear = nn.Linear(self.attn_size, self.attn_size)
        self.v_linear = nn.Linear(self.attn_size, self.attn_size)
        self.k_linear = nn.Linear(self.attn_size, self.attn_size)

        # multi-head self attention
        self.multihead_attn=torch.nn.MultiheadAttention(self.attn_size, self.num_attn_heads)

    # Given a list of sequence lengths, create a mask to indicate which indices are padded
    # e.x. Input: [3, 1, 4], max_human_num = 5
    # Output: [[1, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0]]
    def create_attn_mask(self, each_seq_len, seq_len, max_human_num):
        # mask with value of False means padding and should be ignored by attention
        # why +1: use a sentinel in the end to handle the case when each_seq_len = 18
        if self.device == torch.device("cpu"):
            mask = torch.zeros(seq_len, max_human_num + 1).cpu()
        else:
            mask = torch.zeros(seq_len, max_human_num+1).cuda()
        mask[torch.arange(seq_len), each_seq_len.long()] = 1.
        mask = torch.logical_not(mask.cumsum(dim=1))
        # remove the sentinel
        mask = mask[:, :-1].unsqueeze(-2) # seq_len*nenv, 1, max_human_num
        return mask

    def forward(self, inp, each_seq_len):
        '''
        Forward pass for the model
        params:
        inp : input edge features
        each_seq_len:
        if self.args.sort_humans is True, the true length of the sequence. Should be the number of detected humans
        else, it is the mask itself
        '''
        # inp is padded sequence [seq_len, nenv, max_human_num, 2]

        seq_len, max_human_num, _ = inp.size() #[1, 1, 20, 12] #seq_len: batch_size, nenv, 

        attn_mask = self.create_attn_mask(each_seq_len, seq_len, max_human_num)  # [seq_len*nenv, 1, max_human_num]
        attn_mask = attn_mask.squeeze(1)  # if we use pytorch builtin function

        input_emb=self.embedding_layer(inp).view(seq_len, max_human_num, -1)
        input_emb=torch.transpose(input_emb, dim0=0, dim1=1) # if we use pytorch builtin function, v1.7.0 has no batch first option
        q=self.q_linear(input_emb)
        k=self.k_linear(input_emb)
        v=self.v_linear(input_emb)

        #z=self.multihead_attn(q, k, v, mask=attn_mask)
        z,_=self.multihead_attn(q, k, v, key_padding_mask=torch.logical_not(attn_mask)) # if we use pytorch builtin function #[20,1,512] #[1,20,20]
        z=torch.transpose(z, dim0=0, dim1=1) # if we use pytorch builtin function
        return z

class EdgeAttention_M(nn.Module):
    '''
    Class for the robot-human attention module
    '''
    def __init__(self):
        '''
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        '''
        super(EdgeAttention_M, self).__init__()

        # Store required sizes
        self.human_human_edge_rnn_size = 256
        self.human_node_rnn_size = 128
        self.attention_size = 64

        self.device = torch.device("cuda" if torch.cuda.is_available else "cpu")

        # Linear layer to embed temporal edgeRNN hidden state
        self.temporal_edge_layer=nn.ModuleList()
        self.spatial_edge_layer=nn.ModuleList()

        self.temporal_edge_layer.append(nn.Linear(self.human_human_edge_rnn_size, self.attention_size))

        # Linear layer to embed spatial edgeRNN hidden states
        self.spatial_edge_layer.append(nn.Linear(self.human_human_edge_rnn_size, self.attention_size))

        # number of agents who have spatial edges (complete graph: all 6 agents; incomplete graph: only the robot)
        self.agent_num = 1
        self.num_attention_head = 1

    def create_attn_mask(self, each_seq_len, seq_len, max_human_num):
        # mask with value of False means padding and should be ignored by attention
        # why +1: use a sentinel in the end to handle the case when each_seq_len = 18
        if self.device == torch.device("cpu"):
            mask = torch.zeros(seq_len, max_human_num + 1).cpu()
        else:
            mask = torch.zeros(seq_len, max_human_num + 1).cuda()
        mask[torch.arange(seq_len), each_seq_len.long()] = 1.
        mask = torch.logical_not(mask.cumsum(dim=1))
        # remove the sentinel
        mask = mask[:, :-1].unsqueeze(-2)  # seq_len*nenv, 1, max_human_num
        return mask

    def att_func(self, temporal_embed, spatial_embed, h_spatials, attn_mask=None):
        seq_len, num_edges, h_size = h_spatials.size()  # [1, 12, 30, 256] in testing,  [12, 30, 256] in training
        # print(temporal_embed.size())
        # print(spatial_embed.size())
        attn = temporal_embed * spatial_embed
        #print(attn.shape)
        attn = torch.sum(attn, dim=2)
        #print(attn.shape)

        # Variable length
        temperature = num_edges / np.sqrt(self.attention_size)
        attn = torch.mul(attn, temperature)

        # if we don't want to mask invalid humans, attn_mask is None and no mask will be applied
        # else apply attn masks
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, -1e9)

        # Softmax
        attn = attn.view(seq_len, self.agent_num, self.human_num)
        attn = torch.nn.functional.softmax(attn, dim=-1)
        # print(attn[0, 0, 0].cpu().numpy())

        # Compute weighted value
        # weighted_value = torch.mv(torch.t(h_spatials), attn)

        # reshape h_spatials and attn
        # shape[0] = seq_len, shape[1] = num of spatial edges (6*5 = 30), shape[2] = 256
        h_spatials = h_spatials.view(seq_len, self.agent_num, self.human_num, h_size)
        h_spatials = h_spatials.view(seq_len * self.agent_num, self.human_num, h_size).permute(0, 2,
                                                                                         1)  # [seq_len*nenv*6, 5, 256] -> [seq_len*nenv*6, 256, 5]

        attn = attn.view(seq_len * self.agent_num, self.human_num).unsqueeze(-1)  # [seq_len*nenv*6, 5, 1]
        weighted_value = torch.bmm(h_spatials, attn)  # [seq_len*nenv*6, 256, 1]
        #print(attn.shape)
        #print(weighted_value.shape)

        # reshape back
        weighted_value = weighted_value.squeeze(-1).view(seq_len, self.agent_num, h_size)  # [seq_len, 12, 6 or 1, 256]
        return weighted_value, attn



    # h_temporal: [seq_len, nenv, 1, 256]
    # h_spatials: [seq_len, nenv, 5, 256]
    def forward(self, h_temporal, h_spatials, each_seq_len):
        '''
        Forward pass for the model
        params:
        h_temporal : Hidden state of the temporal edgeRNN
        h_spatials : Hidden states of all spatial edgeRNNs connected to the node.
        each_seq_len:
            if self.args.sort_humans is True, the true length of the sequence. Should be the number of detected humans
            else, it is the mask itself
        '''
        seq_len, max_human_num, _ = h_spatials.size()
        # find the number of humans by the size of spatial edgeRNN hidden state
        self.human_num = max_human_num // self.agent_num

        weighted_value_list, attn_list=[],[]
        for i in range(self.num_attention_head):

            temporal_embed = self.temporal_edge_layer[i](h_temporal)
            # temporal_embed = temporal_embed.squeeze(0)

            # Embed the spatial edgeRNN hidden states
            spatial_embed = self.spatial_edge_layer[i](h_spatials)
            #print(f'temporal_embed size {temporal_embed.size()}')
            #print(f'spatial_embed size {spatial_embed.size()}')

            # Dot based attention
            temporal_embed = temporal_embed.repeat_interleave(self.human_num, dim=1)
            #print(f'temporal_embed size {temporal_embed.size()}')


            attn_mask = self.create_attn_mask(each_seq_len, seq_len, max_human_num)  # [seq_len*nenv, 1, max_human_num]
            attn_mask = attn_mask.squeeze(-2).view(seq_len, max_human_num)

            weighted_value,attn=self.att_func(temporal_embed, spatial_embed, h_spatials, attn_mask=attn_mask)
            weighted_value_list.append(weighted_value)
            attn_list.append(attn)
        #print(f'weighted_value_list size {len(weighted_value_list)}')
        if self.num_attention_head > 1:
            #print(f'1111 weighted_value_list size {len(self.final_attn_linear(torch.cat(weighted_value_list, dim=-1)))}')
            return self.final_attn_linear(torch.cat(weighted_value_list, dim=-1)), attn_list
        else:
            return weighted_value_list[0], attn_list[0]

class EndRNN(RNNBase):
    '''
    Class for the GRU
    '''
    def __init__(self):
        '''
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        '''
        super(EndRNN, self).__init__(edge=False)

        # Store required sizes
        self.rnn_size = 128
        self.output_size = 256
        self.embedding_size = 64
        self.input_size = 3
        self.edge_rnn_size = 256

        # Linear layer to embed input
        self.encoder_linear = nn.Linear(256, self.embedding_size)

        # ReLU and Dropout layers
        self.relu = nn.ReLU()

        # Linear layer to embed attention module output
        self.edge_attention_embed = nn.Linear(self.edge_rnn_size, self.embedding_size)

        # Output linear layer
        self.output_linear = nn.Linear(self.rnn_size, self.output_size)



    def forward(self, robot_s, h_spatial_other):
        '''
        Forward pass for the model
        params:
        pos : input position
        h_temporal : hidden state of the temporal edgeRNN corresponding to this node
        h_spatial_other : output of the attention module
        h : hidden state of the current nodeRNN
        c : cell state of the current nodeRNN
        '''
        # Encode the input position
        encoded_input = self.encoder_linear(robot_s)
        encoded_input = self.relu(encoded_input)

        h_edges_embedded = self.relu(self.edge_attention_embed(h_spatial_other))

        concat_encoded = torch.cat((encoded_input, h_edges_embedded), -1)

        # x, h_new = self._forward_gru(concat_encoded, h, masks)

        # outputs = self.output_linear(x)

        return concat_encoded

class selfAttn_merge_SRNN(BaseFeaturesExtractor):
    """
    Class for the proposed network
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        """
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        """

        super(selfAttn_merge_SRNN, self).__init__(observation_space, features_dim)
        self.is_recurrent = True 

        self.human_num = 10
        #observation_space['spatial_edges'].shape[0] #5

        # Store required sizes
        self.human_node_rnn_size = 128
        self.human_human_edge_rnn_size = 256

        self.output_size = features_dim #256

        # Initialize the Node and Edge RNNs
        self.humanNodeRNN = EndRNN()

        # Initialize attention module
        self.attn = EdgeAttention_M()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        num_inputs = hidden_size = self.output_size

        # self.actor = nn.Sequential(
        #     init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
        #     init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        # self.critic = nn.Sequential(
        #     init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
        #     init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        robot_size = 7
        self.robot_linear = nn.Sequential(init_(nn.Linear(robot_size, 256)), nn.ReLU()) # todo: check dim
        self.human_node_final_linear=init_(nn.Linear(self.output_size,2))


        self.spatial_attn = SpatialEdgeSelfAttn()
        self.spatial_linear = nn.Sequential(init_(nn.Linear(512, 256)), nn.ReLU())

        self.temporal_edges = [0]
        self.spatial_edges = np.arange(1, self.human_num+1)

        self.final_layer = nn.Sequential(nn.Linear(128, 256),
                                         nn.ReLU())

        # dummy_human_mask = [0] * self.human_num
        # dummy_human_mask[0] = 1

        self.device = torch.device("cuda" if torch.cuda.is_available else "cpu")

        # if self.device == torch.device("cpu"):
        #     self.dummy_human_mask = Variable(torch.Tensor([dummy_human_mask]).cpu())
        # else:
        #     self.dummy_human_mask = Variable(torch.Tensor([dummy_human_mask]).cuda())

    def forward(self, inputs):
        # if infer:
        #     # Test/rollout time
        #     seq_length = 1
        #     nenv = self.nenv

        # else:
        #     # Training time
        #     seq_length = self.seq_length
        #     nenv = self.nenv // self.nminibatch

        #robot_node = reshapeT(inputs['robot_node'], seq_length, nenv) #(1, 1, 1, 5)
        robot_node = inputs['robot_node']
        #print(f"robot input size: {inputs['robot_node'].shape}")

        #temporal_edges = reshapeT(inputs['temporal_edges'], seq_length, nenv) #(1, 1, 1, 2)
        temporal_edges = inputs['temporal_edges']
        #print(f"temporal_edges size: {temporal_edges.shape}")

        #spatial_edges = reshapeT(inputs['spatial_edges'], seq_length, nenv) #(1, 1, 20, 12)
        spatial_edges = inputs['spatial_edges']
        spatial_edges = spatial_edges.view(spatial_edges.shape[0] ,self.human_num, -1)
        #spatial_edges = spatial_edges.view(self.human_num, -1)
        #print(f"spatial_edges: {spatial_edges.shape}")

        #detected_human_num = inputs['detected_human_num'].squeeze(-1).cpu().int() #changable
        detected_human_num = inputs['detected_human_num'].int() #changable
        #print(f"detected_human_num size: {detected_human_num.shape}")

        #hidden_states_node_RNNs = reshapeT(rnn_hxs['human_node_rnn'], 1, nenv) #(1, 1, 1, 128)

        # if self.device == torch.device("cpu"):
        #     all_hidden_states_edge_RNNs = Variable(
        #         torch.zeros(1, nenv, 1+self.human_num, rnn_hxs['human_human_edge_rnn'].size()[-1]).cpu())
        # else:
        #     all_hidden_states_edge_RNNs = Variable(
        #         torch.zeros(1, nenv, 1+self.human_num, rnn_hxs['human_human_edge_rnn'].size()[-1]).cuda())

        robot_states = torch.cat((temporal_edges, robot_node), dim=-1)
        robot_states = self.robot_linear(robot_states)  #(1, 1, 1, 256)
        #print(f'robot state size: {robot_states.size()}')

        spatial_attn_out=self.spatial_attn(spatial_edges, detected_human_num)
        #print(f'spatial_attn_out size: {spatial_attn_out.size()}')

        output_spatial = self.spatial_linear(spatial_attn_out)
        #print(f'output_spatial size: {output_spatial.size()}')

        # robot-human attention
        hidden_attn_weighted, _ = self.attn(robot_states, output_spatial, detected_human_num)
        #print(f'hidden_attn_weighted size: {hidden_attn_weighted.size()}')

        # Do a forward pass through GRU
        outputs = self.humanNodeRNN(robot_states, hidden_attn_weighted)


        # Update the hidden and cell states
        #all_hidden_states_node_RNNs = h_nodes
        #outputs_return = outputs

        # rnn_hxs['human_node_rnn'] = all_hidden_states_node_RNNs
        # rnn_hxs['human_human_edge_rnn'] = all_hidden_states_edge_RNNs

        # x is the output and will be sent to actor and critic
        # x = outputs_return[0, :]
        #print(f'outputs size: {outputs.size()}')
        
        final_output = self.final_layer(outputs)
        final_output = final_output.squeeze(1)
        #print(f'final_output size: {final_output.size()}')

        # hidden_critic = self.critic(x)
        # hidden_actor = self.actor(x)

        # for key in rnn_hxs:
        #     rnn_hxs[key] = rnn_hxs[key].squeeze(0)

        return final_output

        #return self.critic_linear(hidden_critic).view(-1, 1), hidden_actor.view(-1, self.output_size), rnn_hxs

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

# def reshapeT(T, seq_length, nenv):
#     shape = T.size()[1:]
#     print(f"shape is {shape}")
#     print(f"seq_length is {seq_length}")
#     print(f"nenv is {nenv}")
#     return T.unsqueeze(0).reshape((seq_length, nenv, *shape))