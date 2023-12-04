import copy
import json
import logging
import math
import sys
from io import open
import numpy as np
from .model_utils import *
import torch.distributed.nn
import torch.distributed as dist
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch_scatter import scatter

from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims 

# full_atom_feature_dims = [x + 10 for x in get_atom_feature_dims()]  # 0 for padding idx
full_atom_feature_dims = [129, 19, 22, 22, 20, 16, 16, 12, 12]
# full_bond_feature_dims = [x + 10 for x in get_bond_feature_dims()]  # 0 for padding idx
full_bond_feature_dims = [15, 16, 12]

gelu = nn.GELU()

class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()
        
        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim, padding_idx=0)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        # x: [bs, nodes, n_feature]
        x_embedding = 0
        for i in range(x.shape[-1]):
            x_embedding += self.atom_embedding_list[i](x[:, :, i])

        return x_embedding


class BondEncoder(torch.nn.Module):
    
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()
        
        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim, padding_idx=0)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[-1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, :, :, :, i])

        return bond_embedding   


class GraphormerConfig(object):

    def __init__(self,
                 config_json_file=-1,
                 max_degree=25,
                 multi_hop_max_dist=3,
                 max_spd = 128,
                 num_kernel = 128,
                 edge_types = 150,
                 pos_encoding_scale = 0.1,
                 hidden_size=512,
                 num_hidden_layers=8,
                 num_attention_heads=64,
                 intermediate_size=768,
                 hidden_dropout_prob=0.1,
                 embedding_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 nodes_attention_dropout_prob=0.0,
                 initializer_range=0.02,
                 layer_norm_eps=1e-5,
                 post_ln=True):
       
        if isinstance(config_json_file, str) or (sys.version_info[0] == 2
                                                               and isinstance(config_json_file, unicode)):
            with open(config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(config_json_file, int):
            self.max_degree = max_degree
            self.multi_hop_max_dist = multi_hop_max_dist
            self.max_spd = max_spd
            self.num_kernel = num_kernel
            self.edge_types = edge_types
            self.pos_encoding_scale = pos_encoding_scale
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.embedding_dropout_prob = embedding_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.nodes_attention_dropout_prob = nodes_attention_dropout_prob
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
            # compute atom embedding scale, including degree embeddings
            self.atom_embedding_scale = math.sqrt(1 / (self.initializer_range * (len(full_atom_feature_dims) + 1)))
            # compute bond scale
            self.bond_embedding_scale = math.sqrt(1 / (self.initializer_range * len(full_bond_feature_dims)))
            # compute spd scale
            self.spd_scale = math.sqrt(1 / self.initializer_range)
            self.post_ln = post_ln
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = GraphormerConfig(config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())


class GraphormerLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(GraphormerLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        # s = x.var(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class GraphormerEmbeddings(nn.Module):

    def __init__(self, config):
        super(GraphormerEmbeddings, self).__init__()
        self.post_ln = config.post_ln
        self.atom_embedding_scale = config.atom_embedding_scale
        self.pos_encoding_scale = config.pos_encoding_scale
        self.atom_embeddings = AtomEncoder(emb_dim=config.hidden_size)
        self.degree_encoder = nn.Embedding(config.max_degree, config.hidden_size, padding_idx=0)
        self.graph_token = nn.Embedding(1, config.hidden_size)
        if self.post_ln:
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.dropout = nn.Dropout(config.embedding_dropout_prob)

    def forward(self, batched_data, pos_encoding=None):
        x, degree = batched_data['x'], batched_data['in_degree']
        batch_size, _, _ = x.shape

        # [batch_size, nodes, hid]
        atom_embeddings = self.atom_embeddings(x)  
        embeddings = atom_embeddings
        
        degree_embeddings = self.degree_encoder(degree)
        embeddings = embeddings + degree_embeddings

        graph_token_embedding = self.graph_token.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        embeddings = torch.cat([graph_token_embedding, embeddings], dim=1)

        embeddings = embeddings * self.atom_embedding_scale

        if pos_encoding is not None:
            pos_encoding = torch.clamp(pos_encoding, min=-1, max=1)
            embeddings = embeddings + pos_encoding * self.pos_encoding_scale

        if not self.post_ln:
            return embeddings
        else:
            embeddings = self.LayerNorm(embeddings)
            return self.dropout(embeddings)


class Graphormer2DAttentionBias(nn.Module):

    def __init__(self, config):
        super(Graphormer2DAttentionBias, self).__init__()
        self.bond_embedding_scale = config.bond_embedding_scale
        self.spd_scale = config.spd_scale
        self.num_heads = config.num_attention_heads
        self.multi_hop_max_dist = config.multi_hop_max_dist
        self.num_layers = config.num_hidden_layers
        
        self.spatial_pos_encoder = nn.Embedding(config.max_spd, config.num_hidden_layers * config.num_attention_heads, padding_idx=0)
        self.graph_token_virtual_dist = nn.Parameter(
            torch.randn(1, config.num_attention_heads * config.num_hidden_layers).normal_(mean=0.0, std=config.initializer_range), requires_grad=True)
        self.edge_encoder = BondEncoder(emb_dim=config.num_hidden_layers * config.num_attention_heads)
        self.edge_dis_encoder = nn.Parameter(
            torch.randn(self.multi_hop_max_dist, self.num_layers * self.num_heads, 
                        self.num_layers * self.num_heads).normal_(mean=0.0, std=config.initializer_range), 
                        requires_grad=True)
        
    def forward(self, batched_data):
        attn_bias, spatial_pos, x, edge_input = (
            batched_data["attn_bias"],
            batched_data["spatial_pos"],
            batched_data["x"],
            batched_data["edge_input"]
        )

        batch_size, nodes, hid = x.shape
        
        graph_attn_bias = attn_bias.clone()
        # [bs, nodes+1, nodes+1] -> [bs, layers, heads, nodes+1, nodes+1]
        graph_attn_bias = graph_attn_bias.unsqueeze(1).unsqueeze(2).repeat(1, self.num_layers, self.num_heads, 1, 1)  
        
        # spatial pos
        # [bs, nodes, nodes, layers*n_head] -> [bs, layers, heads, nodes, nodes]
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2).reshape(batch_size, self.num_layers, self.num_heads, nodes, nodes)
        # spatial_pos_bias = spatial_pos_bias * self.spd_scale
        graph_attn_bias[:, :, :, 1:, 1:] = graph_attn_bias[:, :, :, 1:, 1:] + spatial_pos_bias

        # graph spatial pos here
        graph_token = self.graph_token_virtual_dist.view(1, self.num_layers, self.num_heads, 1)
        graph_attn_bias[:, :, :, 1:, 0] = graph_attn_bias[:, :, :, 1:, 0] + graph_token
        graph_attn_bias[:, :, :, 0, :] = graph_attn_bias[:, :, :, 0, :] + graph_token

        # edge feature
        spatial_pos_ = spatial_pos.clone()
        spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
        spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)  # set 1 to 1, x > 1 to x - 1
        spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)

        edge_input = edge_input[:, :, :, :self.multi_hop_max_dist, :]
        # [bs, nodes, nodes, max_dist, layers*n_head]
        edge_input = self.edge_encoder(edge_input)
        # edge_input = edge_input * self.bond_embedding_scale

        max_dist = edge_input.size(-2)
        # [max_dist, bs*nodes*nodes, layers*n_head]
        edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_layers * self.num_heads)
        # bmm [max_dist, bs*nodes*nodes, layers*n_head] and [max_dist, layers*n_head, layers*n_head] -> [max_dist, bs*nodes*nodes, layers*n_head]
        edge_input_flat = torch.einsum('bij,bjh->bih', edge_input_flat, self.edge_dis_encoder)

        # to [bs, nodes, nodes, max_dist, layers*n_head]
        edge_input = edge_input_flat.reshape(max_dist, batch_size, nodes, nodes, self.num_layers * self.num_heads).permute(1, 2, 3, 0, 4)
        # [bs, nodes, nodes, layers*n_head] -> [bs, layers*n_head, nodes, nodes]
        edge_input = (edge_input.sum(-2) / (spatial_pos_.float().unsqueeze(-1))).permute(0, 3, 1, 2)
        #
        edge_input = edge_input.reshape(batch_size, self.num_layers, self.num_heads, nodes, nodes)

        graph_attn_bias[:, :, :, 1:, 1:] = graph_attn_bias[:, :, :, 1:, 1:] + edge_input

        # [bs, layers, nodes+1, nodes+1]
        return graph_attn_bias


class GraphormerGaussianLayer(nn.Module):
    def __init__(self, config):
        super(GraphormerGaussianLayer, self).__init__()
        self.num_kernel = config.num_kernel

        self.mul = nn.Embedding(config.edge_types, 1, padding_idx=0)  # edge_types = 200
        self.bias = nn.Embedding(config.edge_types, 1, padding_idx=0)
        self.means = nn.Embedding(1, config.num_kernel)
        self.stds = nn.Embedding(1, config.num_kernel)

    def forward(self, dist, edge_types):
        '''
            dist: [bs, nodes+1, nodes+1]
            edge_types: [bs, nodes+1, nodes+1, 2]
        '''
        
        mul = self.mul(edge_types).sum(dim=-2)
        # [bs, nodes+1, nodes+1, 2, 1] -> [bs, nodes+1, nodes+1, 1]
        bias = self.bias(edge_types).sum(dim=-2)
        # [bs, nodes+1, nodes+1, 1] * [bs, nodes+1, nodes+1, 1]
        dist = mul * dist.unsqueeze(-1) + bias
        # [bs, nodes+1, nodes+1, K]
        dist = dist.expand(-1, -1, -1, self.num_kernel)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-2
        # [bs, nodes+1, nodes+1, K]
        return self.gaussian(dist.float(), mean, std).type_as(self.means.weight)

    def gaussian(self, x, mean, std):
        pi = 3.141593
        a = (2 * pi) ** 0.5
        return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class Graphormer3DAttentionBias(nn.Module):
    def __init__(self, config):
        super(Graphormer3DAttentionBias, self).__init__()
        self.num_heads = config.num_attention_heads
        self.num_layers = config.num_hidden_layers + 1  # for equivariant attention head
        self.gbf = GraphormerGaussianLayer(config)
        self.gbf_proj = nn.Sequential(nn.Linear(config.num_kernel, config.num_kernel),
                                      nn.GELU(),
                                      nn.Linear(config.num_kernel, config.num_attention_heads * self.num_layers))
        self.graph_token_virtual_dist = nn.Parameter(
            torch.randn(1, config.num_attention_heads * self.num_layers).normal_(mean=0.0, std=config.initializer_range), requires_grad=True)
        self.edge_proj = nn.Linear(config.num_kernel, config.hidden_size)
        
    def forward(self, batched_data):
        # pos: [bs, nodes+1, nodes+1, 3]
        # node_type_edge: [bs, nodes+1, nodes+1, 2]
        pos, node_type_edge, attn_bias = batched_data['pos'], batched_data['node_type_edge'], batched_data["attn_bias"]

        batch_size, nodes, _ = pos.shape
        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        # [bs, nodes+1, nodes+1]
        dist = delta_pos.norm(dim=-1).view(-1, nodes, nodes) 
        # [bs, nodes+1, nodes+1, 3]
        delta_pos = delta_pos / (dist.unsqueeze(-1) + 1e-6)

        # [bs, nodes+1, nodes+1, K]
        edge_feature = self.gbf(dist, node_type_edge)
        # [bs, nodes+1, nodes+1, K] -> [bs, nodes+1, nodes+1, layers*n_head]
        gbf_result = self.gbf_proj(edge_feature).permute(0, 3, 1, 2).contiguous()
        # [bs, nodes, nodes, layers*n_head] -> [bs, layers, heads, nodes, nodes]
        gbf_result = gbf_result.reshape(batch_size, self.num_layers, self.num_heads, nodes, nodes)

        graph_attn_bias = attn_bias.clone()
        # [bs, nodes+1, nodes+1] -> [bs, layers, heads, nodes+1, nodes+1]
        graph_attn_bias = graph_attn_bias.unsqueeze(1).unsqueeze(2).repeat(1, self.num_layers, self.num_heads, 1, 1)  

        # graph spatial pos here
        graph_token = self.graph_token_virtual_dist.view(1, self.num_layers, self.num_heads, 1)
        graph_attn_bias[:, :, :, 1:, 0] = graph_attn_bias[:, :, :, 1:, 0] + graph_token
        graph_attn_bias[:, :, :, 0, :] = graph_attn_bias[:, :, :, 0, :] + graph_token

        # graph_attn_bias[:, :, :, 1:, 1:] = graph_attn_bias[:, :, :, 1:, 1:] + gbf_result
        graph_attn_bias = graph_attn_bias + gbf_result

        sum_edge_features = edge_feature.mean(dim=-2)
        merge_edge_features = self.edge_proj(sum_edge_features)

        # _, [bs, nodes, hid], [bs, nodes, nodes]
        return graph_attn_bias, merge_edge_features, delta_pos


class GraphormerSelfAttention(nn.Module):
    def __init__(self, config):
        super(GraphormerSelfAttention, self).__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if not self.config.post_ln:
            self.LayerNorm_2d = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.LayerNorm_3d = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.attn_scaling = (config.hidden_size // config.num_attention_heads) ** -0.5

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, attention_bias, mode='2d'):
        if mode == '2d' and not self.config.post_ln:
            hidden_states = self.LayerNorm_2d(hidden_states)
        elif mode == '3d' and not self.config.post_ln:
            hidden_states = self.LayerNorm_3d(hidden_states)

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer) * math.sqrt(self.attn_scaling)
        key_layer = self.transpose_for_scores(mixed_key_layer) * math.sqrt(self.attn_scaling)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores + attention_mask + attention_bias
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer, attention_scores
        

class GraphormerSelfOutput(nn.Module):
    def __init__(self, config):
        super(GraphormerSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class GraphormerAttention(nn.Module):
    def __init__(self, config):
        super(GraphormerAttention, self).__init__()
        self.config = config
        self.self = GraphormerSelfAttention(config)
        self.output = GraphormerSelfOutput(config)
        if config.post_ln:
            self.LayerNorm_2d = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.LayerNorm_3d = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_tensor, attention_mask, attention_bias, mode='2d'):
        self_output = self.self(input_tensor, attention_mask, attention_bias, mode)
        self_output, layer_att = self_output
        attention_output = self.output(self_output, input_tensor)

        if mode == '2d' and self.config.post_ln:
            attention_output = self.LayerNorm_2d(attention_output)
        elif mode == '3d' and self.config.post_ln:
            attention_output = self.LayerNorm_3d(attention_output)
            
        return attention_output, layer_att


class GraphormerIntermediate(nn.Module):
    def __init__(self, config):
        super(GraphormerIntermediate, self).__init__()
        self.config = config
        if not config.post_ln:
            self.LayerNorm_2d = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.LayerNorm_3d = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states, mode='2d'):
        if mode == '2d' and not self.config.post_ln:
            hidden_states = self.LayerNorm_2d(hidden_states)
        elif mode == '3d' and not self.config.post_ln:
            hidden_states = self.LayerNorm_3d(hidden_states)
        
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class GraphormerOutput(nn.Module):
    def __init__(self, config):
        super(GraphormerOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class GraphormerLayer(nn.Module):
    def __init__(self, config):
        super(GraphormerLayer, self).__init__()
        self.config = config
        self.attention = GraphormerAttention(config)
        self.intermediate = GraphormerIntermediate(config)
        self.output = GraphormerOutput(config)
        if config.post_ln:
            self.LayerNorm_2d = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.LayerNorm_3d = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask, attention_bias, mode='2d'):
        attention_output = self.attention(hidden_states, attention_mask, attention_bias, mode)
        attention_output, layer_att = attention_output
        intermediate_output = self.intermediate(attention_output, mode)
        layer_output = self.output(intermediate_output, attention_output)

        if mode == '2d' and self.config.post_ln:
            layer_output = self.LayerNorm_2d(layer_output)
        elif mode == '3d' and self.config.post_ln:
            layer_output = self.LayerNorm_3d(layer_output)
            
        return layer_output, layer_att


class GraphormerEncoder(nn.Module):
    def __init__(self, config):
        super(GraphormerEncoder, self).__init__()
        layer = GraphormerLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, attention_bias, mode='2d'):
        # attention_bias: [bs, layers, heads, nodes+1, nodes+1]
        all_encoder_layers = []
        all_encoder_att = []
        for i, layer_module in enumerate(self.layer):
            all_encoder_layers.append(hidden_states)
            hidden_states = layer_module(all_encoder_layers[i], attention_mask, attention_bias[:, i, :, :, :], mode)
            hidden_states, layer_att = hidden_states
            all_encoder_att.append(layer_att)
        all_encoder_layers.append(hidden_states)
        return all_encoder_layers, all_encoder_att


class GraphormerPooler(nn.Module):
    def __init__(self, config):
        super(GraphormerPooler, self).__init__()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = first_token_tensor
        return pooled_output


class GraphormerPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(GraphormerPreTrainedModel, self).__init__()
        if not isinstance(config, GraphormerConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)        
        elif isinstance(module, (GraphormerLayerNorm, nn.LayerNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, (nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class GraphormerModel(GraphormerPreTrainedModel):
    def __init__(self, config):
        super(GraphormerModel, self).__init__(config)
        self.embeddings_2d = GraphormerEmbeddings(config)
        self.embeddings_3d = GraphormerEmbeddings(config)
        self.attention_2d_bias = Graphormer2DAttentionBias(config)
        self.attention_3d_bias = Graphormer3DAttentionBias(config)
        self.encoder = GraphormerEncoder(config)
        if not config.post_ln:
            self.LayerNorm_2d = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.LayerNorm_3d = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler_2d = GraphormerPooler(config)
        self.pooler_3d = GraphormerPooler(config)

        self.apply(self._init_weights)

    def forward(self, batched_data, mode='2d', return_attention=False, pretraining=False):

        batch_size, nodes, hid = batched_data['x'].shape
        attention_mask = batched_data['attention_mask']
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, nodes + 1)
            
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -1e6

        if mode == '2d':
            attention_2d_bias = self.attention_2d_bias(batched_data)
            attention_bias = attention_2d_bias
            pos_encoding = None
            embedding_output = self.embeddings_2d(batched_data, pos_encoding)
        elif mode == '3d':
            attention_3d_bias, pos_encoding, delta_pos = self.attention_3d_bias(batched_data)
            attention_bias = attention_3d_bias
            embedding_output = self.embeddings_3d(batched_data, pos_encoding)
        
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      attention_bias, mode)

        encoded_layers, attention_layers = encoded_layers

        last_hidden_states = encoded_layers[-1]

        if mode == '2d':
            if not self.config.post_ln:
                last_hidden_states = self.LayerNorm_2d(last_hidden_states)
            pooled_output = self.pooler_2d(last_hidden_states)
            if return_attention:
                return last_hidden_states, pooled_output, attention_layers
            elif pretraining:
                return last_hidden_states, pooled_output
            else:
                return last_hidden_states, pooled_output

        elif mode == '3d':
            if not self.config.post_ln:
                last_hidden_states = self.LayerNorm_3d(last_hidden_states)
            pooled_output = self.pooler_3d(last_hidden_states)
            # return the 3d attn_bias of last layer
            if return_attention:
                return last_hidden_states, pooled_output, attention_3d_bias[:, -1, :, :, :], delta_pos, attention_layers
            elif pretraining:
                return last_hidden_states, pooled_output, attention_3d_bias[:, -1, :, :, :], delta_pos
            else:
                return last_hidden_states, pooled_output, attention_3d_bias[:, -1, :, :, :], delta_pos

        
class GraphormerModelForClassification(GraphormerPreTrainedModel):
    def __init__(self, config, args, classes=1, dropout_p=0.1):
        super(GraphormerModelForClassification, self).__init__(config)

        self.graphormer = GraphormerModel(config)
        self.classification_head = nn.Sequential(nn.Dropout(p=dropout_p),
                                                    nn.Linear(512, classes))

    def forward(self, batched_data):
        _, graph_token = self.graphormer(batched_data)
        logits = self.classification_head(graph_token)

        return logits


class GraphormerModelForRegression(GraphormerPreTrainedModel):
    def __init__(self, config, classes=1, dropout_p=0.1):
        super(GraphormerModelForRegression, self).__init__(config)
        self.graphormer = GraphormerModel(config)
        self.dropout = nn.Dropout(dropout_p)
        self.regression = nn.Linear(config.hidden_size, classes)

    def forward(self, batched_data):   
        _, graph_token = self.graphormer(batched_data)
        graph_token = self.dropout(graph_token)
        logits = self.regression(graph_token)
        return logits
        

class GraphormerModelForRegression_v2(GraphormerPreTrainedModel):
    def __init__(self, config, dropout_p=0.1):
        super(GraphormerModelForRegression, self).__init__(config)

        self.graphormer = GraphormerModel(config)
        self.dropout = nn.Dropout(dropout_p)
        self.regression = nn.Linear(config.hidden_size, 1)
        self.loss = nn.L1Loss(reduction='sum')

    def forward(self, batched_data, target=None):
        
        _, graph_token = self.graphormer(batched_data)
        pred = self.regression(graph_token).squeeze(-1)

        if target is None:
            return pred
        else:
            return self.loss(pred, target.reshape(-1))


class GraphormerNodesHeadTransform(nn.Module):
    def __init__(self, config):
        super(GraphormerNodesHeadTransform, self).__init__()
        self.config = config

        self.in_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.act_fn = gelu

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.denoise_scaling = (config.hidden_size // config.num_attention_heads) ** -0.5

        self.softmax = nn.Softmax(dim=-1)

        self.force_proj1 = nn.Linear(config.hidden_size, 1)
        self.force_proj2 = nn.Linear(config.hidden_size, 1)
        self.force_proj3 = nn.Linear(config.hidden_size, 1)

        self.dropout = nn.Dropout(config.nodes_attention_dropout_prob)

    def forward(self, hidden_states, attention_bias, delta_pos, attention_mask=None):
        '''
            x_i^{l+1} = x_i^l + \text{softmax}(QK^T+Bias)(x_i - x_j)V

        '''        
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -100000.0

        batch_size, nodes, hidden_size = hidden_states.shape
        extended_attention_mask = extended_attention_mask.repeat(1, self.config.num_attention_heads, nodes, 1)

        hidden_states = self.in_proj(hidden_states)
        hidden_states = self.act_fn(hidden_states)

        q = self.q_proj(hidden_states).view(batch_size, nodes, self.config.num_attention_heads, -1).transpose(1, 2) * math.sqrt(self.denoise_scaling)
        k = self.k_proj(hidden_states).view(batch_size, nodes, self.config.num_attention_heads, -1).transpose(1, 2) * math.sqrt(self.denoise_scaling)
        v = self.v_proj(hidden_states).view(batch_size, nodes, self.config.num_attention_heads, -1).transpose(1, 2)
        # [bs, heads, nodes, nodes]
        attention_score = q @ k.transpose(-1, -2)  
        attention_score = attention_score.view(-1, nodes, nodes) + \
                attention_bias.contiguous().view(-1, nodes, nodes) + extended_attention_mask.view(-1, nodes, nodes)
        # [bs*heads, nodes, nodes]
        attention_probs = self.softmax(attention_score)
        # [bs, heads, nodes, nodes]
        attention_probs = self.dropout(attention_probs).reshape(batch_size, self.config.num_attention_heads, nodes, nodes)

        # [bs, heads, nodes, nodes, 1] * [bs, 1, nodes, nodes, 3] -> [bs, heads, nodes, nodes, 3]
        rot_attn_probs = attention_probs.unsqueeze(-1) * delta_pos.unsqueeze(1).type_as(attention_probs)  
        # [bs, heads, 3, nodes, nodes]
        rot_attn_probs = rot_attn_probs.permute(0, 1, 4, 2, 3)

        # [bs, heads, 3, nodes, nodes] @ [bs, heads, 1, nodes, hid] -> [bs, heads, 3, nodes, hid]
        x = rot_attn_probs @ v.unsqueeze(2)  
        # [bs, nodes, 3, heads, hid] -> [bs, nodes, 3, hid]
        x = x.permute(0, 3, 2, 1, 4).contiguous().view(batch_size, nodes, 3, -1)

        f1 = self.force_proj1(x[:, :, 0, :]).view(batch_size, nodes, 1)
        f2 = self.force_proj2(x[:, :, 1, :]).view(batch_size, nodes, 1)
        f3 = self.force_proj3(x[:, :, 2, :]).view(batch_size, nodes, 1)
        cur_force = torch.cat([f1, f2, f3], dim=-1).float()

        return cur_force


class GraphormerDenoisingHead(nn.Module):
    def __init__(self, config, args):
        super(GraphormerDenoisingHead, self).__init__()
        self.node_transform = GraphormerNodesHeadTransform(config)
        self.mse = nn.MSELoss(reduction='none')
        self.coor_head = nn.Linear(3, 3)
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
        self.cos = nn.CosineSimilarity(dim=-1)
        self.args = args
        self.loss_type = args.denoising_loss
        self.denoising_reduction = args.denoising_reduction
        self.global_denoise = args.global_denoise

    def forward(self, hidden_states, pos, pos_target, pos_mask, attention_bias, delta_pos, attention_mask=None):
        '''
        pred: [bs, nodes+1, 3]
        pos: [bs, nodes+1, 3]
        pos_target: [bs, nodes+1, 3]
        attention_bias: [bs, heads, nodes+1, nodes+1]
        '''
        if not self.global_denoise:
            hidden_states = hidden_states[:, 1:, :]
            attention_bias = attention_bias[:, :, 1:, 1:]
            attention_mask = attention_mask[:, 1:]
            delta_pos = delta_pos[:, 1:, 1:]
            pos_target = pos_target[:, 1:, :]
            pos_mask = pos_mask[:, 1:, :]

        pred = self.node_transform(hidden_states, attention_bias, delta_pos, attention_mask)
        pred = self.coor_head(pred)
        # pred = pred + pos

        if self.loss_type == 'regression':
            loss = self.smooth_l1(pred, pos_target)
            loss = torch.where(pos_mask, loss, torch.zeros(loss.shape).to(loss.device).to(loss.dtype)) 
            if self.denoising_reduction == 'sum':
                loss = torch.sum(loss) #/ torch.sum(pos_mask)
            else:
                loss = torch.mean(loss) #/ torch.sum(pos_mask)
        elif self.loss_type == 'cos':
            sim = self.cos(pred, pos_target)
            loss = 1 - loss
            loss = torch.where(pos_mask, sim, torch.zeros(sim.shape).to(sim.device).to(sim.dtype))
            if self.denoising_reduction == 'sum':
                loss = torch.sum(loss) #/ torch.sum(pos_mask)
            else:
                loss = torch.mean(loss) #/ torch.sum(pos_mask)
            
        return loss


class GraphormerCLHead(nn.Module):
    def __init__(self, config, reduction='sum', projection_dim=64, tau=0.1, normalize=False, CL_similarity_metric='infonce'):
        super(GraphormerCLHead, self).__init__()
        self.tau = tau
        self.normalize = normalize
        self.CL_similarity_metric = CL_similarity_metric
        if CL_similarity_metric == 'infonce':
            self.projection_2d = nn.Linear(config.hidden_size, projection_dim)
            self.projection_3d = nn.Linear(config.hidden_size, projection_dim)
        elif CL_similarity_metric == 'directclr':
            pass
        else:
            raise ValueError
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, pooled_output_2d, pooled_output_3d, feature=False):
        if self.CL_similarity_metric == 'infonce':
            pooled_cl_2d = self.projection_2d(pooled_output_2d)
            pooled_cl_3d = self.projection_3d(pooled_output_3d)
        elif self.CL_similarity_metric == 'directclr':
            pooled_cl_2d = pooled_output_2d[:, :self.projection_dim]
            pooled_cl_3d = pooled_output_3d[:, :self.projection_dim]

        if feature:
            return pooled_cl_2d, pooled_cl_3d
        
        if dist.is_initialized():
            pooled_cl_2d = torch.cat(torch.distributed.nn.all_gather(pooled_cl_2d), dim=0)
            pooled_cl_3d = torch.cat(torch.distributed.nn.all_gather(pooled_cl_3d), dim=0)

        CL_loss_1, CL_acc_1 = self.do_CL(pooled_cl_2d, pooled_cl_3d)
        CL_loss_2, CL_acc_2 = self.do_CL(pooled_cl_3d, pooled_cl_2d)

        return (CL_loss_1 + CL_loss_2) / 2, (CL_acc_1 + CL_acc_2) / 2


    def do_CL(self, X, Y):
        if self.normalize:
            X = F.normalize(X, dim=-1)
            Y = F.normalize(Y, dim=-1)

        B = X.shape[0]
        logits = torch.mm(X, Y.transpose(1, 0)) 
        logits = torch.div(logits, self.tau)
        labels = torch.arange(B).long().to(logits.device)

        CL_loss = self.criterion(logits, labels)
        pred = logits.argmax(dim=1, keepdim=False)
        CL_acc = pred.eq(labels).sum().detach().cpu().item() * 1. / B

        return CL_loss, CL_acc


class GraphormerInfoMotifHead(nn.Module):
    def __init__(self, config, reduce='sum', tau=0.1, sample_n=1, neg_n=50, node_proj_dim=64):
        super(GraphormerInfoMotifHead, self).__init__()
        self.reduce = reduce
        self.tau = tau
        self.pos_n = sample_n
        self.neg_n = neg_n
        self.node_proj_dim = node_proj_dim
        self.node_proj = nn.Linear(config.hidden_size, node_proj_dim)

    def forward(self, hidden_states, pos_col_indices, num_atoms, attention_mask=None):
        hidden_states = self.node_proj(hidden_states[:, 1:, :])
        node_emb = torch.einsum('ijk,ij->ijk', hidden_states, attention_mask[:, 1:])

        batch_size, max_atoms, hid = node_emb.shape
        pos_row_indices = row_sample(batch_size, max_atoms, sample_n=self.pos_n)
        neg_row_indices, neg_col_indices = fast_negative_sample(batch_size, max_atoms, self.neg_n)
        pos = node_emb[pos_row_indices.view(-1), pos_col_indices.view(-1), :].reshape(batch_size, max_atoms, -1, hid)
        neg = node_emb[neg_row_indices.view(-1), neg_col_indices.view(-1), :].reshape(batch_size, max_atoms, -1, hid)

        loss, acc = self.info_nce_v5(node_emb, pos, neg, attention_mask, self.reduce)

        return loss, acc

    def info_nce_v4(self, anchor, pos, neg, mask=None, reduce='sum'):
        '''
        anchor: [bs, nodes, hid]
        pos: [bs, nodes, pos_n, hid]
        neg: [bs, nodes, neg_n, hid]
        '''
        anchor = F.normalize(anchor, dim=-1)
        pos = F.normalize(pos, dim=-1).detach()
        neg = F.normalize(neg, dim=-1).detach()

        pos_logits_n = torch.einsum('ijk,ijpk->ijp', anchor, pos)  # [bs, nodes, pos_n]
        pos_logits_n = torch.where(pos_logits_n.float().abs() < 1e-5, -9e5, pos_logits_n.float())
        pos_logits = torch.exp(pos_logits_n / self.tau)  
        pos_logits = pos_logits.sum(-1)  # [bs, nodes]

        neg_logits_n = torch.einsum('ijk,ijnk->ijn', anchor, neg)  # [bs, nodes, neg_n]
        neg_logits_n = torch.where(neg_logits_n.float().abs() < 1e-5, -9e5, neg_logits_n.float())
        neg_logits = torch.exp(neg_logits_n / self.tau)  
        neg_logits = neg_logits.sum(-1)  # [bs, nodes]
        neg_logits = neg_logits + pos_logits

        loss = pos_logits / neg_logits
        loss = loss.reshape(-1)
        if mask is not None:
            mask = mask[:, 1:].reshape(-1)
            loss = torch.log(loss)
            loss = scatter(loss, mask, dim=0, reduce=reduce)[1]
            loss = -loss

        else:
            loss = -torch.log(loss).mean()

        acc = self.get_multipul_acc(pos_logits_n, neg_logits_n, mask)

        return loss, acc

    def info_nce_v5(self, anchor, pos, neg, mask=None, reduce='sum'):
        '''
        anchor: [bs, nodes, hid]
        pos: [bs, nodes, pos_n, hid]
        neg: [bs, nodes, neg_n, hid]
        '''
        anchor = F.normalize(anchor, dim=-1)
        pos = F.normalize(pos, dim=-1).detach()
        neg = F.normalize(neg, dim=-1).detach()

        pos_logits_n = torch.einsum('ijk,ijpk->ijp', anchor, pos)  # [bs, nodes, pos_n]
        pos_logits_n = torch.where(pos_logits_n.float().abs() < 1e-5,
                                    torch.zeros_like(pos_logits_n, dtype=torch.float32).fill_(-9).to(pos_logits_n.device), 
                                    pos_logits_n.float())
        pos_logits = torch.exp(pos_logits_n / self.tau)  
        pos_logits = pos_logits.sum(-1)  # [bs, nodes]

        neg_logits_n = torch.einsum('ijk,ijnk->ijn', anchor, neg)  # [bs, nodes, neg_n]
        neg_logits_n = torch.where(neg_logits_n.float().abs() < 1e-5, 
                                   torch.zeros_like(neg_logits_n, dtype=torch.float32).fill_(-9).to(neg_logits_n.device), 
                                   neg_logits_n.float())
        neg_logits = torch.exp(neg_logits_n / self.tau)  
        neg_logits = neg_logits.sum(-1)  # [bs, nodes]
        neg_logits = neg_logits + pos_logits + 1e-5

        loss = pos_logits / neg_logits
        loss = torch.log(loss)
        mask = mask[:, 1:]
        loss = torch.where(mask.bool(), loss, torch.zeros(loss.shape).to(loss.device).to(loss.dtype))
        loss = -torch.sum(loss)

        acc = self.get_multipul_acc(pos_logits_n, neg_logits_n, mask)

        return loss, acc

    def get_multipul_acc(self, pos_logits, neg_logits, mask=None):
        pos_neg_logits = torch.cat([pos_logits, neg_logits], dim=-1)  # [bs, nodes, pos+neg]
        _, indices = torch.topk(pos_neg_logits, self.pos_n, dim=-1)  # [bs, nodes, top_posn]
        indices = indices.reshape(-1, self.pos_n)  # [bs*nodes, top_posn]
        
        correct = indices.sum(-1)[mask.reshape(-1).bool()].eq(1).sum()

        acc = correct / mask.sum()

        return acc.item()


class GraphormerCLandProtoHead(nn.Module):
    def __init__(self, config, reduction='sum', projection_dim=64, tau=0.1, normalize=True, target_T=1.0, CL_similarity_metric='infonce', CL_projection='nonlinear', spherical=False, PBT=False):
        super(GraphormerCLandProtoHead, self).__init__()
        self.tau = tau
        self.normalize = normalize
        self.target_T = target_T
        self.projection_dim = projection_dim
        self.CL_similarity_metric = CL_similarity_metric
        self.spherical = spherical
        self.PBT = PBT
        if CL_similarity_metric == 'infonce' and CL_projection == 'nonlinear':
            self.projection_2d = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                               nn.Tanh(),
                                               nn.Linear(config.hidden_size, projection_dim))
            self.projection_3d = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                               nn.Tanh(),
                                               nn.Linear(config.hidden_size, projection_dim))
        elif CL_similarity_metric == 'infonce' and CL_projection == 'linear':
            self.projection_2d = nn.Linear(config.hidden_size, projection_dim)
            self.projection_3d = nn.Linear(config.hidden_size, projection_dim)
        elif CL_similarity_metric == 'directclr':
            pass
        else:
            raise ValueError
        
        self.proto_proj_2d = nn.Linear(config.hidden_size, projection_dim)
        self.proto_proj_3d = nn.Linear(config.hidden_size, projection_dim)

        self.criterion = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, pooled_output_2d, pooled_output_3d, feature=False, 
                mol_2d_centroids_1=None, mol_2d_labels_1=None,
                mol_2d_centroids_2=None, mol_2d_labels_2=None,
                mol_2d_centroids_3=None, mol_2d_labels_3=None,
                mol_3d_centroids_1=None, mol_3d_labels_1=None,
                mol_3d_centroids_2=None, mol_3d_labels_2=None,
                mol_3d_centroids_3=None, mol_3d_labels_3=None,):
        
        if feature and self.CL_similarity_metric == 'infonce':
            if self.spherical:
                pooled_pt_2d = self.proto_proj_2d(pooled_output_2d)
                pooled_pt_3d = self.proto_proj_3d(pooled_output_3d)
                pooled_pt_2d = F.normalize(pooled_pt_2d, dim=-1)
                pooled_pt_3d = F.normalize(pooled_pt_3d, dim=-1)
            else:
                pooled_pt_2d = self.proto_proj_2d(pooled_output_2d)
                pooled_pt_3d = self.proto_proj_3d(pooled_output_3d)

            return pooled_pt_2d, pooled_pt_3d 
        elif feature and self.CL_similarity_metric == 'directclr':
            pooled_pt_2d = pooled_output_2d[:, :self.projection_dim].contiguous()
            pooled_pt_3d = pooled_output_3d[:, :self.projection_dim].contiguous()
            return pooled_pt_2d, pooled_pt_3d
        
        if self.CL_similarity_metric == 'infonce':
            pooled_2d = self.projection_2d(pooled_output_2d)
            pooled_3d = self.projection_3d(pooled_output_3d)
        elif self.CL_similarity_metric == 'directclr':
            pooled_2d = pooled_output_2d[:, :self.projection_dim]
            pooled_3d = pooled_output_3d[:, :self.projection_dim]

        if self.normalize:
            pooled_cl_2d = F.normalize(pooled_2d, dim=-1)
            pooled_cl_3d = F.normalize(pooled_3d, dim=-1)
        
        # do CL
        if dist.is_initialized():
            gathered_pooled_cl_2d = torch.cat(torch.distributed.nn.all_gather(pooled_cl_2d), dim=0)
            gathered_pooled_cl_3d = torch.cat(torch.distributed.nn.all_gather(pooled_cl_3d), dim=0)

            CL_loss_1, CL_acc_1 = self.do_CL(gathered_pooled_cl_2d, gathered_pooled_cl_3d)
            CL_loss_2, CL_acc_2 = self.do_CL(gathered_pooled_cl_3d, gathered_pooled_cl_2d)
        else:
            CL_loss_1, CL_acc_1 = self.do_CL(pooled_cl_2d, pooled_cl_3d)
            CL_loss_2, CL_acc_2 = self.do_CL(pooled_cl_3d, pooled_cl_2d)

        CL_loss = (CL_loss_1 + CL_loss_2) / 2
        CL_acc = (CL_acc_1 + CL_acc_2) / 2

        if mol_2d_centroids_1 is None:
            return CL_loss, CL_acc
        else:
            if not self.spherical:
                pooled_pt_2d = self.proto_proj_2d(pooled_output_2d)
                pooled_pt_3d = self.proto_proj_3d(pooled_output_3d)
            else:
                pooled_pt_2d = pooled_cl_2d
                pooled_pt_3d = pooled_cl_3d
            
            if self.PBT:
                # do 2D Proto
                protoloss_2d_1, proto2d_acc_1 = self.do_proto(pooled_pt_2d, mol_2d_centroids_1, mol_3d_labels_1)
                protoloss_2d_2, proto2d_acc_2 = self.do_proto(pooled_pt_2d, mol_2d_centroids_2, mol_3d_labels_2)
                protoloss_2d_3, proto2d_acc_3 = self.do_proto(pooled_pt_2d, mol_2d_centroids_3, mol_3d_labels_3)

                # do 3D proto
                protoloss_3d_1, proto3d_acc_1 = self.do_proto(pooled_pt_3d, mol_3d_centroids_1, mol_2d_labels_1)
                protoloss_3d_2, proto3d_acc_2 = self.do_proto(pooled_pt_3d, mol_3d_centroids_2, mol_2d_labels_2)
                protoloss_3d_3, proto3d_acc_3 = self.do_proto(pooled_pt_3d, mol_3d_centroids_3, mol_2d_labels_3)
            else:
                # do 2D Proto
                protoloss_2d_1, proto2d_acc_1 = self.do_proto(pooled_pt_2d, mol_2d_centroids_1, mol_2d_labels_1)
                protoloss_2d_2, proto2d_acc_2 = self.do_proto(pooled_pt_2d, mol_2d_centroids_2, mol_2d_labels_2)
                protoloss_2d_3, proto2d_acc_3 = self.do_proto(pooled_pt_2d, mol_2d_centroids_3, mol_2d_labels_3)

                # do 3D proto
                protoloss_3d_1, proto3d_acc_1 = self.do_proto(pooled_pt_3d, mol_3d_centroids_1, mol_3d_labels_1)
                protoloss_3d_2, proto3d_acc_2 = self.do_proto(pooled_pt_3d, mol_3d_centroids_2, mol_3d_labels_2)
                protoloss_3d_3, proto3d_acc_3 = self.do_proto(pooled_pt_3d, mol_3d_centroids_3, mol_3d_labels_3)

            return CL_loss, CL_acc, protoloss_2d_1, protoloss_2d_2, protoloss_2d_3,\
                   protoloss_3d_1, protoloss_3d_2, protoloss_3d_3, \
                   proto2d_acc_1, proto2d_acc_2, proto2d_acc_3, \
                   proto3d_acc_1, proto3d_acc_2, proto3d_acc_3


    def do_CL(self, X, Y):

        B = X.shape[0]
        logits = torch.mm(X, Y.transpose(1, 0))  # B*B
        logits = torch.div(logits, self.tau)
        labels = torch.arange(B).long().to(logits.device)

        CL_loss = self.criterion(logits, labels)
        pred = logits.argmax(dim=1, keepdim=False)
        CL_acc = pred.eq(labels).sum().detach().cpu().item() * 1. / B

        return CL_loss, CL_acc
    

    def do_proto(self, feature, centroids, labels):
        '''
        feature: [bs, hid]
        centroids: [k, hid]
        labels: [bs]
        '''
        B = feature.shape[0]
        pred_scores = torch.einsum('ij,kj->ik', feature, centroids)
        loss = F.cross_entropy(pred_scores, labels, reduction='sum')
        acc = pred_scores.argmax(dim=-1, keepdim=False).eq(labels).sum().detach().cpu().item() * 1. / B

        return loss, acc


class GraphormerModelForCL(GraphormerPreTrainedModel):
    def __init__(self, config, args):
        super(GraphormerModelForCL, self).__init__(config)
        self.graphormer = GraphormerModel(config)
        self.CL_head = GraphormerCLHead(config, args.CL_reduction, args.projection_dim, args.CL_T, args.normalize)

        self.apply(self._init_weights)

    def forward(self, batched_data):
        attention_mask = batched_data['attention_mask']
        node_2d, graph_2d = self.graphormer(batched_data, mode='2d')
        node_3d, graph_3d, _, _ = self.graphormer(batched_data, mode='3d')

        loss, acc = self.CL_head(graph_2d, graph_3d)

        return loss, acc


class GraphormerModelForInfoMotif(GraphormerPreTrainedModel):
    def __init__(self, config, args=None):
        super(GraphormerModelForInfoMotif, self).__init__(config)
        self.graphormer = GraphormerModel(config)
        self.infomotif_head_2d = GraphormerInfoMotifHead(config, 
                                                         reduce=args.infomotif_reduction, 
                                                         tau=args.tau, 
                                                         sample_n=args.pos_n, 
                                                         neg_n=args.neg_n,
                                                         node_proj_dim=args.node_proj_dim)
        self.infomotif_head_3d = GraphormerInfoMotifHead(config, 
                                                         reduce=args.infomotif_reduction, 
                                                         tau=args.tau, 
                                                         sample_n=args.pos_n, 
                                                         neg_n=args.neg_n,
                                                         node_proj_dim=args.node_proj_dim)       

        self.apply(self._init_weights)                                          

    def forward(self, batched_data):
        node_3d, graph_3d, attention_bias_3d, delta_pos = self.graphormer(batched_data, mode='3d')
        node_2d, graph_2d = self.graphormer(batched_data, mode='2d')      

        pos_col_indices = batched_data['sample_col_indices']
        num_atoms = batched_data['num_atoms']

        loss_infomotif_2d, acc_2d = self.infomotif_head_2d(node_2d, \
            pos_col_indices, num_atoms, batched_data['attention_mask'])
        loss_infomotif_3d, acc_3d = self.infomotif_head_3d(node_3d, \
            pos_col_indices, num_atoms, batched_data['attention_mask'])

        return loss_infomotif_2d, loss_infomotif_3d, acc_2d, acc_3d


class GraphormerModelForCLandIM(GraphormerPreTrainedModel):
    def __init__(self, config, args=None):
        super(GraphormerModelForCLandIM, self).__init__(config)
        self.args = args
        self.graphormer = GraphormerModel(config)
        if args.do_denoising:
            self.denoising_head = GraphormerDenoisingHead(config, args)
        self.infomotif_head_2d = GraphormerInfoMotifHead(config, 
                                                         reduce=args.infomotif_reduction, 
                                                         tau=args.tau, 
                                                         sample_n=args.pos_n, 
                                                         neg_n=args.neg_n,
                                                         node_proj_dim=args.node_proj_dim)
        self.cl_proto_head = GraphormerCLandProtoHead(config,
                                                reduction=args.CL_reduction,
                                                projection_dim=args.projection_dim,
                                                tau=args.CL_T,
                                                normalize=args.normalize,
                                                target_T=args.target_T,
                                                CL_similarity_metric=args.CL_similarity_metric,
                                                CL_projection=args.CL_projection)

        # MTL weights
        self.denoising_weight = args.denoising_weight
        self.CL_weight = args.CL_weight
        self.infomotif_2d_weight = args.infomotif_2d_weight
        self.norm_weight = args.norm_weight

        self.apply(self._init_weights)

    def forward(self, batched_data, feature=False):
        
        if feature:
            node_2d, graph_2d = self.graphormer(batched_data, mode='2d')
            node_3d, graph_3d, attention_bias_3d, delta_pos = self.graphormer(batched_data, mode='3d')
            graph_2d, graph_3d = self.cl_proto_head(graph_2d, graph_3d, feature=feature)
            return graph_2d, graph_3d


        node_2d, graph_2d = self.graphormer(batched_data, mode='2d', pretraining=True)
        node_3d, graph_3d, attention_bias_3d, delta_pos = self.graphormer(batched_data, mode='3d', pretraining=True)

        # for denoising
        if self.args.do_denoising:
            loss_denoise = self.denoising_head(node_3d, batched_data['pos'], batched_data['pos_target'], batched_data['pos_mask'],
                                               attention_bias_3d, delta_pos, batched_data['attention_mask'])
        else:
            loss_denoise = None

        # for InfoMotif
        pos_col_indices = batched_data['sample_col_indices']
        num_atoms = batched_data['num_atoms']
        loss_infomotif_2d, im2d_acc = self.infomotif_head_2d(node_2d, pos_col_indices, num_atoms, batched_data['attention_mask'])

        # for CL 
        loss_CL, CL_acc = self.cl_proto_head(graph_2d, graph_3d)


        if loss_denoise is not None:
            # order: first loss then acc 
            return loss_CL * self.CL_weight, \
                   loss_infomotif_2d * self.infomotif_2d_weight, \
                   loss_denoise * self.denoising_weight, \
                   CL_acc, \
                   im2d_acc
        else:
            return loss_CL * self.CL_weight, \
                   loss_infomotif_2d * self.infomotif_2d_weight, \
                   CL_acc, \
                   im2d_acc


class GraphormerModelForPretraining(GraphormerPreTrainedModel):
    def __init__(self, config, args=None):
        super().__init__(config)
        self.graphormer = GraphormerModel(config)
        self.args = args
        self.infomotif_head_2d = GraphormerInfoMotifHead(config, 
                                                         reduce=args.infomotif_reduction, 
                                                         tau=args.tau, 
                                                         sample_n=args.pos_n, 
                                                         neg_n=args.neg_n,
                                                         node_proj_dim=args.node_proj_dim)
        if args.do_denoising:
            self.denoising_head = GraphormerDenoisingHead(config, args)
        self.cl_proto_head = GraphormerCLandProtoHead(config,
                                                      reduction=args.proto_reduction,
                                                      projection_dim=args.projection_dim,
                                                      tau=args.CL_T,
                                                      normalize=args.normalize,
                                                      target_T=args.target_T,
                                                      CL_similarity_metric=args.CL_similarity_metric,
                                                      CL_projection=args.CL_projection,
                                                      spherical=args.spherical,
                                                      PBT=args.PBT)

        # MTL weights
        self.denoising_weight = args.denoising_weight * 1
        self.CL_weight = args.CL_weight # * 1
        self.infomotif_2d_weight = args.infomotif_2d_weight * 1
        self.proto_weight = args.proto_weight

        self.apply(self._init_weights)

    def forward(self, batched_data, feature=False, clustering=None):
        
        if feature:
            node_2d, graph_2d = self.graphormer(batched_data, mode='2d')
            node_3d, graph_3d, attention_bias_3d, delta_pos = self.graphormer(batched_data, mode='3d')
            graph_2d, graph_3d = self.cl_proto_head(graph_2d, graph_3d, feature=feature)
            return graph_2d, graph_3d


        node_2d, graph_2d = self.graphormer(batched_data, mode='2d')
        node_3d, graph_3d, attention_bias_3d, delta_pos = self.graphormer(batched_data, mode='3d')

        # for denoising
        if self.args.do_denoising:
            loss_denoise = self.denoising_head(node_3d, batched_data['pos'], batched_data['pos_target'], batched_data['pos_mask'],
                                               attention_bias_3d, delta_pos, batched_data['attention_mask'])
        else:
            loss_denoise = None

        # for InfoMotif
        pos_col_indices = batched_data['sample_col_indices']
        num_atoms = batched_data['num_atoms']
        loss_infomotif_2d, im2d_acc = self.infomotif_head_2d(node_2d, pos_col_indices, num_atoms, batched_data['attention_mask'])

        # for CL and prototypical clustering
        mol_2d_centroids_1, mol_2d_labels_1 = clustering.mol_2d_centroids_1, clustering.mol_2d_labels_1[batched_data['episodic_index']].to(graph_2d.device)
        mol_2d_centroids_2, mol_2d_labels_2 = clustering.mol_2d_centroids_2, clustering.mol_2d_labels_2[batched_data['episodic_index']].to(graph_2d.device)
        mol_2d_centroids_3, mol_2d_labels_3 = clustering.mol_2d_centroids_3, clustering.mol_2d_labels_3[batched_data['episodic_index']].to(graph_2d.device)

        mol_3d_centroids_1, mol_3d_labels_1 = clustering.mol_3d_centroids_1, clustering.mol_3d_labels_1[batched_data['episodic_index']].to(graph_2d.device)
        mol_3d_centroids_2, mol_3d_labels_2 = clustering.mol_3d_centroids_2, clustering.mol_3d_labels_2[batched_data['episodic_index']].to(graph_2d.device)
        mol_3d_centroids_3, mol_3d_labels_3 = clustering.mol_3d_centroids_3, clustering.mol_3d_labels_3[batched_data['episodic_index']].to(graph_2d.device)

        temp_return = self.cl_proto_head(pooled_output_2d=graph_2d, 
                                         pooled_output_3d=graph_3d,
                                         feature=feature,
                                         mol_2d_centroids_1=mol_2d_centroids_1, 
                                         mol_2d_labels_1=mol_2d_labels_1,
                                         mol_2d_centroids_2=mol_2d_centroids_2, 
                                         mol_2d_labels_2=mol_2d_labels_2,
                                         mol_2d_centroids_3=mol_2d_centroids_3, 
                                         mol_2d_labels_3=mol_2d_labels_3,
                                         mol_3d_centroids_1=mol_3d_centroids_1, 
                                         mol_3d_labels_1=mol_3d_labels_1,
                                         mol_3d_centroids_2=mol_3d_centroids_2, 
                                         mol_3d_labels_2=mol_3d_labels_2,
                                         mol_3d_centroids_3=mol_3d_centroids_3, 
                                         mol_3d_labels_3=mol_3d_labels_3)

        # loss_CL, CL_acc, loss_proto2d, proto2d_acc, loss_proto3d, proto3d_acc = temp_return
        loss_CL, CL_acc, protoloss_2d_1, protoloss_2d_2, protoloss_2d_3,\
                   protoloss_3d_1, protoloss_3d_2, protoloss_3d_3, \
                   proto2d_acc_1, proto2d_acc_2, proto2d_acc_3, \
                   proto3d_acc_1, proto3d_acc_2, proto3d_acc_3 = temp_return
        
        return loss_CL * self.CL_weight, \
               loss_infomotif_2d * self.infomotif_2d_weight, \
               loss_denoise * self.denoising_weight, \
               protoloss_2d_1 * self.proto_weight, \
               protoloss_2d_2 * self.proto_weight, \
               protoloss_2d_3 * self.proto_weight, \
               protoloss_3d_1 * self.proto_weight, \
               protoloss_3d_2 * self.proto_weight, \
               protoloss_3d_3 * self.proto_weight, \
               CL_acc, im2d_acc, \
               proto2d_acc_1, proto2d_acc_2, proto2d_acc_3,\
               proto3d_acc_1, proto3d_acc_2, proto3d_acc_3,


if __name__ == '__main__':
    print(full_atom_feature_dims)
    print(full_bond_feature_dims)
