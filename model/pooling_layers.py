import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np


def get_last_hidden_state(backbone_outputs):
    last_hidden_state = backbone_outputs[0]
    return last_hidden_state


def get_all_hidden_states(backbone_outputs):
    all_hidden_states = torch.stack(backbone_outputs[1])
    return all_hidden_states


def get_input_ids(inputs):
    return inputs['input_ids']


def get_attention_mask(inputs):
    return inputs['attention_mask']


class MeanPooling(nn.Module):
    def __init__(self, backbone_config):
        super(MeanPooling, self).__init__()
        self.output_dim = backbone_config.hidden_size

    def forward(self, inputs, backbone_outputs):
        attention_mask = get_attention_mask(inputs)
        last_hidden_state = get_last_hidden_state(backbone_outputs)

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class MaxPooling(nn.Module):
    def __init__(self, backbone_config):
        super(MaxPooling, self).__init__()
        self.output_dim = backbone_config.hidden_size

    def forward(self, inputs, backbone_outputs):
        attention_mask = get_attention_mask(inputs)
        last_hidden_state = get_last_hidden_state(backbone_outputs)

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = -1e4
        max_embeddings, _ = torch.max(embeddings, dim=1)
        return max_embeddings


class MeanMaxPooling(nn.Module):
    def __init__(self, backbone_config):
        super(MeanMaxPooling, self).__init__()
        self.output_dim = 2 * backbone_config.hidden_size

        self.mean_pooler = MeanPooling(backbone_config)
        self.max_pooler = MaxPooling(backbone_config)

    def forward(self, inputs, backbone_outputs):
        mean_pooler = self.mean_pooler(inputs, backbone_outputs)
        max_pooler =  self.max_pooler(inputs, backbone_outputs)
        out = torch.concat([mean_pooler, max_pooler] , 1)
        return out


class LSTMPooling(nn.Module):
    def __init__(self, backbone_config, pooling_config):
        super(LSTMPooling, self).__init__()

        self.num_hidden_layers = backbone_config.num_hidden_layers
        self.hidden_size = backbone_config.hidden_size
        self.hidden_lstm_size = pooling_config.hidden_size
        self.dropout_rate = pooling_config.dropout_rate
        self.bidirectional = pooling_config.bidirectional

        self.output_dim = (2 * pooling_config.hidden_size
                           if self.bidirectional
                           else pooling_config.hidden_size)

        if pooling_config.is_lstm:
            self.lstm = nn.LSTM(self.hidden_size,
                                self.hidden_lstm_size,
                                bidirectional=self.bidirectional,
                                batch_first=True)
        else:
            self.lstm = nn.GRU(self.hidden_size,
                               self.hidden_lstm_size,
                               bidirectional=self.bidirectional,
                               batch_first=True)

        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, inputs, backbone_outputs):
        all_hidden_states = get_all_hidden_states(backbone_outputs)

        hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(1, self.num_hidden_layers + 1)], dim=-1)
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        out, _ = self.lstm(hidden_states, None)
        out = self.dropout(out[:, -1, :])
        return out


class WeightedLayerPooling(nn.Module):
    def __init__(self, backbone_config, pooling_config):
        super(WeightedLayerPooling, self).__init__()

        self.num_hidden_layers = backbone_config.num_hidden_layers
        self.layer_start = pooling_config.layer_start
        self.layer_weights = nn.Parameter(torch.tensor([1] * (self.num_hidden_layers + 1 - self.layer_start), dtype=torch.float))

        self.output_dim = backbone_config.hidden_size

    def forward(self, inputs, backbone_outputs):
        all_hidden_states = get_all_hidden_states(backbone_outputs)

        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average[:, 0]


class AttentionPooling(nn.Module):
    def __init__(self, backbone_config, pooling_config):
        super(AttentionPooling, self).__init__()
        self.num_hidden_layers = backbone_config.num_hidden_layers
        self.hidden_size = backbone_config.hidden_size
        self.hiddendim_fc = pooling_config.hiddendim_fc
        self.dropout = nn.Dropout(pooling_config.dropout)
        
        '''
        q_t = np.random.normal(loc=0.0, scale=0.1, size=(1, self.hidden_size))
        self.q = nn.Parameter(torch.from_numpy(q_t)).float()#.to(self.device)
        w_ht = np.random.normal(loc=0.0, scale=0.1, size=(self.hidden_size, self.hiddendim_fc))
        self.w_h = nn.Parameter(torch.from_numpy(w_ht)).float()#.to(self.device)
        '''
        
        self.q = nn.Parameter(torch.zeros((1, self.hidden_size), dtype=torch.float32))
        self.q.data.normal_(mean=0.0, std=0.1)
        self.w_h = nn.Parameter(torch.zeros((self.hidden_size, self.hiddendim_fc), dtype=torch.float32))
        self.w_h.data.normal_(mean=0.0, std=0.1)

        self.output_dim = self.hiddendim_fc

    def forward(self, inputs, backbone_outputs):
        all_hidden_states = get_all_hidden_states(backbone_outputs)

        hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(1, self.num_hidden_layers + 1)], dim=-1)
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        out = self.attention(hidden_states)
        out = self.dropout(out)
        return out

    def attention(self, h):
        v = torch.matmul(self.q, h.transpose(-2, -1)).squeeze(1)
        v = F.softmax(v, -1)
        v_temp = torch.matmul(v.unsqueeze(1), h).transpose(-2, -1)
        v = torch.matmul(self.w_h.transpose(1, 0), v_temp).squeeze(2)
        return v


class GeMTextPooling(nn.Module):
    def __init__(self, backbone_config, pooling_config):
        super(GeMTextPooling, self).__init__()

        self.dim = pooling_config.dim
        self.eps = pooling_config.eps
        self.feat_mult = 1

        self.p = Parameter(torch.ones(1) * pooling_config.p)

        self.output_dim = backbone_config.hidden_size

    def forward(self, inputs, backbone_output):
        attention_mask = get_attention_mask(inputs)
        last_hidden_state = get_last_hidden_state(backbone_output)

        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.shape)
        x = (last_hidden_state.clamp(min=self.eps) * attention_mask_expanded).pow(self.p).sum(self.dim)
        ret = x / attention_mask_expanded.sum(self.dim).clip(min=self.eps)
        ret = ret.pow(1 / self.p)
        return ret


def get_pooling_layer(config, backbone_config):
    if config.model.pooling.type == 'MeanPooling':
        return MeanPooling(backbone_config)
    if config.model.pooling.type == 'MaxPooling':
        return MaxPooling(backbone_config)
    if config.model.pooling.type == 'MeanMaxPooling':
        return MeanMaxPooling(backbone_config)
    if config.model.pooling.type == 'LSTMPooling':
        return LSTMPooling(backbone_config, config.model.pooling.lstm)
    if config.model.pooling.type == 'WeightedLayerPooling':
        return WeightedLayerPooling(backbone_config, config.model.pooling.weighted)
    if config.model.pooling.type == 'AttentionPooling':
        return AttentionPooling(backbone_config, config.model.pooling.attention)
    if config.model.pooling.type == 'GeMTextPooling':
        return GeMTextPooling(backbone_config, config.model.pooling.gemtext)
    
    raise ValueError(f'Invalid pooling type: {config.model.pooling.type}')
