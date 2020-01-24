import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
from Common.BilinearAttention import *

def gru_forward(gru, input, lengths, state=None, batch_first=True):
    gru.flatten_parameters()
    input_lengths, perm = torch.sort(lengths, descending=True)

    input = input[perm]
    if state is not None:
        state = state[perm].transpose(0, 1).contiguous()

    total_length=input.size(1)
    if not batch_first:
        input = input.transpose(0, 1)  # B x L x N -> L x B x N
    packed = torch.nn.utils.rnn.pack_padded_sequence(input, input_lengths, batch_first)

    outputs, state = gru(packed, state)
    outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=batch_first, total_length=total_length)  # unpack (back to padded)

    _, perm = torch.sort(perm, descending=False)
    if not batch_first:
        outputs = outputs.transpose(0, 1)
    outputs=outputs[perm]
    state = state.transpose(0, 1)[perm]

    return outputs, state

def build_map(b_map, max=None):
    batch_size, b_len = b_map.size()
    if max is None:
        max=b_map.max() + 1
    if torch.cuda.is_available():
        b_map_ = torch.cuda.FloatTensor(batch_size, b_len, max).fill_(0)
    else:
        b_map_ = torch.zeros(batch_size, b_len, max)
    b_map_.scatter_(2, b_map.unsqueeze(2), 1.)
    # b_map_[:, :, 0] = 0.
    b_map_.requires_grad=False
    return b_map_

class RepeatNet(nn.Module):
    def __init__(self, embedding_size, hidden_size, item_vocab_size):
        super(RepeatNet, self).__init__()

        self.embedding_size=embedding_size
        self.hidden_size=hidden_size
        self.item_vocab_size=item_vocab_size

        self.item_emb = nn.Embedding(item_vocab_size, embedding_size, padding_idx=0)

        self.enc = nn.GRU(embedding_size, int(hidden_size / 2), num_layers=1, bidirectional=True, batch_first=True)

        self.mode_attn = BilinearAttention(hidden_size, hidden_size, hidden_size)
        self.mode=nn.Linear(hidden_size, 2)

        self.repeat_attn = BilinearAttention(hidden_size, hidden_size, hidden_size)
        self.explore_attn = BilinearAttention(hidden_size, hidden_size, hidden_size)
        self.explore = nn.Linear(hidden_size, item_vocab_size)

    def model(self, data):
        batch_size=data['item_seq'].size(0)
        mask = data['item_seq'].ne(0)
        lengths = mask.float().sum(dim=-1).long()

        item_seq_embs = F.dropout(self.item_emb(data['item_seq']), p=0.5, training=self.training)

        output, state = gru_forward(self.enc, item_seq_embs, lengths, batch_first=True)
        state = F.dropout(state, p=0.5, training=self.training)
        output = F.dropout(output, p=0.5, training=self.training)

        explore_feature, attn, norm_attn = self.explore_attn(state.reshape(batch_size, -1).unsqueeze(1), output, output, mask=mask.unsqueeze(1))
        p_explore = self.explore(explore_feature.squeeze(1))
        explore_mask=torch.bmm((data['item_seq']>0).float().unsqueeze(1), data['source_map']).squeeze(1)
        p_explore = p_explore.masked_fill(explore_mask.bool(), float('-inf')) # not sure we need to mask this out, depends on experiment results
        p_explore = F.softmax(p_explore, dim=-1)

        _, p_repeat = self.repeat_attn.score(state.reshape(batch_size, -1).unsqueeze(1), output, mask=mask.unsqueeze(1))
        p_repeat=torch.bmm(p_repeat, data['source_map']).squeeze(1)

        mode_feature, attn, norm_attn = self.mode_attn(state.reshape(batch_size, -1).unsqueeze(1), output, output, mask=mask.unsqueeze(1))
        p_mode=F.softmax(self.mode(mode_feature.squeeze(1)), dim=-1)

        p = p_mode[:, 0].unsqueeze(-1)*p_explore + p_mode[:, 1].unsqueeze(-1)*p_repeat

        return p

    def do_train(self, data):
        scores=self.model(data)
        loss = F.nll_loss((scores+1e-8).log(), data['item_tgt'].reshape(-1), ignore_index=0)#0 is used as padding
        return loss

    def do_infer(self, data):
        scores = self.model(data)
        scores, index=torch.sort(scores, dim=-1, descending=True)
        return scores, index

    def forward(self, data, method='mle_train'):
        data['source_map'] = build_map(data['item_seq'], max=self.item_vocab_size)
        if method == 'train':
            return self.do_train(data)
        elif method == 'infer':
            return self.do_infer(data)