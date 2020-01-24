import torch
import torch.nn as nn
import torch.nn.functional as F

class BilinearAttention(nn.Module):
    def __init__(self, query_size, key_size, hidden_size):
        super().__init__()
        self.linear_key = nn.Linear(key_size, hidden_size, bias=False)
        self.linear_query = nn.Linear(query_size, hidden_size, bias=True)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.hidden_size=hidden_size

    def score(self, query, key, softmax_dim=-1, mask=None):
        attn=self.matching(query, key, mask)

        norm_attn = F.softmax(attn, dim=softmax_dim)

        if mask is not None:
            norm_attn = norm_attn.masked_fill(~mask, 0)

        return attn, norm_attn


    def matching(self, query, key, mask=None):
        '''
        :param query: [batch_size, *, query_seq_len, query_size]
        :param key: [batch_size, *, key_seq_len, key_size]
        :param mask: [batch_size, *, query_seq_len, key_seq_len]
        :return: [batch_size, *, query_seq_len, key_seq_len]
        '''
        wq = self.linear_query(query)
        wq = wq.unsqueeze(-2)

        uh = self.linear_key(key)
        uh = uh.unsqueeze(-3)

        wuc = wq + uh

        wquh = torch.tanh(wuc)

        attn = self.v(wquh).squeeze(-1)

        if mask is not None:
            attn = attn.masked_fill(~mask, -float('inf'))

        return attn

    def forward(self, query, key, value, mask=None):
        '''
        :param query: [batch_size, *, query_seq_len, query_size]
        :param key: [batch_size, *, key_seq_len, key_size]
        :param value: [batch_size, *, value_seq_len=key_seq_len, value_size]
        :param mask: [batch_size, *, query_seq_len, key_seq_len]
        :return: [batch_size, *, query_seq_len, value_size]
        '''

        attn, norm_attn = self.score(query, key, mask=mask)
        h = torch.bmm(norm_attn.view(-1, norm_attn.size(-2), norm_attn.size(-1)), value.view(-1, value.size(-2), value.size(-1)))

        return h.view(list(value.size())[:-2]+[norm_attn.size(-2), -1]), attn, norm_attn