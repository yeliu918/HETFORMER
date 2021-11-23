from typing import List
import math
import torch
from torch import nn
import json
import torch.nn.functional as F
from hetformer.diagonaled_mm_tvm import diagonaled_mm as diagonaled_mm_tvm, mask_invalid_locations
from hetformer.sliding_chunks import sliding_chunks_matmul_qk, sliding_chunks_matmul_pv
from hetformer.sliding_chunks import sliding_chunks_no_overlap_matmul_qk, sliding_chunks_no_overlap_matmul_pv
# from transformers.models.roberta.modeling_roberta import RobertaConfig, RobertaModel, RobertaForMaskedLM
from transformers.modeling_roberta import RobertaConfig
from transformer_local import RobertaModel, RobertaForMaskedLM

class Heterformer(RobertaModel):
    def __init__(self, config):
        super(Heterformer, self).__init__(config)
        if config.attention_mode == 'n2':
            pass  # do nothing, use BertSelfAttention instead
        else:
            for i, layer in enumerate(self.encoder.layer):
                layer.attention.self = HeterformerSelfAttention(config, layer_id=i)


class HeterformerForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config):
        super(HeterformerForMaskedLM, self).__init__(config)
        if config.attention_mode == 'n2':
            pass  # do nothing, use BertSelfAttention instead
        else:
            for i, layer in enumerate(self.roberta.encoder.layer):
                layer.attention.self = HeterformerSelfAttention(config, layer_id=i)


class HeterformerConfig(RobertaConfig):
    def __init__(self, attention_window: List[int] = None, attention_dilation: List[int] = None,
                 autoregressive: bool = False, attention_mode: str = 'sliding_chunks', **kwargs):
        """
        Args:
            attention_window: list of attention window sizes of length = number of layers.
                window size = number of attention locations on each side.
                For an affective window size of 512, use `attention_window=[256]*num_layers`
                which is 256 on each side.
            attention_dilation: list of attention dilation of length = number of layers.
                attention dilation of `1` means no dilation.
            autoregressive: do autoregressive attention or have attention of both sides
            attention_mode: 'n2' for regular n^2 self-attention, 'tvm' for TVM implemenation of Heterformer
                selfattention, 'sliding_chunks' for another implementation of Heterformer selfattention
        """
        super().__init__(**kwargs)
        self.attention_window = attention_window
        self.attention_dilation = attention_dilation
        self.autoregressive = autoregressive
        self.attention_mode = attention_mode
        assert self.attention_mode in ['tvm', 'sliding_chunks', 'n2', 'sliding_chunks_no_overlap']


class HeterformerSelfAttention(nn.Module):
    def __init__(self, config, layer_id):
        super(HeterformerSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size

        self.query = nn.Linear(config.hidden_size, self.embed_dim)
        self.key = nn.Linear(config.hidden_size, self.embed_dim)
        self.value = nn.Linear(config.hidden_size, self.embed_dim)

        self.query_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.key_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.value_global = nn.Linear(config.hidden_size, self.embed_dim)

        self.query_entity = nn.Linear(config.hidden_size, self.embed_dim)
        self.key_entity = nn.Linear(config.hidden_size, self.embed_dim)
        self.value_entity = nn.Linear(config.hidden_size, self.embed_dim)

        self.dropout = config.attention_probs_dropout_prob

        self.layer_id = layer_id
        if isinstance(config.attention_window, str):
            config.attention_window = json.loads(config.attention_window)
        self.attention_window = config.attention_window[self.layer_id]
        self.attention_dilation = config.attention_dilation[self.layer_id]
        # self.attention_mode = config.autoregressive ##TODO: change back
        self.autoregressive = config.autoregressive
        self.attention_mode = config.attention_mode
        assert self.attention_window > 0
        assert self.attention_dilation > 0
        assert self.attention_mode in ['tvm', 'sliding_chunks', 'sliding_chunks_no_overlap']
        if self.attention_mode in ['sliding_chunks', 'sliding_chunks_no_overlap']:
            assert not self.autoregressive  # not supported
            assert self.attention_dilation == 1  # dilation is not supported

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        entity_mask = None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        '''
        The `attention_mask` is changed in `BertModel.forward` from 0, 1, 2 to
            -1: no attention
              0: local attention
            +1: global attention
        '''
        assert encoder_hidden_states is None, "`encoder_hidden_states` is not supported and should be None"
        assert encoder_attention_mask is None, "`encoder_attention_mask` is not supported and should be None"

        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(dim=2).squeeze(dim=1)
            key_padding_mask = attention_mask < 0
            extra_attention_mask = attention_mask > 0
            remove_from_windowed_attention_mask = attention_mask != 0

            num_extra_indices_per_batch = extra_attention_mask.long().sum(dim=1)
            max_num_extra_indices_per_batch = num_extra_indices_per_batch.max()
            if max_num_extra_indices_per_batch <= 0:
                extra_attention_mask = None
            else:
                # To support the case of variable number of global attention in the rows of a batch,
                # we use the following three selection masks to select global attention embeddings
                # in a 3d tensor and pad it to `max_num_extra_indices_per_batch`
                # 1) selecting embeddings that correspond to global attention
                extra_attention_mask_nonzeros = extra_attention_mask.nonzero(as_tuple=True)
                zero_to_max_range = torch.arange(0, max_num_extra_indices_per_batch,
                                                 device=num_extra_indices_per_batch.device)
                # mask indicating which values are actually going to be padding
                selection_padding_mask = zero_to_max_range < num_extra_indices_per_batch.unsqueeze(dim=-1)
                # 2) location of the non-padding values in the selected global attention
                selection_padding_mask_nonzeros = selection_padding_mask.nonzero(as_tuple=True)
                # 3) location of the padding values in the selected global attention
                selection_padding_mask_zeros = (selection_padding_mask == 0).nonzero(as_tuple=True)
        else:
            remove_from_windowed_attention_mask = None
            extra_attention_mask = None
            key_padding_mask = None

        hidden_states = hidden_states.transpose(0, 1)
        seq_len, bsz, embed_dim = hidden_states.size()
        assert embed_dim == self.embed_dim
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        q /= math.sqrt(self.head_dim)

        q = q.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        # attn_weights = (bsz, seq_len, num_heads, window*2+1)
        if self.attention_mode == 'tvm':
            q = q.float().contiguous()
            k = k.float().contiguous()
            attn_weights = diagonaled_mm_tvm(q, k, self.attention_window, self.attention_dilation, False, 0, False)
        elif self.attention_mode == "sliding_chunks":
            attn_weights = sliding_chunks_matmul_qk(q, k, self.attention_window, padding_value=0)
        elif self.attention_mode == "sliding_chunks_no_overlap":
            attn_weights = sliding_chunks_no_overlap_matmul_qk(q, k, self.attention_window, padding_value=0)
        else:
            raise False
        mask_invalid_locations(attn_weights, self.attention_window, self.attention_dilation, False)
        if remove_from_windowed_attention_mask is not None:
            # This implementation is fast and takes very little memory because num_heads x hidden_size = 1
            # from (bsz x seq_len) to (bsz x seq_len x num_heads x hidden_size)
            remove_from_windowed_attention_mask = remove_from_windowed_attention_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
            # cast to float/half then replace 1's with -inf
            float_mask = remove_from_windowed_attention_mask.type_as(q).masked_fill(remove_from_windowed_attention_mask, -10000.0)
            repeat_size = 1 if isinstance(self.attention_dilation, int) else len(self.attention_dilation)
            float_mask = float_mask.repeat(1, 1, repeat_size, 1)
            ones = float_mask.new_ones(size=float_mask.size())  # tensor of ones
            # diagonal mask with zeros everywhere and -inf inplace of padding
            if self.attention_mode == 'tvm':
                d_mask = diagonaled_mm_tvm(ones, float_mask, self.attention_window, self.attention_dilation, False, 0, False)
            elif self.attention_mode == "sliding_chunks":
                d_mask = sliding_chunks_matmul_qk(ones, float_mask, self.attention_window, padding_value=0)
            elif self.attention_mode == "sliding_chunks_no_overlap":
                d_mask = sliding_chunks_no_overlap_matmul_qk(ones, float_mask, self.attention_window, padding_value=0)

            attn_weights += d_mask
        assert list(attn_weights.size())[:3] == [bsz, seq_len, self.num_heads]
        assert attn_weights.size(dim=3) in [self.attention_window * 2 + 1, self.attention_window * 3]

        # the extra attention
        if extra_attention_mask is not None:
            selected_k = k.new_zeros(bsz, max_num_extra_indices_per_batch, self.num_heads, self.head_dim)
            selected_k[selection_padding_mask_nonzeros] = k[extra_attention_mask_nonzeros]
            # (bsz, seq_len, num_heads, max_num_extra_indices_per_batch)
            selected_attn_weights = torch.einsum('blhd,bshd->blhs', (q, selected_k))
            selected_attn_weights[selection_padding_mask_zeros[0], :, :, selection_padding_mask_zeros[1]] = -10000
            # concat to attn_weights
            # (bsz, seq_len, num_heads, extra attention count + 2*window+1)
            attn_weights = torch.cat((selected_attn_weights, attn_weights), dim=-1)
        attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)  # use fp32 for numerical stability
        if key_padding_mask is not None:
            # softmax sometimes inserts NaN if all positions are masked, replace them with 0
            attn_weights_float = torch.masked_fill(attn_weights_float, key_padding_mask.unsqueeze(-1).unsqueeze(-1), 0.0)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)
        v = v.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        attn = 0
        if extra_attention_mask is not None:
            selected_attn_probs = attn_probs.narrow(-1, 0, max_num_extra_indices_per_batch)
            selected_v = v.new_zeros(bsz, max_num_extra_indices_per_batch, self.num_heads, self.head_dim)
            selected_v[selection_padding_mask_nonzeros] = v[extra_attention_mask_nonzeros]
            # use `matmul` because `einsum` crashes sometimes with fp16
            # attn = torch.einsum('blhs,bshd->blhd', (selected_attn_probs, selected_v))
            attn = torch.matmul(selected_attn_probs.transpose(1, 2), selected_v.transpose(1, 2).type_as(selected_attn_probs)).transpose(1, 2)
            attn_probs = attn_probs.narrow(-1, max_num_extra_indices_per_batch, attn_probs.size(-1) - max_num_extra_indices_per_batch).contiguous()

        if self.attention_mode == 'tvm':
            v = v.float().contiguous()
            attn += diagonaled_mm_tvm(attn_probs, v, self.attention_window, self.attention_dilation, True, 0, False)
        elif self.attention_mode == "sliding_chunks":
            attn += sliding_chunks_matmul_pv(attn_probs, v, self.attention_window)
        elif self.attention_mode == "sliding_chunks_no_overlap":
            attn += sliding_chunks_no_overlap_matmul_pv(attn_probs, v, self.attention_window)
        else:
            raise False

        attn = attn.type_as(hidden_states)
        assert list(attn.size()) == [bsz, seq_len, self.num_heads, self.head_dim]
        attn = attn.transpose(0, 1).reshape(seq_len, bsz, embed_dim).contiguous()

        if entity_mask is not None:
            max_cluster = entity_mask.max()
            for cluster_id in range(1, max_cluster+1):
                cluster_index_hidden_mask = entity_mask == cluster_id
                num_cluster_per_batch = cluster_index_hidden_mask.long().sum(dim=1)
                max_num_cluster_per_batch = num_cluster_per_batch.max()
                if max_num_cluster_per_batch == 0:
                    continue
                batch_size = bsz #(cluster_index_hidden_mask.long().sum(dim=1)!=0).sum().long()
                cluster_attention_mask_nonzeros = cluster_index_hidden_mask.nonzero(as_tuple=True)
                new_hidden_states_idx = torch.arange(0, max_num_cluster_per_batch,
                                                     device=cluster_index_hidden_mask.device)
                selection_cluster_mask_zeros= []
                selection_cluster_mask_nonzeros = []
                for b_size in range(batch_size):
                    if num_cluster_per_batch[b_size] == 0:
                        continue
                    selection_cluster_mask_zeros.append(new_hidden_states_idx > cluster_index_hidden_mask.long().sum(dim=1)[b_size])
                    selection_cluster_mask_nonzeros.append(new_hidden_states_idx < cluster_index_hidden_mask.long().sum(dim=1)[b_size])
                if len(selection_cluster_mask_nonzeros) > 1:
                    selection_cluster_mask_zeros = (torch.stack(selection_cluster_mask_zeros) == 0).nonzero(as_tuple = True)
                    selection_cluster_mask_nonzeros = torch.stack(selection_cluster_mask_nonzeros).nonzero(as_tuple = True)
                else:
                    selection_cluster_mask_zeros = (torch.stack(selection_cluster_mask_zeros) == 0).nonzero(as_tuple = True)
                    selection_cluster_mask_nonzeros = torch.stack(selection_cluster_mask_nonzeros).nonzero(as_tuple = True)

                cluster_hidden_states = hidden_states.new_zeros(max_num_cluster_per_batch, batch_size, embed_dim)
                cluster_hidden_states[selection_cluster_mask_nonzeros[::-1]] = hidden_states[cluster_attention_mask_nonzeros[::-1]]

                q_en = self.query_entity(cluster_hidden_states)
                k_en = self.key_entity(cluster_hidden_states)
                v_en = self.value_entity(cluster_hidden_states)
                q_en /= math.sqrt(self.head_dim)

                q_en = q_en.contiguous().view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)  # (bsz*self.num_heads, max_num_extra_indices_per_batch, head_dim)
                k_en = k_en.contiguous().view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)  # bsz * self.num_heads, seq_len, head_dim)
                v_en = v_en.contiguous().view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)  # bsz * self.num_heads, seq_len, head_dim)
                attn_weights = torch.bmm(q_en, k_en.transpose(1, 2))
                if list(attn_weights.size()) != [batch_size * self.num_heads, max_num_cluster_per_batch, max_num_cluster_per_batch]:
                    print("check")

                attn_weights = attn_weights.view(batch_size, self.num_heads, max_num_cluster_per_batch, max_num_cluster_per_batch)
                attn_weights[selection_cluster_mask_zeros[0], :, selection_cluster_mask_zeros[1], :] = -10000.0
                # if key_padding_mask is not None:
                #     attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), -10000.0, )

                attn_weights = attn_weights.view(batch_size * self.num_heads, max_num_cluster_per_batch, max_num_cluster_per_batch)
                attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)  # use fp32 for numerical stability
                attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)
                selected_attn = torch.bmm(attn_probs, v_en)
                assert list(selected_attn.size()) == [batch_size * self.num_heads, max_num_cluster_per_batch, self.head_dim]
                selected_attn_4d = selected_attn.view(batch_size, self.num_heads, max_num_cluster_per_batch, self.head_dim)
                nonzero_selected_attn = selected_attn_4d[selection_cluster_mask_nonzeros[0], :, selection_cluster_mask_nonzeros[1]]
                attn[cluster_attention_mask_nonzeros[::-1]] = nonzero_selected_attn.view(len(selection_cluster_mask_nonzeros[0]), -1).type_as(hidden_states)

        # For this case, we'll just recompute the attention for these indices
        # and overwrite the attn tensor. TODO: remove the redundant computation
        if extra_attention_mask is not None:
            selected_hidden_states = hidden_states.new_zeros(max_num_extra_indices_per_batch, bsz, embed_dim)
            selected_hidden_states[selection_padding_mask_nonzeros[::-1]] = hidden_states[extra_attention_mask_nonzeros[::-1]]

            q = self.query_global(selected_hidden_states)
            k = self.key_global(hidden_states)
            v = self.value_global(hidden_states)
            q /= math.sqrt(self.head_dim)

            q = q.contiguous().view(max_num_extra_indices_per_batch, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # (bsz*self.num_heads, max_num_extra_indices_per_batch, head_dim)
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # bsz * self.num_heads, seq_len, head_dim)
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # bsz * self.num_heads, seq_len, head_dim)
            attn_weights = torch.bmm(q, k.transpose(1, 2))
            assert list(attn_weights.size()) == [bsz * self.num_heads, max_num_extra_indices_per_batch, seq_len]

            attn_weights = attn_weights.view(bsz, self.num_heads, max_num_extra_indices_per_batch, seq_len)
            attn_weights[selection_padding_mask_zeros[0], :, selection_padding_mask_zeros[1], :] = -10000.0
            if key_padding_mask is not None:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    -10000.0,
                )
            attn_weights = attn_weights.view(bsz * self.num_heads, max_num_extra_indices_per_batch, seq_len)
            attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)  # use fp32 for numerical stability
            attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)
            selected_attn = torch.bmm(attn_probs, v)
            assert list(selected_attn.size()) == [bsz * self.num_heads, max_num_extra_indices_per_batch, self.head_dim]

            selected_attn_4d = selected_attn.view(bsz, self.num_heads, max_num_extra_indices_per_batch, self.head_dim)
            nonzero_selected_attn = selected_attn_4d[selection_padding_mask_nonzeros[0], :, selection_padding_mask_nonzeros[1]]
            attn[extra_attention_mask_nonzeros[::-1]] = nonzero_selected_attn.view(len(selection_padding_mask_nonzeros[0]), -1).type_as(hidden_states)

        context_layer = attn.transpose(0, 1)
        if output_attentions:
            if extra_attention_mask is not None:
                # With global attention, return global attention probabilities only
                # batch_size x num_heads x max_num_global_attention_tokens x sequence_length
                # which is the attention weights from tokens with global attention to all tokens
                # It doesn't not return local attention
                # In case of variable number of global attantion in the rows of a batch,
                # attn_weights are padded with -10000.0 attention scores
                attn_weights = attn_weights.view(bsz, self.num_heads, max_num_extra_indices_per_batch, seq_len)
            else:
                # without global attention, return local attention probabilities
                # batch_size x num_heads x sequence_length x window_size
                # which is the attention weights of every token attending to its neighbours
                attn_weights = attn_weights.permute(0, 2, 1, 3)
        outputs = (context_layer, attn_weights) if output_attentions else (context_layer,)
        return outputs
