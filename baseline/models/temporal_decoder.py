"""
Temporal Multimodal Decoder
modified from VITA: https://github.com/sukjunhwang/VITA
"""
from math import ceil
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from einops import rearrange, repeat
from .position_encoding import SeqEmbeddingSine, PositionEmbeddingSine1D


class SelfAttentionLayer(nn.Module):

	def __init__(self, d_model, nhead, dropout=0.0,
				 activation="relu", normalize_before=False):
		super().__init__()
		self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

		self.norm = nn.LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout)

		self.activation = _get_activation_fn(activation)
		self.normalize_before = normalize_before

		self._reset_parameters()
	
	def _reset_parameters(self):
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)

	def with_pos_embed(self, tensor, pos: Optional[Tensor]):
		return tensor if pos is None else tensor + pos

	def forward_post(self, tgt,
					 tgt_mask: Optional[Tensor] = None,
					 tgt_key_padding_mask: Optional[Tensor] = None,
					 query_pos: Optional[Tensor] = None):
		q = k = self.with_pos_embed(tgt, query_pos)
		tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
							  key_padding_mask=tgt_key_padding_mask)[0]
		tgt = tgt + self.dropout(tgt2)
		tgt = self.norm(tgt)

		return tgt

	def forward_pre(self, tgt,
					tgt_mask: Optional[Tensor] = None,
					tgt_key_padding_mask: Optional[Tensor] = None,
					query_pos: Optional[Tensor] = None):
		tgt2 = self.norm(tgt)
		q = k = self.with_pos_embed(tgt2, query_pos)
		tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
							  key_padding_mask=tgt_key_padding_mask)[0]
		tgt = tgt + self.dropout(tgt2)
		
		return tgt

	def forward(self, tgt,
				tgt_mask: Optional[Tensor] = None,
				tgt_key_padding_mask: Optional[Tensor] = None,
				query_pos: Optional[Tensor] = None):
		if self.normalize_before:
			return self.forward_pre(tgt, tgt_mask,
									tgt_key_padding_mask, query_pos)
		return self.forward_post(tgt, tgt_mask,
								 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

	def __init__(self, d_model, nhead, dropout=0.0,
				 activation="relu", normalize_before=False):
		super().__init__()
		self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

		self.norm = nn.LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout)

		self.activation = _get_activation_fn(activation)
		self.normalize_before = normalize_before

		self._reset_parameters()
	
	def _reset_parameters(self):
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)

	def with_pos_embed(self, tensor, pos: Optional[Tensor]):
		return tensor if pos is None else tensor + pos

	def forward_post(self, tgt, memory,
					 memory_mask: Optional[Tensor] = None,
					 memory_key_padding_mask: Optional[Tensor] = None,
					 pos: Optional[Tensor] = None,
					 query_pos: Optional[Tensor] = None):
		tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
								   key=self.with_pos_embed(memory, pos),
								   value=memory, attn_mask=memory_mask,
								   key_padding_mask=memory_key_padding_mask)[0]
		tgt = tgt + self.dropout(tgt2)
		tgt = self.norm(tgt)
		
		return tgt

	def forward_pre(self, tgt, memory,
					memory_mask: Optional[Tensor] = None,
					memory_key_padding_mask: Optional[Tensor] = None,
					pos: Optional[Tensor] = None,
					query_pos: Optional[Tensor] = None):
		tgt2 = self.norm(tgt)
		tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
								   key=self.with_pos_embed(memory, pos),
								   value=memory, attn_mask=memory_mask,
								   key_padding_mask=memory_key_padding_mask)[0]
		tgt = tgt + self.dropout(tgt2)

		return tgt

	def forward(self, tgt, memory,
				memory_mask: Optional[Tensor] = None,
				memory_key_padding_mask: Optional[Tensor] = None,
				pos: Optional[Tensor] = None,
				query_pos: Optional[Tensor] = None):
		if self.normalize_before:
			return self.forward_pre(tgt, memory, memory_mask,
									memory_key_padding_mask, pos, query_pos)
		return self.forward_post(tgt, memory, memory_mask,
								 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

	def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
				 activation="relu", normalize_before=False):
		super().__init__()
		# Implementation of Feedforward model
		self.linear1 = nn.Linear(d_model, dim_feedforward)
		self.dropout = nn.Dropout(dropout)
		self.linear2 = nn.Linear(dim_feedforward, d_model)

		self.norm = nn.LayerNorm(d_model)

		self.activation = _get_activation_fn(activation)
		self.normalize_before = normalize_before

		self._reset_parameters()
	
	def _reset_parameters(self):
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)

	def with_pos_embed(self, tensor, pos: Optional[Tensor]):
		return tensor if pos is None else tensor + pos

	def forward_post(self, tgt):
		tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
		tgt = tgt + self.dropout(tgt2)
		tgt = self.norm(tgt)
		return tgt

	def forward_pre(self, tgt):
		tgt2 = self.norm(tgt)
		tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
		tgt = tgt + self.dropout(tgt2)
		return tgt

	def forward(self, tgt):
		if self.normalize_before:
			return self.forward_pre(tgt)
		return self.forward_post(tgt)


def _get_activation_fn(activation):
	"""Return an activation function given a string"""
	if activation == "relu":
		return F.relu
	if activation == "gelu":
		return F.gelu
	if activation == "glu":
		return F.glu
	raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
	""" Very simple multi-layer perceptron (also called FFN)"""

	def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
		super().__init__()
		self.num_layers = num_layers
		h = [hidden_dim] * (num_layers - 1)
		self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

	def forward(self, x):
		for i, layer in enumerate(self.layers):
			x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
		return x


class TemporalMultimodalDecoder(nn.Module):

	def __init__(
		self,
		in_channels,
		aux_loss,
		hidden_dim: int,
		num_frame_queries: int,
		num_queries: int,
		nheads: int,
		dim_feedforward: int,
		enc_layers: int,
		dec_layers: int,
		enc_window_size: int,
		pre_norm: bool,
		enforce_input_project: bool,
	):
		"""
		NOTE: this interface is experimental.
		Args:
			in_channels: channels of the input features
			hidden_dim: Transformer feature dimension
			num_queries: number of queries
			nheads: number of heads
			dim_feedforward: feature dimension in feedforward network
			enc_layers: number of Transformer encoder layers
			dec_layers: number of Transformer decoder layers
			pre_norm: whether to use pre-LayerNorm or not
			enforce_input_project: add input project 1x1 conv even if input
				channels and hidden dim is identical
		"""
		super().__init__()

		# define Transformer decoder here
		self.num_heads = nheads
		self.num_layers = dec_layers
		self.transformer_self_attention_layers = nn.ModuleList()
		self.transformer_cross_attention_layers = nn.ModuleList()
		self.transformer_ffn_layers = nn.ModuleList()

		self.enc_layers = enc_layers
		self.time_enc_layers = dec_layers
		self.window_size = enc_window_size
		self.aux_loss = aux_loss

		if enc_layers > 0:
			self.enc_self_attn = nn.ModuleList()
			self.enc_ffn = nn.ModuleList()
			for _ in range(self.enc_layers):
				self.enc_self_attn.append(
					SelfAttentionLayer(
						d_model=hidden_dim,
						nhead=nheads,
						dropout=0.0,
						normalize_before=pre_norm,
					),
				)
				self.enc_ffn.append(
					FFNLayer(
						d_model=hidden_dim,
						dim_feedforward=dim_feedforward,
						dropout=0.0,
						normalize_before=pre_norm,
					)
				)

		self.text_pos = PositionEmbeddingSine1D(hidden_dim, normalize=True)

		for _ in range(self.num_layers):
			self.transformer_self_attention_layers.append(
				SelfAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm,)
			)

			self.transformer_cross_attention_layers.append(
				CrossAttentionLayer(d_model=hidden_dim,nhead=nheads, dropout=0.0, normalize_before=pre_norm,)
			)

			self.transformer_ffn_layers.append(
				FFNLayer(d_model=hidden_dim, dim_feedforward=dim_feedforward, dropout=0.0, normalize_before=pre_norm,)
			)

		self.time_cross_attn = Language2Time(d_model=hidden_dim, nhead=nheads,)
		self.text_proj = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim, bias=True), 
			nn.LayerNorm(hidden_dim, eps=1e-12), 
			nn.Dropout(0.01)
		)
		self.time_proj = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim, bias=True), 
			nn.LayerNorm(hidden_dim, eps=1e-12), 
			nn.Dropout(0.0)
		)

		self.decoder_norm = nn.LayerNorm(hidden_dim)

		self.num_queries = num_queries
		# learnable query features, use text_embed instead
		# self.query_feat = nn.Embedding(num_queries, hidden_dim)
		# learnable query p.e.
		self.query_embed = nn.Embedding(num_queries, hidden_dim)
		self.time_embed = SeqEmbeddingSine(128, hidden_dim)

		# self.fq_pos = nn.Embedding(num_frame_queries, hidden_dim)   # sequence level embeddings, each seq corresponds to 1 query
		self.relevance_norm = nn.LayerNorm(hidden_dim)
		self.dropout = nn.Dropout(0.0)

		if in_channels != hidden_dim or enforce_input_project:
			self.input_proj_dec = nn.Linear(hidden_dim, hidden_dim)
		else:
			self.input_proj_dec = nn.Sequential()
		self.src_embed = nn.Identity()

	def forward(self, frame_query, text_embed, B, T, text_features=None):
		"""
		L: Number of Layers.
		B: Batch size.
		T: Temporal window size. Number of frames per video.
		C: Channel size.
		fQ: Number of frame-wise queries from IFC.
		cQ: Number of clip-wise queries to decode Q.
		fQ == cQ
		text_embed: b, Q, c
		"""
		if not self.training:
			frame_query = frame_query[[-1]]

		L, _, fQ, C = frame_query.shape     # num_layers, batch*frames, queries, channels

		frame_query = rearrange(frame_query, 'l (b t) q c -> (l b) t q c', l=L, b=B, t=T, q=fQ)
		frame_query = frame_query.permute(1, 2, 0, 3).contiguous()      # T, Q, BL, C
		frame_query = self.input_proj_dec(frame_query) # T, fQ, LB, C

		if self.window_size > 0:
			pad = int(ceil(T / self.window_size)) * self.window_size - T
			_T = pad + T
			frame_query = F.pad(frame_query, (0,0,0,0,0,0,0,pad))   # _T, fQ, LB, C
			enc_mask = frame_query.new_ones(L*B, _T).bool()         # LB, _T
			enc_mask[:, :T] = False
		else:
			enc_mask = None

		frame_query = self.encode_frame_query(frame_query, enc_mask)
		src = frame_query[:T].flatten(0,1)              # TfQ, LB, C
		time_pos = self.time_embed(T)                  # fT, C

		src = self.src_embed(src)   # TfQ, LB, C        do nothing, identical mapping
		# dec_pos = self.fq_pos.weight[None, :, None, :].repeat(T, 1, L*B, 1).flatten(0, 1) # TfQ, LB, C
		dec_pos = repeat(time_pos, 't x c -> (t q) (x l b) c', t=T, l=L, b=B, c=C, q=fQ)

		# num_queries, LB, C
		query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, L*B, 1)    # cQ, LB, C
		output = text_embed.repeat(1, L, 1)                                     # num_queries, LB, C
		# first -> cQ, 1, B, c, then -> cQ, L, B, c, finally -> cQ, LB, c

		decoder_outputs = []
		for i in range(self.num_layers):
			# attention: cross-attention first
			output = self.transformer_cross_attention_layers[i](
				output, src,
				memory_mask=None,
				memory_key_padding_mask=None,
				pos=dec_pos, query_pos=query_embed
			)

			output = self.transformer_self_attention_layers[i](
				output, tgt_mask=None,
				tgt_key_padding_mask=None,
				query_pos=query_embed
			)

			# FFN
			output = self.transformer_ffn_layers[i](
				output
			)

			# use layernorm before each module, do extra layernorm after | 
			if (self.training and self.aux_loss) or (i == self.num_layers - 1):
				dec_out = self.decoder_norm(output) # cQ, LB, C | num_queries, LB, C
				dec_out = dec_out.transpose(0, 1)   # LB, cQ, C | LB, num_queries, C, sequence-level features, each query -> L(spatial layers), B (batch), C (channels)
				decoder_outputs.append(dec_out.view(L, B, self.num_queries, C))

		decoder_outputs = torch.stack(decoder_outputs, dim=0)   # D, L, B, cQ, C

		"""
		frame_query: fT, fQ, B*L, C
		"""
		frame_query = frame_query[:T]
		fT = frame_query.shape[0]   # T of frame_query, AFTER temporal self-attention
		frame_query = rearrange(frame_query, 't q (b l) c -> t (q b l) c', t=fT, q=fQ, b=B, l=L)    # fT, fQ*B*L, C

		# prepare inputs for cross-attention                    
		time_pos = repeat(time_pos, 't x c -> t (x q b l) c', t=T, b=B, q=fQ, c=C, l=L) # fT, B*fQ*L, C
		time_mask = torch.ones(B*fQ*L, fT).bool().to(frame_query.device)                # B*fQ*L, fT
		time_mask[:, :T] = False

		text_pos = self.text_pos(text_features).permute(2, 0, 1).repeat(1, fQ*L, 1)       # [length, batch_size, c]  
		text_features, text_masks = text_features.decompose()
		text_features = text_features.repeat(fQ*L, 1, 1)
		text_masks = text_masks.repeat(fQ*L, 1)
		text_features = text_features.permute(1, 0, 2)

		# for each layer in L
		time_outputs = self.time_proj(frame_query)
		word_features = self.text_proj(text_features)
		time_outputs2 = self.time_cross_attn(tgt=time_outputs,
									memory=word_features,
									tgt_key_padding_mask=time_mask,
									memory_key_padding_mask=text_masks,
									pos=text_pos,
									query_pos=time_pos,
		)
		time_outputs = self.relevance_norm(time_outputs + self.dropout(time_outputs2))

		time_outputs = time_outputs[:T]		# fT, B*fQ*L, C
		time_outputs = rearrange(time_outputs, 't (b q l) c -> t q b l c', t=T, b=B, q=fQ, l=L).permute(0,1,3,2,4)

		return decoder_outputs, time_outputs

	@torch.jit.unused
	def _set_aux_loss(
		self, outputs_cls, outputs_mask_embed, outputs_cq_embed, outputs_fq_embed
	):
		return [{"pred_logits": a, "pred_mask_embed": b, "pred_cq_embed": c, "pred_fq_embed": outputs_fq_embed}
				for a, b, c in zip(outputs_cls[:-1], outputs_mask_embed[:-1], outputs_cq_embed[:-1])]

	def encode_frame_query(self, frame_query, attn_mask, rel=False):
		"""
		input shape (frame_query)   : T, fQ, LB, C
		output shape (frame_query)  : T, fQ, LB, C | attn_mask: L, T
		"""
		# Not using window-based attention if self.window_size == 0.
		if self.window_size == 0:
			return_shape = frame_query.shape        # T, fQ, LB, C
			frame_query = frame_query.flatten(0, 1) # TfQ, LB, C

			if not rel:
				for i in range(self.enc_layers):
					frame_query = self.enc_self_attn[i](frame_query)
					frame_query = self.enc_ffn[i](frame_query)
			else:
				frame_query = self.enc_self_attn_rel[i](frame_query)
				frame_query = self.enc_ffn_rel[i](frame_query)

			frame_query = frame_query.view(return_shape)
			return frame_query
		# Using window-based attention if self.window_size > 0.
		else:
			T, fQ, LB, C = frame_query.shape    # num_frames, num_queries, L, B, C
			W = self.window_size                # window size
			Nw = T // W                         # number of windows
			half_W = int(ceil(W / 2))           # 

			window_mask = attn_mask.view(LB*Nw, W)[..., None].repeat(1, 1, fQ).flatten(1)   # L, T -> 

			_attn_mask  = torch.roll(attn_mask, half_W, 1)
			_attn_mask  = _attn_mask.view(LB, Nw, W)[..., None].repeat(1, 1, 1, W)    # LB, Nw, W, W
			_attn_mask[:,  0] = _attn_mask[:,  0] | _attn_mask[:,  0].transpose(-2, -1)
			_attn_mask[:, -1] = _attn_mask[:, -1] | _attn_mask[:, -1].transpose(-2, -1)
			_attn_mask[:, 0, :half_W, half_W:] = True
			_attn_mask[:, 0, half_W:, :half_W] = True
			_attn_mask  = _attn_mask.view(LB*Nw, 1, W, 1, W, 1).repeat(1, self.num_heads, 1, fQ, 1, fQ).view(LB*Nw*self.num_heads, W*fQ, W*fQ)
			shift_window_mask = _attn_mask.float() * -1000

			for layer_idx in range(self.enc_layers):
				if self.training or layer_idx % 2 == 0:
					frame_query = self._window_attn(frame_query, window_mask, layer_idx, rel)
				else:
					frame_query = self._shift_window_attn(frame_query, shift_window_mask, layer_idx, rel)
			return frame_query

	def time_encode_frame_query(self, frame_query, attn_mask):
		"""
		input shape (frame_query)   : T, fQ, LB, C
		output shape (frame_query)  : T, fQ, LB, C
		"""

		# Not using window-based attention if self.window_size == 0.
		if self.window_size == 0:
			return_shape = frame_query.shape        # T, fQ, LB, C
			frame_query = frame_query.flatten(0, 1) # TfQ, LB, C

			for i in range(self.time_enc_layers):
				frame_query = self.time_enc_self_attn[i](frame_query)
				frame_query = self.time_enc_ffn[i](frame_query)

			frame_query = frame_query.view(return_shape)
			return frame_query
		# Using window-based attention if self.window_size > 0.
		else:
			T, fQ, LB, C = frame_query.shape
			W = self.window_size
			Nw = T // W
			half_W = int(ceil(W / 2))

			window_mask = attn_mask.view(LB*Nw, W)[..., None].repeat(1, 1, fQ).flatten(1)

			_attn_mask  = torch.roll(attn_mask, half_W, 1)
			_attn_mask  = _attn_mask.view(LB, Nw, W)[..., None].repeat(1, 1, 1, W)    # LB, Nw, W, W
			_attn_mask[:,  0] = _attn_mask[:,  0] | _attn_mask[:,  0].transpose(-2, -1)
			_attn_mask[:, -1] = _attn_mask[:, -1] | _attn_mask[:, -1].transpose(-2, -1)
			_attn_mask[:, 0, :half_W, half_W:] = True
			_attn_mask[:, 0, half_W:, :half_W] = True
			_attn_mask  = _attn_mask.view(LB*Nw, 1, W, 1, W, 1).repeat(1, self.num_heads, 1, fQ, 1, fQ).view(LB*Nw*self.num_heads, W*fQ, W*fQ)
			shift_window_mask = _attn_mask.float() * -1000

			for layer_idx in range(self.time_enc_layers):
				if self.training or layer_idx % 2 == 0:
					frame_query = self._time_window_attn(frame_query, window_mask, layer_idx)
				else:
					frame_query = self._time_shift_window_attn(frame_query, shift_window_mask, layer_idx)
			return frame_query

	def _time_window_attn(self, frame_query, attn_mask, layer_idx):
		T, fQ, LB, C = frame_query.shape
		# LBN, WTfQ = attn_mask.shape

		W = self.window_size
		Nw = T // W

		frame_query = frame_query.view(Nw, W, fQ, LB, C)
		frame_query = frame_query.permute(1,2,3,0,4).reshape(W*fQ, LB*Nw, C)

		frame_query = self.time_enc_self_attn[layer_idx](frame_query, tgt_key_padding_mask=attn_mask)
		frame_query = self.time_enc_ffn[layer_idx](frame_query)
		frame_query = frame_query.reshape(W, fQ, LB, Nw, C).permute(3,0,1,2,4).reshape(T, fQ, LB, C)

		return frame_query
	
	def _time_shift_window_attn(self, frame_query, attn_mask, layer_idx):
		T, fQ, LB, C = frame_query.shape
		# LBNH, WfQ, WfQ = attn_mask.shape

		W = self.window_size
		Nw = T // W
		half_W = int(ceil(W / 2))

		frame_query = torch.roll(frame_query, half_W, 0)
		frame_query = frame_query.view(Nw, W, fQ, LB, C)
		frame_query = frame_query.permute(1,2,3,0,4).reshape(W*fQ, LB*Nw, C)

		frame_query = self.time_enc_self_attn[layer_idx](frame_query, tgt_mask=attn_mask)
		frame_query = self.time_enc_ffn[layer_idx](frame_query)
		frame_query = frame_query.reshape(W, fQ, LB, Nw, C).permute(3,0,1,2,4).reshape(T, fQ, LB, C)

		frame_query = torch.roll(frame_query, -half_W, 0)

		return frame_query

	def _window_attn(self, frame_query, attn_mask, layer_idx, rel):
		T, fQ, LB, C = frame_query.shape
		# LBN, WTfQ = attn_mask.shape

		W = self.window_size
		Nw = T // W

		frame_query = frame_query.view(Nw, W, fQ, LB, C)
		frame_query = frame_query.permute(1,2,3,0,4).reshape(W*fQ, LB*Nw, C)

		if not rel:
			frame_query = self.enc_self_attn[layer_idx](frame_query, tgt_key_padding_mask=attn_mask)
			frame_query = self.enc_ffn[layer_idx](frame_query)
		else:
			frame_query = self.enc_self_attn_rel[layer_idx](frame_query, tgt_key_padding_mask=attn_mask)
			frame_query = self.enc_ffn_rel[layer_idx](frame_query)
		frame_query = frame_query.reshape(W, fQ, LB, Nw, C).permute(3,0,1,2,4).reshape(T, fQ, LB, C)

		return frame_query

	def _shift_window_attn(self, frame_query, attn_mask, layer_idx, rel):
		T, fQ, LB, C = frame_query.shape
		# LBNH, WfQ, WfQ = attn_mask.shape

		W = self.window_size
		Nw = T // W
		half_W = int(ceil(W / 2))

		frame_query = torch.roll(frame_query, half_W, 0)
		frame_query = frame_query.view(Nw, W, fQ, LB, C)
		frame_query = frame_query.permute(1,2,3,0,4).reshape(W*fQ, LB*Nw, C)

		if not rel:
			frame_query = self.enc_self_attn[layer_idx](frame_query, tgt_mask=attn_mask)
			frame_query = self.enc_ffn[layer_idx](frame_query)
		else:
			frame_query = self.enc_self_attn_rel[layer_idx](frame_query, tgt_mask=attn_mask)
			frame_query = self.enc_ffn_rel[layer_idx](frame_query)
		frame_query = frame_query.reshape(W, fQ, LB, Nw, C).permute(3,0,1,2,4).reshape(T, fQ, LB, C)

		frame_query = torch.roll(frame_query, -half_W, 0)

		return frame_query


class Language2Time(nn.Module):
	def __init__(self, d_model, nhead, dropout=0.0):
		super().__init__()
		self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
		self.layer_norm = nn.LayerNorm(d_model, eps=1e-12)
		self.dropout = nn.Dropout(dropout)

	def with_pos_embed(self, tensor, pos: Optional[Tensor]):
		return tensor if pos is None else tensor + pos

	def forward(self, tgt, memory,
				tgt_key_padding_mask: Optional[Tensor] = None,
				memory_key_padding_mask: Optional[Tensor] = None,
				pos: Optional[Tensor] = None,
				query_pos: Optional[Tensor] = None):
		tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
								   key=self.with_pos_embed(memory, pos),
								   value=memory, attn_mask=None,
								   key_padding_mask=memory_key_padding_mask)[0]
		tgt = self.layer_norm(tgt + self.dropout(tgt2))
		return tgt