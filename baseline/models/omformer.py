"""
OMFormer model class.
Modified from ReferFormer (https://github.com/wjn922/ReferFormer)
"""
import os
import math
import torch
import torch.nn.functional as F
from torch import nn
from transformers import RobertaModel, RobertaTokenizerFast
import copy
from einops import rearrange, repeat


# custom packages
from util.misc import (NestedTensor, nested_tensor_from_videos_list,
                       inverse_sigmoid)
from .position_encoding import PositionEmbeddingSine1D
from .backbone import build_backbone
from .deformable_transformer import build_deforamble_transformer
from .segmentation import CrossModalFPNDecoder, Language2VisionFusionModule, Vision2LanguageFusionModule
from .matcher import build_matcher
from .criterion import SetCriterion
from .postprocessors import build_postprocessors
from .temporal_decoder import TemporalMultimodalDecoder


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # this disables a huggingface tokenizer warning (printed every epoch)

class OMFormer(nn.Module):
    """ This is the OMFormer module that performs referring video object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels, 
                    num_frames, mask_dim, dim_feedforward,
                    controller_layers, dynamic_mask_channels, num_frames_global, temporal_window_size,
                    aux_loss=False, with_box_refine=False, two_stage=False, 
                    freeze_text_encoder=False, rel_coord=True, text_model='roberta-base'):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         OMFormer can detect in a video. For ytvos, we recommend 5 queries for each frame.
            num_frames:  number of clip frames
            mask_dim: dynamic conv inter layer channel number.
            dim_feedforward: vision-language fusion module ffn channel number.
            dynamic_mask_channels: the mask feature output channel number.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.span_embed = MLP(hidden_dim, hidden_dim, 2, 2, dropout=0.3)
        self.valid_embed = MLP(hidden_dim, hidden_dim, 1, 2, dropout=0.3)
        self.num_feature_levels = num_feature_levels
        self.num_frames_global = num_frames_global
        self.temporal_window_size = temporal_window_size
        
        # Build Transformer
        # NOTE: different deformable detr, the query_embed out channels is
        # hidden_dim instead of hidden_dim * 2
        # This is because, the input to the decoder is text embedding feature
        self.query_embed = nn.Embedding(num_queries, hidden_dim) 
        
        # follow deformable-detr, we use the last three stages of backbone
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides[-3:])
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[-3:][_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs): # downsample 2x
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[-3:][0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.num_frames = num_frames
        self.mask_dim = mask_dim
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        assert two_stage == False, "args.two_stage must be false!"

        # initialization
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            self.span_embed = _get_clones(self.span_embed, num_pred)
            self.valid_embed = _get_clones(self.valid_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

        # Build Text Encoder
        self.tokenizer = RobertaTokenizerFast.from_pretrained(text_model)
        self.text_encoder = RobertaModel.from_pretrained(text_model)

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)
        
        # resize the bert output channel to transformer d_model
        self.resizer = FeatureResizer(
            input_feat_size=768,
            output_feat_size=hidden_dim,
            dropout=0.1,
        )

        # vision-language feature fuser
        self.l2v_fusion_module = Language2VisionFusionModule(d_model=hidden_dim, nhead=8)
        self.v2l_fusion_module = Vision2LanguageFusionModule(d_model=hidden_dim, nhead=8)

        # temporal multimodal decoder
        self.temporal_decoder = TemporalMultimodalDecoder(in_channels=self.hidden_dim,
                                                            aux_loss=True,
                                                            hidden_dim=self.hidden_dim,
                                                            num_frame_queries=self.num_queries,
                                                            num_queries=self.num_queries,
                                                            nheads=8,
                                                            dim_feedforward=2048,
                                                            enc_layers=6,
                                                            dec_layers=3,
                                                            enc_window_size=6,
                                                            pre_norm=True,
                                                            enforce_input_project=True,)

        self.text_pos = PositionEmbeddingSine1D(hidden_dim, normalize=True)

        # Build FPN Decoder
        self.rel_coord = rel_coord
        feature_channels = [self.backbone.num_channels[0]] + 3 * [hidden_dim]
        self.pixel_decoder = CrossModalFPNDecoder(feature_channels=feature_channels, conv_dim=hidden_dim, 
                                                  mask_dim=mask_dim, dim_feedforward=dim_feedforward, norm="GN")

        # Build Dynamic Conv
        self.controller_layers = controller_layers 
        self.in_channels = mask_dim
        self.dynamic_mask_channels = dynamic_mask_channels
        self.mask_out_stride = 4
        self.mask_feat_stride = 4

        weight_nums, bias_nums = [], []
        for l in range(self.controller_layers):
            if l == 0:
                if self.rel_coord:
                    weight_nums.append((self.in_channels + 2) * self.dynamic_mask_channels)
                else:
                    weight_nums.append(self.in_channels * self.dynamic_mask_channels)
                bias_nums.append(self.dynamic_mask_channels)
            elif l == self.controller_layers - 1:
                weight_nums.append(self.dynamic_mask_channels * 1) # output layer c -> 1
                bias_nums.append(1)
            else:
                weight_nums.append(self.dynamic_mask_channels * self.dynamic_mask_channels)
                bias_nums.append(self.dynamic_mask_channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

        self.controller = MLP(hidden_dim, hidden_dim, self.num_gen_params, 3)
        for layer in self.controller.layers:
            nn.init.zeros_(layer.bias)
            nn.init.xavier_uniform_(layer.weight)   
        

    def forward(self, samples: NestedTensor, captions, targets):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensors: image sequences, of shape [num_frames x 3 x H x W]
               - samples.mask: a binary mask of shape [num_frames x H x W], containing 1 on padded pixels
               - captions: list[str]
               - targets:  list[dict]

            It returns a dict with the following elements:
               - "pred_masks": Shape = [batch_size x num_queries x out_h x out_w]

               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if 'span' in targets[0]:
            return self.spatial_temporal_forward(samples, captions, targets)

        # Vision Backbone
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list(samples) 

        # features (list[NestedTensor]): res2 -> res5, shape of tensors is [B*T, Ci, Hi, Wi]
        # pos (list[Tensor]): shape of [B*T, C, Hi, Wi]
        features, pos = self.backbone(samples) 

        b = len(captions)
        t = pos[0].shape[0] // b

        # Language Backbone
        text_features, text_sentence_features = self.forward_text(captions, device=pos[0].device)
       
        text_pos = self.text_pos(text_features).permute(2, 0, 1)    # [length, batch_size, c]
        text_word_features, text_word_masks = text_features.decompose() # [length, batch_size]
        text_word_features = text_word_features.permute(1, 0, 2)    # [length, batch_size, c]
        length = text_word_features.shape[0]
        
        text_word_features = repeat(text_word_features, 'l b c -> l (b t) c', l=length, b=b, t=t)  # [length, b t, c]
        text_word_masks = repeat(text_word_masks, 'b l -> (b t) l', l=length, b=b, t=t)  # [length, b t, c]
        text_pos = repeat(text_pos, 'l b c -> l (b t) c', l=length, b=b, t=t)

        # Vision-Language Feature Fusion
        srcs, masks, poses, text_word_features = self.vision_language_fusion_forward(b, t, samples, features, pos, text_word_features, text_word_masks, text_pos)
        text_word_features = text_word_features.permute(1, 2, 0)    # [batch_size, c, length], after permute
        text_sentence_features = text_word_features[:,:,0]
        text_features_fused = NestedTensor(text_word_features.permute(0, 2, 1), text_word_masks)  # batch_size, t, c

        # Transformer
        query_embeds = self.query_embed.weight  # [num_queries, c]
        spatial_text_embed = repeat(text_sentence_features, '(b t) c -> b t q c', t=t, q=self.num_queries)
        hs, memory, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, inter_samples = \
                                            self.transformer(srcs, spatial_text_embed, masks, poses, query_embeds)
        
        out = {}
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid() # cxcywh, range in [0,1]
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        # rearrange
        outputs_class = rearrange(outputs_class, 'l (b t) q k -> l b t q k', b=b, t=t)
        outputs_coord = rearrange(outputs_coord, 'l (b t) q n -> l b t q n', b=b, t=t)
        out['pred_logits'] = outputs_class[-1] # [batch_size, time, num_queries_per_frame, num_classes]
        out['pred_boxes'] = outputs_coord[-1]  # [batch_size, time, num_queries_per_frame, 4]

        # Segmentation
        mask_features = self.pixel_decoder(features, text_features_fused, pos, memory, nf=t) # [batch_size*time, c, out_h, out_w]
        mask_features = rearrange(mask_features, '(b t) c h w -> b t c h w', b=b, t=t)

        # dynamic conv
        outputs_seg_masks = []
        for lvl in range(hs.shape[0]):
            dynamic_mask_head_params = self.controller(hs[lvl])   # [batch_size*time, num_queries_per_frame, num_params]
            dynamic_mask_head_params = rearrange(dynamic_mask_head_params, '(b t) q n -> b (t q) n', b=b, t=t)
            lvl_references = inter_references[lvl, ..., :2]
            lvl_references = rearrange(lvl_references, '(b t) q n -> b (t q) n', b=b, t=t)
            outputs_seg_mask = self.dynamic_mask_with_coords(mask_features, dynamic_mask_head_params, lvl_references, targets)
            outputs_seg_mask = rearrange(outputs_seg_mask, 'b (t q) h w -> b t q h w', t=t)
            outputs_seg_masks.append(outputs_seg_mask)
        out['pred_masks'] = outputs_seg_masks[-1]  # [batch_size, time, num_queries_per_frame, out_h, out_w]

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_seg_masks)
        
        if not self.training:
            # for visualization
            inter_references = inter_references[-2, :, :, :2]  # [batch_size*time, num_queries_per_frame, 2]
            inter_references = rearrange(inter_references, '(b t) q n -> b t q n', b=b, t=t) 
            out['reference_points'] = inter_references  # the reference points of last layer input
        return out

    def spatial_temporal_forward(self, samples: NestedTensor, captions, targets):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensors: image sequences, of shape [num_frames_full x 3 x H x W]
               - samples.mask: a binary mask of shape [num_frames_full x H x W], containing 1 on padded pixels
               - captions: list[str]
               - spans: list[[start_f, end_f]]
               - targets:  list[dict]

            It returns a dict with the following elements:
               - "pred_masks": Shape = [batch_size x num_queries x out_h x out_w]

               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        # Vision Backbone
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list(samples) 

        # features (list[NestedTensor]): res2 -> res5, shape of tensors is [B*T, Ci, Hi, Wi]
        # pos (list[Tensor]): shape of [B*T, C, Hi, Wi]
        features, pos = self.backbone(samples) 

        b = len(captions)
        t = pos[0].shape[0] // b

        # Language Backbone
        text_features, text_sentence_features = self.forward_text(captions, device=pos[0].device)
       
        text_pos = self.text_pos(text_features).permute(2, 0, 1)    # [length, batch_size, c]
        text_word_features, text_word_masks = text_features.decompose() # [length, batch_size]
        text_word_features = text_word_features.permute(1, 0, 2)    # [length, batch_size, c]
        length = text_word_features.shape[0]
        
        # repeat for language-2-vision fusion
        text_word_features = repeat(text_word_features, 'l b c -> l (b t) c', l=length, b=b, t=t)  # [length, b t, c]
        text_word_masks = repeat(text_word_masks, 'b l -> (b t) l', l=length, b=b, t=t)  # [length, b t, c]
        text_pos = repeat(text_pos, 'l b c -> l (b t) c', l=length, b=b, t=t)

        # Vision-Language Feature Fusion
        srcs, masks, poses, text_word_features = self.vision_language_fusion_forward(b, t, samples, features, pos, text_word_features, text_word_masks, text_pos)
        text_word_features = text_word_features.permute(1, 2, 0)    # [batch_size, c, length], after permute
        text_sentence_features_fused = text_word_features[:,:,0]
        text_sentence_features_fused = rearrange(text_sentence_features_fused, '(b t) c -> b t c', b=b, t=t)

        text_features_fused = NestedTensor(text_word_features.permute(0, 2, 1), text_word_masks)  # batch_size, t, c

        # Transformer
        query_embeds = self.query_embed.weight  # [num_queries, c]
        spatial_text_embed = repeat(text_sentence_features_fused, 'b t c -> b t q c', b=b, t=t, q=self.num_queries)
        hs, memory, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, inter_samples = \
                                            self.transformer(srcs, spatial_text_embed, masks, poses, query_embeds)
        # hs: [l, batch_size*time, num_queries_per_frame, c]
        # memory: list[Tensor], shape of tensor is [batch_size*time, c, hi, wi]
        # init_reference: [batch_size*time, num_queries_per_frame, 2]
        # inter_references: [l, batch_size*time, num_queries_per_frame, 4]

        init_reference_spatial = rearrange(init_reference, '(b t) q c -> (b t) q c', b=b, t=t)

        # stop gradient between spatial and spatial-temporal operations
        temporal_text_embed = repeat(text_sentence_features, 'b c -> b q c', b=b, q=self.num_queries).permute(1, 0, 2)  # q, b, c
        ts, tsn = self.temporal_decoder(hs, temporal_text_embed, b, t, text_features)
        # tsn: T, q, l, b, c
        out = {}
        # prediction
        outputs_classes = []
        outputs_coords = []
        outputs_valids = []
        outputs_spans = []

        # temporal modelling results-guided prediction
        ts2 = ts.repeat(1, 1, t, 1, 1)
        for lvl in range(ts2[-1].shape[0]):
            tsn_lvl = tsn[:,:,lvl].permute(2, 0, 1, 3)  # t, q, b, c -> [b, t, q, c] on the lvl layer
            tsn_lvl = rearrange(tsn_lvl, 'b t q c -> (b t) q c', b=b, t=t, q=self.num_queries)  # b*t, q, c
            outputs_span = self.span_embed[lvl](tsn_lvl)
            outputs_valid = self.valid_embed[lvl](tsn_lvl)

            if lvl == 0:
                reference = init_reference_spatial
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](ts2[-1][lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])     # from spatial results
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid() # cxcywh, range in [0,1]
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_valids.append(outputs_valid)
            outputs_spans.append(outputs_span)      # keep before sigmoid
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_valid = torch.stack(outputs_valids)
        outputs_span = torch.stack(outputs_spans)
        # rearrange
        outputs_class = rearrange(outputs_class, 'l (b t) q k -> l b t q k', b=b, t=t)
        outputs_coord = rearrange(outputs_coord, 'l (b t) q n -> l b t q n', b=b, t=t)
        outputs_valid = rearrange(outputs_valid, 'l (b t) q n -> l b t q n', b=b, t=t)
        outputs_span = rearrange(outputs_span, 'l (b t) q n -> l b t q n', b=b, t=t)
        out['pred_logits'] = outputs_class[-1]  # [batch_size, time, num_queries_per_frame, num_classes]
        out['pred_boxes'] = outputs_coord[-1]   # [batch_size, time, num_queries_per_frame, 4]
        out['pred_valids'] = outputs_valid[-1]  # [batch_size, time, num_queries_per_frame, 4]
        out['pred_spans'] = outputs_span[-1]    # [batch_size, time, num_queries_per_frame, 2] start/end logits

        # Segmentation (on mask_id frames only)
        # features = rearrange(features, '(b t) c h w')
        trim_pos = []
        trim_features = []
        for f_l, p_l in zip(features, pos):
            feat = rearrange(f_l.tensors, '(b t) c h w -> b t c h w', b=b, t=t)
            msk = rearrange(f_l.mask, '(b t) h w -> b t h w', b=b, t=t)
            ps = rearrange(p_l, '(b t) c h w -> b t c h w', b=b, t=t)

            tmp_ft = []
            tmp_ps = []
            tmp_mk = []
            for ti, tgt_ in enumerate(targets):
                mask_id = tgt_['mask_id']
                tmp_ft.append(feat[ti, mask_id])
                tmp_mk.append(msk[ti, mask_id])
                tmp_ps.append(ps[ti, mask_id])

            trim_pos.append(torch.stack(tmp_ps, 0).flatten(0,1))
            trim_features.append(NestedTensor(torch.stack(tmp_ft, 0).flatten(0,1), torch.stack(tmp_mk, 0).flatten(0,1)))

        trim_mem = []
        for m_l in memory:
            mem = rearrange(m_l, '(b t) c h w -> b t c h w', b=b, t=t)
            tmp_mm = []
            for ti, tgt_ in enumerate(targets):
                mask_id = tgt_['mask_id']
                tmp_mm.append(mem[ti, mask_id])

            trim_mem.append(torch.stack(tmp_mm, 0).flatten(0,1))

        txtf = rearrange(text_features_fused.tensors, '(b t) l c -> b t l c', b=b, t=t)
        txtm = rearrange(text_features_fused.mask, '(b t) l -> b t l', b=b, t=t)

        tmp_txtf = []
        tmp_txtm = []
        for ti, tgt_ in enumerate(targets):
            mask_id = tgt_['mask_id']
            tmp_txtf.append(txtf[ti, mask_id])
            tmp_txtm.append(txtm[ti, mask_id])

        trim_txt = NestedTensor(torch.stack(tmp_txtf, 0).flatten(0,1), torch.stack(tmp_txtm, 0).flatten(0,1))
        t_short = len(targets[0]['mask_id'])
        mask_features = self.pixel_decoder(trim_features, trim_txt, trim_pos, trim_mem, nf=t_short) # [batch_size*time, c, out_h, out_w]
        mask_features = rearrange(mask_features, '(b t) c h w -> b t c h w', b=b, t=t_short)
        # end of mask feature generation

        # dynamic conv
        hs_spatial = hs.detach()
        hs_spatial = rearrange(hs_spatial, 'l (b t) q c -> l b t q c', l=hs.shape[0], b=b, t=t, q=self.num_queries)
        hss = []
        for bi, tmp_t in enumerate(targets):
            hss.append(hs_spatial[:,bi,tmp_t['mask_id']])
        hs_spatial = torch.stack(hss, 1)
        # hs_spatial = torch.stack([hs_spatial[:,:,tmp_t['mask_id']] for tmp_t in targets], 1)   # l b t_s q c
        hs_spatial = rearrange(hs_spatial, 'l b t q c -> l (b t) q c', l=hs.shape[0], b=b, t=t_short, q=self.num_queries)
        inter_references_spatial = inter_references.detach()
        inter_references_spatial = rearrange(inter_references_spatial, 'l (b t) q c -> l b t q c', b=b, t=t)
        
        irs = []
        for bi, tmp_t in enumerate(targets):
            irs.append(inter_references_spatial[:,bi,tmp_t['mask_id']])
        inter_references_spatial = torch.stack(irs, 1)
        
        inter_references_spatial = rearrange(inter_references_spatial, 'l b t q c -> l (b t) q c', l=hs.shape[0], b=b, t=t_short, q=self.num_queries)

        outputs_seg_masks = []
        for lvl in range(hs_spatial.shape[0]):  # hs, initial, L, b*t, q, c
            dynamic_mask_head_params = self.controller(hs_spatial[lvl])   # [batch_size*time, num_queries_per_frame, num_params]
            dynamic_mask_head_params = rearrange(dynamic_mask_head_params, '(b t) q n -> b (t q) n', b=b, t=t_short)
            lvl_references = inter_references_spatial[lvl, ..., :2]
            lvl_references = rearrange(lvl_references, '(b t) q n -> b (t q) n', b=b, t=t_short)
            outputs_seg_mask = self.dynamic_mask_with_coords(mask_features, dynamic_mask_head_params, lvl_references, targets)
            outputs_seg_mask = rearrange(outputs_seg_mask, 'b (t q) h w -> b t q h w', t=t_short)
            outputs_seg_masks.append(outputs_seg_mask)
        out['pred_masks'] = outputs_seg_masks[-1]  # [batch_size, time, num_queries_per_frame, out_h, out_w]

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss_mix(outputs_class, outputs_coord, outputs_span, outputs_seg_masks, outputs_valid)
        
        if not self.training:
            # for visualization
            inter_references = inter_references[-2, :, :, :2]  # [batch_size*time, num_queries_per_frame, 2]
            inter_references = rearrange(inter_references, '(b t) q n -> b t q n', b=b, t=t) 
            out['reference_points'] = inter_references  # the reference points of last layer input
        return out
    
    def spatial_inference(self, samples: NestedTensor, captions, targets):
        # Vision Backbone
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list(samples) 

        # features (list[NestedTensor]): res2 -> res5, shape of tensors is [B*T, Ci, Hi, Wi]
        # pos (list[Tensor]): shape of [B*T, C, Hi, Wi]
        features, pos = self.backbone(samples) 

        b = len(captions)
        assert b == 1, 'inference only supports b = 1'
        t = pos[0].shape[0] // b

        # Language Backbone
        text_features, text_sentence_features = self.forward_text(captions, device=pos[0].device)
       
        text_pos = self.text_pos(text_features).permute(2, 0, 1)    # [length, batch_size, c]
        text_word_features, text_word_masks = text_features.decompose() # [length, batch_size]
        text_word_features = text_word_features.permute(1, 0, 2)    # [length, batch_size, c]
        length = text_word_features.shape[0]
        
        text_word_features = repeat(text_word_features, 'l b c -> l (b t) c', l=length, b=b, t=t)  # [length, b t, c]
        text_word_masks = repeat(text_word_masks, 'b l -> (b t) l', l=length, b=b, t=t)  # [length, b t, c]
        text_pos = repeat(text_pos, 'l b c -> l (b t) c', l=length, b=b, t=t)

        # Vision-Language Feature Fusion
        srcs, masks, poses, text_word_features = self.vision_language_fusion_forward(b, t, samples, features, pos, text_word_features, text_word_masks, text_pos)
        text_word_features = text_word_features.permute(1, 2, 0)    # [batch_size, c, length], after permute

        text_features_ = NestedTensor(text_word_features.permute(0, 2, 1), text_word_masks)  # batch_size, t, c

        # Transformer
        query_embeds = self.query_embed.weight  # [num_queries, c]
        spatial_text_embed = repeat(text_sentence_features, 'b c -> b t q c', t=t, q=self.num_queries)
        hs, memory, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, inter_samples = \
                                            self.transformer(srcs, spatial_text_embed, masks, poses, query_embeds)
        # hs: [l, batch_size*time, num_queries_per_frame, c]
        # memory: list[Tensor], shape of tensor is [batch_size*time, c, hi, wi]
        # init_reference: [batch_size*time, num_queries_per_frame, 2]
        # inter_references: [l, batch_size*time, num_queries_per_frame, 4]

        # tsn: T, q, l, b, c
        out = {}

        # prediction
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid() # cxcywh, range in [0,1]
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        # rearrange
        outputs_class = rearrange(outputs_class, 'l (b t) q k -> l b t q k', b=b, t=t)
        outputs_coord = rearrange(outputs_coord, 'l (b t) q n -> l b t q n', b=b, t=t)
        out['pred_logits'] = outputs_class[-1] # [batch_size, time, num_queries_per_frame, num_classes]
        out['pred_boxes'] = outputs_coord[-1]  # [batch_size, time, num_queries_per_frame, 4]

        # Segmentation
        mask_features = self.pixel_decoder(features, text_features_, pos, memory, nf=t) # [batch_size*time, c, out_h, out_w]
        mask_features = rearrange(mask_features, '(b t) c h w -> b t c h w', b=b, t=t)

        # dynamic conv
        outputs_seg_masks = []
        for lvl in range(hs.shape[0]):
            dynamic_mask_head_params = self.controller(hs[lvl])   # [batch_size*time, num_queries_per_frame, num_params]
            dynamic_mask_head_params = rearrange(dynamic_mask_head_params, '(b t) q n -> b (t q) n', b=b, t=t)
            lvl_references = inter_references[lvl, ..., :2]
            lvl_references = rearrange(lvl_references, '(b t) q n -> b (t q) n', b=b, t=t)
            outputs_seg_mask = self.dynamic_mask_with_coords(mask_features, dynamic_mask_head_params, lvl_references, targets)
            outputs_seg_mask = rearrange(outputs_seg_mask, 'b (t q) h w -> b t q h w', t=t)
            outputs_seg_masks.append(outputs_seg_mask)
        out['pred_masks'] = outputs_seg_masks[-1]  # [batch_size, time, num_queries_per_frame, out_h, out_w]

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_seg_masks)
        
        if not self.training:
            # for visualization
            inter_references = inter_references[-2, :, :, :2]  # [batch_size*time, num_queries_per_frame, 2]
            inter_references = rearrange(inter_references, '(b t) q n -> b t q n', b=b, t=t) 
            out['reference_points'] = inter_references  # the reference points of last layer input

        # mask: 1, len_seq, 5, h, w
        # hs:   4, len_seq, 5, c
        return out['pred_masks'], hs, text_features, text_sentence_features

    def temporal_inference(self, masks, hs, text_features, text_sentence_features):
        """
        # masks:    1, num_frames, 5, h, w
        # hs:       4, num_frames, 5, c
        """
        b, t, n_q, h, w = masks.shape
        
        # split into sub samples
        num_sub_samples = t // self.num_frames_global
        sample_interval = 2
        t_list = list(range(t))
        slices = [t_list[i::num_sub_samples] for i in range(num_sub_samples) if i%sample_interval == 0]
        starts = []
        ends = []
        idxs = []
        overall_valid = torch.ones((t, 5)).to(hs.device) * -1.
        for s in slices:
            temporal_text_embed = repeat(text_sentence_features, 'b c -> b q c', b=b, q=self.num_queries).permute(1, 0, 2)  # q, b, c
            ts, tsn = self.temporal_decoder(hs[:,s], temporal_text_embed, b, len(s), text_features)

            # temporal modelling results-guided prediction
            ts2 = ts.repeat(1, 1, len(s), 1, 1)

            outputs_classes = []
            outputs_spans = []
            outputs_valids = []
            for lvl in range(ts2[-1].shape[0]):
                tsn_lvl = tsn[:,:,lvl].permute(2, 0, 1, 3)  # t, q, b, c -> [b, t, q, c] on the lvl layer
                tsn_lvl = rearrange(tsn_lvl, 'b t q c -> (b t) q c', b=b, t=len(s), q=self.num_queries)  # b*t, q, c
                outputs_span = self.span_embed[lvl](tsn_lvl)
                outputs_valid = self.valid_embed[lvl](tsn_lvl)
                outputs_class = self.class_embed[lvl](ts2[-1][lvl])
                outputs_classes.append(outputs_class)
                outputs_valids.append(outputs_valid)
                outputs_spans.append(outputs_span)      # keep before sigmoid

            outputs_class = torch.stack(outputs_classes)
            outputs_span = torch.stack(outputs_spans)
            outputs_valid = torch.stack(outputs_valids)

            outputs_class.squeeze(0).squeeze(-1)[0]        # num_frames (0), 5 (num_queries)

            idx = torch.argmax(outputs_class)
            t_s = torch.argmax(outputs_span[0, :, idx, 0])
            t_e = t_s + torch.argmax(outputs_span[0, t_s:, idx, 1])

            overall_valid[s] = outputs_valid[0,:,:,0]

            starts.append(s[t_s])
            ends.append(s[t_e])
            idxs.append(idx.item())

        starts.sort()
        ends.sort(reverse=True)

        t_s = sum(starts[:self.temporal_window_size])//self.temporal_window_size
        t_e = sum(ends[:self.temporal_window_size])//self.temporal_window_size
        idx = max(set(idxs), key=idxs.count)

        template_masks = torch.ones_like(masks[:,:,0]) * (-1e2)    # b, t, h, w
        template_masks[:, t_s:t_e] = masks[:, t_s:t_e, idx]
        # template_masks = masks[:, :, idx]       # for ablates
        # template_masks[:,overall_valid[:,idx]>0] = masks[:, overall_valid[:,idx]>0, idx]    # for ablates

        return template_masks

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_boxes": b, "pred_masks": c} 
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_seg_masks[:-1])]

    @torch.jit.unused
    def _set_aux_loss_mix(self, outputs_class, outputs_coord, outputs_span, outputs_seg_masks, outputs_valid):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_boxes": b, "pred_spans": c, "pred_masks": d, "pred_valids": e} 
                for a, b, c, d, e in zip(outputs_class[:-1], outputs_coord[:-1], outputs_span[:-1], outputs_seg_masks[:-1], outputs_valid[:-1])]

    def vision_language_fusion_forward(self, b, t, samples, v_feat, v_pos, l_feat, l_mask, l_pos):
        """
        Vision-Language Feature Fuser
        b: Batch
        t: Num of frames
        """
        srcs = []
        srcs_embedded = []
        masks = []
        poses = []

        # Follow Deformable-DETR, we use the last three stages outputs from backbone
        for l, (feat, pos_l) in enumerate(zip(v_feat[-3:], v_pos[-3:])): 
            src, mask = feat.decompose()            
            src_proj_l = self.input_proj[l](src)
            n, c, h, w = src_proj_l.shape

            # VISION to LANGUAGE fusion
            src_proj_l = rearrange(src_proj_l, '(b t) c h w -> (h w) (b t) c', b=b, t=t)
            srcs_embedded.append(src_proj_l.clone())    # for subsequent vision-to-language fusion
            src_proj_l_fused = self.l2v_fusion_module(tgt=src_proj_l,
                                             memory=l_feat,
                                             memory_key_padding_mask=l_mask,
                                             pos=l_pos,
                                             query_pos=None
            ) 
            src_proj_l_fused = rearrange(src_proj_l_fused, '(h w) (b t) c -> (b t) c h w', t=t, h=h, w=w)

            srcs.append(src_proj_l_fused)
            masks.append(mask)
            poses.append(pos_l)
            assert mask is not None
        
        if self.num_feature_levels > (len(v_feat) - 1):
            _len_srcs = len(v_feat) - 1 # fpn level
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](v_feat[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                n, c, h, w = src.shape

                # vision language early-fusion
                src = rearrange(src, '(b t) c h w -> (h w) (b t) c', b=b, t=t)
                srcs_embedded.append(src.clone())    # for subsequent vision-to-language fusion
                src_fused = self.l2v_fusion_module(tgt=src,
                                            memory=l_feat,
                                            memory_key_padding_mask=l_mask,
                                            pos=l_pos,
                                            query_pos=None
                )
                src_fused = rearrange(src_fused, '(h w) (b t) c -> (b t) c h w', t=t, h=h, w=w)

                srcs.append(src_fused)
                masks.append(mask)
                poses.append(pos_l)
        
        # vision to language fusion
        for (src_proj_l, mask_l) in zip(srcs_embedded, masks):

            # LANGUAGE to VISION fusion
            mask_l = rearrange(mask_l, '(b t) h w -> (b t) (h w)', b=b, t=t)    # [bt, hw]
 
            l_feat = self.v2l_fusion_module(tgt=l_feat,
                                            memory=src_proj_l,
                                            memory_key_padding_mask=mask_l,
                                            pos=None,
                                            query_pos=l_pos)                    #[length, batch_size, c]
        
        return srcs, masks, poses, l_feat

    def forward_text(self, captions, device):
        if isinstance(captions[0], str):
            tokenized = self.tokenizer.batch_encode_plus(captions, padding="longest", return_tensors="pt").to(device)
            encoded_text = self.text_encoder(**tokenized)
            # encoded_text.last_hidden_state: [batch_size, length, 768]
            # encoded_text.pooler_output: [batch_size, 768]
            text_attention_mask = tokenized.attention_mask.ne(1).bool()
            # text_attention_mask: [batch_size, length]

            text_features = encoded_text.last_hidden_state 
            text_features = self.resizer(text_features)    
            text_masks = text_attention_mask              
            text_features = NestedTensor(text_features, text_masks) # NestedTensor

            text_sentence_features = encoded_text.pooler_output  
            text_sentence_features = self.resizer(text_sentence_features)  
        else:
            raise ValueError("Please mask sure the caption is a list of string")
        return text_features, text_sentence_features
        # return text_features, None

    def dynamic_mask_with_coords(self, mask_features, mask_head_params, reference_points, targets):
        """
        Add the relative coordinates to the mask_features channel dimension,
        and perform dynamic mask conv.

        Args:
            mask_features: [batch_size, time, c, h, w]
            mask_head_params: [batch_size, time * num_queries_per_frame, num_params]
            reference_points: [batch_size, time * num_queries_per_frame, 2], cxcy
            targets (list[dict]): length is batch size
                we need the key 'size' for computing location.
        Return:
            outputs_seg_mask: [batch_size, time * num_queries_per_frame, h, w]
        """
        device = mask_features.device
        b, t, c, h, w = mask_features.shape
        # this is the total query number in all frames
        _, num_queries = reference_points.shape[:2]  
        q = num_queries // t  # num_queries_per_frame

        # prepare reference points in image size (the size is input size to the model)
        new_reference_points = [] 
        for i in range(b):
            img_h, img_w = targets[i]['size']
            scale_f = torch.stack([img_w, img_h], dim=0) 
            tmp_reference_points = reference_points[i] * scale_f[None, :] 
            new_reference_points.append(tmp_reference_points)
        new_reference_points = torch.stack(new_reference_points, dim=0) 
        # [batch_size, time * num_queries_per_frame, 2], in image size
        reference_points = new_reference_points  

        # prepare the mask features
        if self.rel_coord:
            reference_points = rearrange(reference_points, 'b (t q) n -> b t q n', t=t, q=q) 
            locations = compute_locations(h, w, device=device, stride=self.mask_feat_stride) 
            relative_coords = reference_points.reshape(b, t, q, 1, 1, 2) - \
                                    locations.reshape(1, 1, 1, h, w, 2) # [batch_size, time, num_queries_per_frame, h, w, 2]
            relative_coords = relative_coords.permute(0, 1, 2, 5, 3, 4) # [batch_size, time, num_queries_per_frame, 2, h, w]

            # concat features
            mask_features = repeat(mask_features, 'b t c h w -> b t q c h w', q=q) # [batch_size, time, num_queries_per_frame, c, h, w]
            mask_features = torch.cat([mask_features, relative_coords], dim=3)
        else:
            mask_features = repeat(mask_features, 'b t c h w -> b t q c h w', q=q) # [batch_size, time, num_queries_per_frame, c, h, w]
        mask_features = mask_features.reshape(1, -1, h, w) 

        # parse dynamic params
        mask_head_params = mask_head_params.flatten(0, 1) 
        weights, biases = parse_dynamic_params(
            mask_head_params, self.dynamic_mask_channels,
            self.weight_nums, self.bias_nums
        )

        # dynamic mask conv
        mask_logits = self.mask_heads_forward(mask_features, weights, biases, mask_head_params.shape[0]) 
        mask_logits = mask_logits.reshape(-1, 1, h, w)

        # upsample predicted masks
        assert self.mask_feat_stride >= self.mask_out_stride
        assert self.mask_feat_stride % self.mask_out_stride == 0

        mask_logits = aligned_bilinear(mask_logits, int(self.mask_feat_stride / self.mask_out_stride))
        mask_logits = mask_logits.reshape(b, num_queries, mask_logits.shape[-2], mask_logits.shape[-1])

        return mask_logits  # [batch_size, time * num_queries_per_frame, h, w]

    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x


def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts)

    return weight_splits, bias_splits

def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4 
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )

    return tensor[:, :, :oh - 1, :ow - 1]


def compute_locations(h, w, device, stride=1):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device)

    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            if self.dropout and i < self.num_layers:
                x = self.dropout(x)
        return x

class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


def build(args):
    if args.binary:
        num_classes = 1
    else:
        if args.dataset_file == 'ytvos':
            num_classes = 65 
        elif args.dataset_file == 'davis':
            num_classes = 78
        elif args.dataset_file == 'a2d' or args.dataset_file == 'jhmdb':
            num_classes = 1
        else: 
            num_classes = 91 # for coco
    device = torch.device(args.device)

    # backbone
    if 'video_swin' in args.backbone:
        from .video_swin_transformer import build_video_swin_backbone
        backbone = build_video_swin_backbone(args)
    elif 'swin' in args.backbone:
        from .swin_transformer import build_swin_backbone
        backbone = build_swin_backbone(args) 
    else:
        backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)

    model = OMFormer(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        num_frames=args.num_frames,
        mask_dim=args.mask_dim,
        dim_feedforward=args.dim_feedforward,
        controller_layers=args.controller_layers,
        dynamic_mask_channels=args.dynamic_mask_channels,
        num_frames_global=args.num_frames_global,
        temporal_window_size=args.temporal_window_size,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        freeze_text_encoder=args.freeze_text_encoder,
        rel_coord=args.rel_coord,
        text_model=args.text_model, 
    )
    matcher = build_matcher(args)
    weight_dict = {}
    weight_dict['loss_ce'] = args.cls_loss_coef
    weight_dict['loss_bbox'] = args.bbox_loss_coef
    weight_dict['loss_giou'] = args.giou_loss_coef
    weight_dict['loss_span'] = args.span_loss_coef
    weight_dict['loss_valid'] = args.relevance_loss_coef
    if args.masks: # always true
        weight_dict['loss_mask'] = args.mask_loss_coef
        weight_dict['loss_dice'] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes']
    if args.masks:
        losses += ['masks']
    criterion = SetCriterion(
            num_classes, 
            matcher=matcher,
            weight_dict=weight_dict, 
            irrelevance_coef=args.irrelevance_coef, 
            losses=losses,
            focal_alpha=args.focal_alpha)
    criterion.to(device)

    # postprocessors, this is used for coco pretrain but not for rvos
    postprocessors = build_postprocessors(args, args.dataset_file)
    return model, criterion, postprocessors