# -*- coding: utf-8 -*

from loguru import logger

import torch
import torch.nn.functional as F
from videoanalyst.model.common_opr.common_block import (conv_bn_relu,
                                                        xcorr_depthwise)
from videoanalyst.model.aia_transformer.transformer_impl.position_encoding import PositionEmbeddingSine
from videoanalyst.model.aia_transformer.transformer_impl.transformer import (TransformerEncoderLayer,
                                                                             TransformerEncoder)                                   
from videoanalyst.model.module_base import ModuleBase
from videoanalyst.model.task_model.taskmodel_base import (TRACK_TASKMODELS,
                                                          VOS_TASKMODELS)

torch.set_printoptions(precision=8)


@TRACK_TASKMODELS.register
@VOS_TASKMODELS.register
class STMTrack(ModuleBase):

    default_hyper_params = dict(pretrain_model_path="",
                                head_width=256,
                                conv_weight_std=0.01,
                                corr_fea_output=False,
                                amp=False)

    support_phases = ["train", "memorize", "track"]

    def __init__(self, backbone_m, backbone_q, neck_m, neck_q, head, loss=None):
        super(STMTrack, self).__init__()
        self.basemodel_m = backbone_m
        self.basemodel_q = backbone_q
        self.neck_m = neck_m
        self.neck_q = neck_q
        self.head = head
        self.loss = loss
        HIDDEN_DIM =256
        N_steps=HIDDEN_DIM // 2
        self.p_e=PositionEmbeddingSine(N_steps, normalize=True)
        MATCH_DIM=64
        self.i_e=PositionEmbeddingSine(MATCH_DIM // 2, normalize=True)

 
        encoder_layer = TransformerEncoderLayer(HIDDEN_DIM, 
                                                nhead=4, 
                                                dim_feedforward=1024,
                                                dropout=0.1, 
                                                activation='relu', 
                                                normalize_before=False,
                                                divide_norm=False, 
                                                use_AiA=True,
                                                match_dim=MATCH_DIM, 
                                                feat_size=400)
        encoder_norm = None
        self.encoder = TransformerEncoder(encoder_layer, 
                                         num_layers=2, 
                                         norm=encoder_norm)
    




        self._phase = "train"

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, p):
        assert p in self.support_phases
        self._phase = p

    def memorize(self, im_crop,im_mask, fg_bg_label_map):
        fm = self.basemodel_m(im_crop, fg_bg_label_map)
        fm = self.neck_m(fm)


        fm_mask = F.interpolate(im_mask, size=fm.shape[-2:]).to(torch.bool)
        b,t,h,w=fm_mask.size()
        fm_mask=fm_mask.view(b*t,h,w)


        fm_pos=self.p_e(fm,fm_mask)
        fm_inr=self.i_e(fm,fm_mask)



        b,c,w,h=fm.size()
        fm=(fm).view(b,c,-1).permute(2, 0, 1).contiguous()
        fm_mask=fm_mask.permute(1,2,0).view(-1,b).contiguous()
        # print("fm_pos : ",fm_pos.shape)
        # print("b,c,-1 : ",b,c,-1)
        fm_pos=(fm_pos).view(b,c,-1).contiguous().permute(2, 0, 1)

        # input()
        _,c1,_,_=fm_inr.size()
        fm_inr=fm_inr.view(b,c1,-1).permute(2, 0, 1).contiguous()

        fm=self.encoder(fm, src_key_padding_mask=fm_mask, pos=fm_pos, inr=fm_inr)

        fm=fm.permute(1,2,0).view(b,c,w,h)
        fm = fm.permute(1, 0, 2, 3).unsqueeze(0).contiguous()  # B, C, T, H, W



        return fm

    def train_forward(self, training_data):
        memory_img = training_data["im_m"]
        mask_m=training_data["mask_m"]
        query_img = training_data["im_q"]
        mask_q=training_data["mask_q"]


        #backbone feature
        assert len(memory_img.shape) == 5
        B, T, C, H, W = memory_img.shape

        memory_img = memory_img.view(-1, C, H, W)  # no memory copy
        target_fg_bg_label_map = training_data["fg_bg_label_map"].view(-1, 1, H, W)

        fm = self.basemodel_m(memory_img, target_fg_bg_label_map)
        fm = self.neck_m(fm)  # B * T, C, H, W

        fm_mask = F.interpolate(mask_m, size=fm.shape[-2:]).to(torch.bool)
        b,t,h,w=fm_mask.size()
        fm_mask=fm_mask.view(b*t,h,w)
        fm_pos=self.p_e(fm,fm_mask)
        fm_inr=self.i_e(fm,fm_mask)





        b,c,w,h=fm.size()
        fm=(fm).view(b,c,-1).permute(2, 0, 1).contiguous()
        fm_mask=fm_mask.permute(1,2,0).view(-1,b).contiguous()
        fm_pos=(fm_pos).view(b,c,-1).contiguous().permute(2, 0, 1)
        _,c1,_,_=fm_inr.size()
        fm_inr=fm_inr.view(b,c1,-1).permute(2, 0, 1).contiguous()
        fm=self.encoder(fm, src_key_padding_mask=fm_mask, pos=fm_pos, inr=fm_inr)
        fm=fm.permute(1,2,0).view(b,c,w,h)
        fm = fm.view(B, T, *fm.shape[-3:]).contiguous()  # B, T, C, H, W
        fm = fm.permute(0, 2, 1, 3, 4).contiguous()  # B, C, T, H, W





        fq = self.basemodel_q(query_img)
        fq = self.neck_q(fq)
        fq_mask = F.interpolate(mask_q, size=fq.shape[-2:]).to(torch.bool)
        b,t,h,w=fq_mask.size()
        fq_mask=fq_mask.view(b*t,h,w)
        fq_pos=self.p_e(fq,fq_mask)
        fq_inr=self.i_e(fq,fq_mask)


        b,c,w,h=fq.size()
        fq=(fq).view(b,c,-1).permute(2, 0, 1).contiguous()
        fq_mask=fq_mask.permute(1,2,0).view(-1,b).contiguous()


        fq_pos=(fq_pos).view(b,c,-1).contiguous().permute(2, 0, 1)
        _,c1,_,_=fq_inr.size()
        fq_inr=fq_inr.view(b,c1,-1).permute(2, 0, 1).contiguous()
        fq=self.encoder(fq, src_key_padding_mask=fq_mask, pos=fq_pos, inr=fq_inr)
        fq=fq.permute(1,2,0).view(b,c,w,h)
        fq_list={}

        fq_list["fq"]=fq
        fq_list["fq_mask"]=fq_mask
        fq_list["fq_pos"]=fq_pos
        fq_list["fq_inr"]=fq_inr

        fcos_cls_score_final, fcos_ctr_score_final, fcos_bbox_final, corr_fea = self.head(fm, fq_list)
        predict_data = dict(
            cls_pred=fcos_cls_score_final,
            ctr_pred=fcos_ctr_score_final,
            box_pred=fcos_bbox_final,
        )
        if self._hyper_params["corr_fea_output"]:
            predict_data["corr_fea"] = corr_fea
        return predict_data

    def forward(self, *args, phase=None):
        if phase is None:
            phase = self._phase
        # used during training
        if phase == 'train':
            # resolve training data
            if self._hyper_params["amp"]:
                with torch.cuda.amp.autocast():
                    return self.train_forward(args[0])
            else:
                return self.train_forward(args[0])

        elif phase == 'memorize':
            target_img,mask, fg_bg_label_map = args
            fm = self.memorize(target_img,mask, fg_bg_label_map)
            out_list = fm

        elif phase == 'track':
            assert len(args) == 3
            search_img,serach_mask, fm = args
            fq = self.basemodel_q(search_img)
            fq = self.neck_q(fq)  # B, C, H, W

            fq_mask = F.interpolate(serach_mask, size=fq.shape[-2:]).to(torch.bool)
            b,t,h,w=fq_mask.size()
            fq_mask=fq_mask.view(b*t,h,w)
            fq_pos=self.p_e(fq,fq_mask)
            fq_inr=self.i_e(fq,fq_mask)
            b,c,w,h=fq.size()
            fq=(fq).view(b,c,-1).permute(2, 0, 1).contiguous()
            fq_mask=fq_mask.permute(1,2,0).view(-1,b).contiguous()
            fq_pos=(fq_pos).view(b,c,-1).contiguous().permute(2, 0, 1)
            _,c1,_,_=fq_inr.size()
            fq_inr=fq_inr.view(b,c1,-1).permute(2, 0, 1).contiguous()
            fq=self.encoder(fq, src_key_padding_mask=fq_mask, pos=fq_pos, inr=fq_inr)
            fq=fq.permute(1,2,0).view(b,c,w,h)
            fq_list={}
            fq_list["fq"]=fq
            fq_list["fq_mask"]=fq_mask
            fq_list["fq_pos"]=fq_pos
            fq_list["fq_inr"]=fq_inr
            fcos_cls_score_final, fcos_ctr_score_final, fcos_bbox_final, corr_fea = self.head(
                fm, fq_list, search_img.size(-1))
            # apply sigmoid
            fcos_cls_prob_final = torch.sigmoid(fcos_cls_score_final)
            fcos_ctr_prob_final = torch.sigmoid(fcos_ctr_score_final)
            # apply centerness correction
            fcos_score_final = fcos_cls_prob_final * fcos_ctr_prob_final

            extra = dict()
            # output
            out_list = fcos_score_final, fcos_bbox_final, fcos_cls_prob_final, fcos_ctr_prob_final, extra
        else:
            raise ValueError("Phase non-implemented.")

        return out_list

    def update_params(self):
        self._make_convs()
        self._initialize_conv()
        super().update_params()

    def _make_convs(self):
        head_width = self._hyper_params['head_width']

        # feature adjustment
        self.r_z_k = conv_bn_relu(head_width, head_width, 1, 3, 0, has_relu=False)
        self.c_z_k = conv_bn_relu(head_width, head_width, 1, 3, 0, has_relu=False)
        self.r_x = conv_bn_relu(head_width, head_width, 1, 3, 0, has_relu=False)
        self.c_x = conv_bn_relu(head_width, head_width, 1, 3, 0, has_relu=False)

    def _initialize_conv(self, ):
        conv_weight_std = self._hyper_params['conv_weight_std']
        conv_list = [
            self.r_z_k.conv, self.c_z_k.conv, self.r_x.conv, self.c_x.conv
        ]
        for ith in range(len(conv_list)):
            conv = conv_list[ith]
            torch.nn.init.normal_(conv.weight,
                                  std=conv_weight_std)  # conv_weight_std=0.01

    def set_device(self, dev):
        if not isinstance(dev, torch.device):
            dev = torch.device(dev)
        self.to(dev)
        if self.loss is not None:
            for loss_name in self.loss:
                self.loss[loss_name].to(dev)
