#CUDA_VISIBLE_DEVICES=0 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorandpp_dcnv3_variant.py
#CUDA_VISIBLE_DEVICES=0 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorandpp_dcnv3.py
#CUDA_VISIBLE_DEVICES=0 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorandpp_dcnv3_wocn.py
#CUDA_VISIBLE_DEVICES=0 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand.py
#CUDA_VISIBLE_DEVICES=0 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_chnl64_shareG.py
#CUDA_VISIBLE_DEVICES=0 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_chnl64_repForward.py
#CUDA_VISIBLE_DEVICES=5 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand4lora_chnl64_convFormat_memef.py

#CUDA_VISIBLE_DEVICES=5 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_adaptformer_d128.py
#CUDA_VISIBLE_DEVICES=5 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_mona_wo1x1.py

#CUDA_VISIBLE_DEVICES=6 python tools/train.py peft_mmdet/retinanet/retinanet_dinov2-l-p14-w16_fpn_1x_voc_lora.py
CUDA_VISIBLE_DEVICES=6 python tools/train.py peft_mmdet/retinanet/retinanet_mae-b-p14-w16_fpn_1x_voc_lora.py