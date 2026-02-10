#CUDA_VISIBLE_DEVICES=0 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorandpp_dcnv3_variant.py
#CUDA_VISIBLE_DEVICES=0 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorandpp_dcnv3.py
#CUDA_VISIBLE_DEVICES=0 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorandpp_dcnv3_wocn.py
#CUDA_VISIBLE_DEVICES=0 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand.py
#CUDA_VISIBLE_DEVICES=0 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_chnl64_shareG.py
#CUDA_VISIBLE_DEVICES=0 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_chnl64_repForward.py
#CUDA_VISIBLE_DEVICES=4 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand4lora_chnl64_convFormat.py

#CUDA_VISIBLE_DEVICES=4 bash tools/dist_train.sh peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_cr_chnl64_shareAB_k32_AdaLoRA_SVDInit_Coef0p5_wMoE.py 1

#CUDA_VISIBLE_DEVICES=4 python tools/train.py peft_mmdet/retinanet/retinanet_dinov2-l-p14-w16_fpn_1x_voc_mona.py
CUDA_VISIBLE_DEVICES=4 python tools/train.py peft_mmdet/retinanet/retinanet_mae-b-p14-w16_fpn_1x_voc_mona.py