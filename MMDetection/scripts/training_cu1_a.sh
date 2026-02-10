#CUDA_VISIBLE_DEVICES=1 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorandpp_dcnv3_me.py
#CUDA_VISIBLE_DEVICES=1 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorandpp_dcnv3_variant.py
#CUDA_VISIBLE_DEVICES=1 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorandpp_dw7.py
#CUDA_VISIBLE_DEVICES=1 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_chnl64.py
#CUDA_VISIBLE_DEVICES=1 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_chnl64_shareG_k32_b4.py
#CUDA_VISIBLE_DEVICES=1 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_chnl64_repForward_shareAB.py
#CUDA_VISIBLE_DEVICES=1 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_cr_chnl64_shareG_convFormat.py

#CUDA_VISIBLE_DEVICES=1 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_cr_chnl64_shareAB_k32_AdaLoRA_SVDInit_Coef0p5_wConv_scaleConv.py

#CUDA_VISIBLE_DEVICES=1 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_cr_chnl64_shareAB_k32_AdaLoRA_SVDInit_Coef0p5_wConv_scaleConv_adjustLrwhole.py --auto-resume

CUDA_VISIBLE_DEVICES=1 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_cr_chnl64_shareAB_k32_AdaLoRA_SVDInit_Coef0p5_wConvMoE_adjustLrMore.py --auto-resume