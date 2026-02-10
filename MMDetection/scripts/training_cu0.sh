#CUDA_VISIBLE_DEVICES=0 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorandpp_dcnv3_variant.py
#CUDA_VISIBLE_DEVICES=0 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorandpp_dcnv3.py
#CUDA_VISIBLE_DEVICES=0 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorandpp_dcnv3_wocn.py
#CUDA_VISIBLE_DEVICES=0 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand.py
#CUDA_VISIBLE_DEVICES=0 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_chnl64_shareG.py
#CUDA_VISIBLE_DEVICES=0 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_chnl64_repForward.py
#CUDA_VISIBLE_DEVICES=0 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_cr_chnl64.py
#CUDA_VISIBLE_DEVICES=0 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_cr_chnl64_shareAB_k16.py
#CUDA_VISIBLE_DEVICES=0 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_cr_chnl64_shareAB_k32_outShortcut.py
#CUDA_VISIBLE_DEVICES=0 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_cr_chnl64_shareAB_k32_wDp.py
#CUDA_VISIBLE_DEVICES=0 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_cr_chnl64_shareAB_k32_wDCNv3.py
#CUDA_VISIBLE_DEVICES=0 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_cr_chnl64_shareAB_k32_AdaLoRA.py --auto-resume

#CUDA_VISIBLE_DEVICES=0 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_cr_chnl64_shareAB_k32_AdaLoRA_SVDInit_Coef0p5_wConv.py

CUDA_VISIBLE_DEVICES=0 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_adaptformer_MSAMlp.py
#CUDA_VISIBLE_DEVICES=0 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_dmlora.py --auto-resume
#CUDA_VISIBLE_DEVICES=0 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_adaptformer_dualPath.py
