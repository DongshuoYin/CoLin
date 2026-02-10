#CUDA_VISIBLE_DEVICES=1 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorandpp_dcnv3_me.py
#CUDA_VISIBLE_DEVICES=1 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorandpp_dcnv3_variant.py
#CUDA_VISIBLE_DEVICES=1 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorandpp_dw7.py
#CUDA_VISIBLE_DEVICES=1 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_chnl64.py
#CUDA_VISIBLE_DEVICES=1 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_chnl64_shareG_k32_b4.py
#CUDA_VISIBLE_DEVICES=1 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_chnl64_repForward_shareAB.py
#CUDA_VISIBLE_DEVICES=1 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_cr_chnl64_shareG.py
#CUDA_VISIBLE_DEVICES=1 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_cr_chnl64_shareAB_k48.py
#CUDA_VISIBLE_DEVICES=1 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_cr_chnl64_shareAB_k32_wConv.py
#CUDA_VISIBLE_DEVICES=1 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_cr_chnl64_shareAB_k32_wConv_wDp.py
#CUDA_VISIBLE_DEVICES=1 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_cr_chnl64_shareAB_k32_wDCNv4.py
#CUDA_VISIBLE_DEVICES=1 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_cr_chnl64_shareAB_k32_wConvMlp.py
#CUDA_VISIBLE_DEVICES=1 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_cr_chnl64_shareAB_k32_wConv_CN.py
#CUDA_VISIBLE_DEVICES=1 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_cr_chnl64_shareAB_k32_AdaLoRA_SVDInit.py --auto-resume
#CUDA_VISIBLE_DEVICES=1 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_cr_chnl64_shareAB_k32_AdaLoRA_SVDInit_Coef0p5.py --auto-resume
#CUDA_VISIBLE_DEVICES=1 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_cr_chnl64_shareAB_k32_AdaLoRA_SVDInit_Coef1p0.py --auto-resume
#CUDA_VISIBLE_DEVICES=1 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_cr_chnl64_shareAB_k32_AdaLoRA_SVDInit_Coef1p5.py --auto-resume

#CUDA_VISIBLE_DEVICES=1 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_cr_chnl64_shareAB_k32_AdaLoRA_SVDInit_Coef0p5_wMoE.py --auto-resume
#CUDA_VISIBLE_DEVICES=1 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_cr_chnl64_shareAB_k32_AdaLoRA_SVDInit_Coef0p5_wConvMoE.py --auto-resume

#CUDA_VISIBLE_DEVICES=1 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_cr_chnl64_shareAB_k32_AdaLoRA_SVDInit_Coef0p5_wConv_scaleConv_adjustLr.py --auto-resume
#CUDA_VISIBLE_DEVICES=1 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_cr_chnl64_shareAB_k32_AdaLoRA_SVDInit_Coef0p5_wConv_adjustLr.py --auto-resume

#CUDA_VISIBLE_DEVICES=1 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_cr_chnl64_shareAB_k32_AdaLoRA_SVDInit_Coef0p5_wConv_scaleConv_parallel_adjustLr.py --auto-resume
#CUDA_VISIBLE_DEVICES=1 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_cr_chnl64_shareAB_k32_AdaLoRA_SVDInit_Coef0p5_wConv_scaleConv_adjustLrMore.py --auto-resume

#CUDA_VISIBLE_DEVICES=1 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_cr_chnl64_shareAB_k32_AdaLoRA_SVDInit_Coef0p5_wConv_scaleConv_k7_adjustLr.py --auto-resume
#CUDA_VISIBLE_DEVICES=1 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_cr_chnl64_shareAB_k32_AdaLoRA_SVDInit_Coef0p5_wConv_scaleConv_k21_adjustLr.py --auto-resume

#CUDA_VISIBLE_DEVICES=1 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_mona_pw.py
CUDA_VISIBLE_DEVICES=1 python tools/train.py peft_mmdet/retinanet/retinanet_dinov2-l-p14-w16_fpn_1x_voc_mona_woWin.py
