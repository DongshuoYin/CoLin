#CUDA_VISIBLE_DEVICES=2 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorandpp_dcnv3_fme.py
#CUDA_VISIBLE_DEVICES=2 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorandpp_dcnv3_wocn.py
#CUDA_VISIBLE_DEVICES=2 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorandpp_dw7_woMlp.py
#CUDA_VISIBLE_DEVICES=2 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_chnl64_k32_b1.py
#CUDA_VISIBLE_DEVICES=2 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_chnl64_k32_b2.py
#CUDA_VISIBLE_DEVICES=2 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_chnl64_repForward_SVD.py
#CUDA_VISIBLE_DEVICES=2 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_cr_chnl64_shareAB.py
#CUDA_VISIBLE_DEVICES=2 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_cr_chnl64_shareAB_k32_b2.py
#CUDA_VISIBLE_DEVICES=2 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_cr_chnl64_shareAB_k32_wConv_outShortcut.py
#CUDA_VISIBLE_DEVICES=2 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_cr_chnl80_shareAB_k32_wConv.py
#CUDA_VISIBLE_DEVICES=2 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_cr_chnl64_shareAB_k32_wSwin.py

CUDA_VISIBLE_DEVICES=2 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_mona_inception.py