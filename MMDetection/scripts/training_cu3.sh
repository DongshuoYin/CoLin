#CUDA_VISIBLE_DEVICES=3 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorandpp_dcnv3_variant_memef.py
#CUDA_VISIBLE_DEVICES=3 bash tools/dist_train.sh peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorandpp_dcnv3_variant_memef.py 1
#CUDA_VISIBLE_DEVICES=3 bash tools/dist_train.sh peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorandpp_dw7_woShortcut.py 1
#CUDA_VISIBLE_DEVICES=3 bash tools/dist_train.sh peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_chnl64_k32_b4.py 1
#CUDA_VISIBLE_DEVICES=3 bash tools/dist_train.sh peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_chnl64_k16_b2.py 1
#CUDA_VISIBLE_DEVICES=3 bash tools/dist_train.sh peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_chnl64_frepForward.py 1
#CUDA_VISIBLE_DEVICES=3 bash tools/dist_train.sh peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_cr_chnl64_shareAB_k32.py 1
#CUDA_VISIBLE_DEVICES=3 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_cr_chnl64_shareAB_k16_b8.py
#CUDA_VISIBLE_DEVICES=3 python tools/train.py peft_mmdet/retinanet/retinanet_swin-l-p4-w7_fpn_1x_voc_lorand_cr_chnl64_shareAB_k32_wConv_k7.py

#CUDA_VISIBLE_DEVICES=3 python tools/train.py peft_mmdet/retinanet/retinanet_dinov2-l-p14-w16_fpn_1x_voc.py
#CUDA_VISIBLE_DEVICES=3 python tools/train.py peft_mmdet/retinanet/retinanet_mae-b-p14-w16_fpn_1x_voc.py
