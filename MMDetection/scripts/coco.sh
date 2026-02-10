#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash tools/dist_train.sh peft_mmdet/cascade_rcnn/cascade_mask_rcnn_swin-b-p4-w7_official_0p5x_coco.py 8
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash tools/dist_train.sh peft_mmdet/cascade_rcnn/cascade_mask_rcnn_swinMona-b-p4-w7_official_0p5x_coco.py 8

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash tools/dist_train.sh peft_mmdet/cascade_rcnn/cascade_mask_rcnn_swinLoRandpp-b-p4-w7_official_3x_coco.py 8
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash tools/dist_train.sh peft_mmdet/cascade_rcnn/cascade_mask_rcnn_swinLoRandppD64K64B4-b-p4-w7_official_3x_coco.py 8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash tools/dist_train.sh peft_mmdet/cascade_rcnn/cascade_mask_rcnn_swinLoRandppwoConv-b-p4-w7_official_3x_coco.py 8
