_base_ = [
    "../_base_/models/retinanet_r50_fpn.py",
    "../_base_/datasets/voc0712.py",
    "../_base_/default_runtime.py",
]

pretrained = "/root/shared-nvme/XXX/swin_large_patch4_window7_224_22k.pth"

model = dict(
    backbone=dict(
        _delete_=True,
        type="SwinCoLin",
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type="Pretrained", checkpoint=pretrained),
    ),
    neck=dict(in_channels=[384, 768, 1536], start_level=0, num_outs=5),
    bbox_head=dict(num_classes=20),
)

# dataset settings.
# data = dict(
#     samples_per_gpu=2,
# )

data = dict(
    samples_per_gpu=2,
)

# optimizer settings.
optimizer = dict(
    # _delete_=True,
    type="AdamW",
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    
    paramwise_cfg=dict(
        bypass_duplicate=True,
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)

# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(policy='step', step=[3])
# runtime settings
runner = dict(
    type='EpochBasedRunner', max_epochs=4)  # actual epoch = 4 * 3 = 12

# fp16 settings
optimizer_config = dict(
    # _delete_=True,
    loss_scale="dynamic",
    grad_clip=None,
)

# fp16 placeholder
fp16 = dict()

# learning policy.
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric="mAP")

# runtime settings.
# custom_imports = dict(
#     imports=["parasite.core.hook.parasite_gpu_memory_profiler_hook"],
#     allow_failed_imports=False,
# )

custom_hooks = [
    dict(type="NumClassCheckHook"),
    # dict(type="ParasiteMemoryProfilerHook", interval=50),
]
