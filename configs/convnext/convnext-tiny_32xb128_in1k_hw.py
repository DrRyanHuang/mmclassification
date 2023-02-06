_base_ = [
    '../_base_/models/convnext/convnext-tiny.py',
    # '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/datasets/imagenet_bs32_hw1_.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]

data = dict(samples_per_gpu=128)

optimizer = dict(lr=4e-3)

custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]

# --------------- New ADD ---------------

model = dict(
    head=dict(
        num_classes=5, # 模型只有5类
    ))

# Download from Github
load_from = 'checkpoints/convnext-tiny_3rdparty_32xb128-noema_in1k_20220222-2908964a.pth'

# checkpoint saving
checkpoint_config = dict(interval=10)
runner = dict(type='EpochBasedRunner', max_epochs=100)