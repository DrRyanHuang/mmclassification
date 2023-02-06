_base_ = ['../_base_/datasets/coco_bs64.py', '../_base_/default_runtime.py']

# use different head for multilabel task
model = dict(
    type='ImageClassifier',
    backbone=dict(type='VGG', depth=16, num_classes=80),
    neck=None,
    head=dict(
        type='MultiLabelClsHead',
        loss=dict(type='AsymmetricLoss', use_sigmoid=True, loss_weight=1.0)))

# load model pretrained on imagenet

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.001/4,
    momentum=0.9,
    weight_decay=0,
    paramwise_cfg=dict(custom_keys={'.backbone.classifier': dict(lr_mult=10)}))
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy='step', step=20, gamma=0.1)
runner = dict(type='EpochBasedRunner', max_epochs=200)

# checkpoint saving
checkpoint_config = dict(interval=5)
resume_from = "work_dirs/vgg16_8xb64_coco/latest.pth"