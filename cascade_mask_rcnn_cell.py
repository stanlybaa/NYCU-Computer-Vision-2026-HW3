_base_ = "mmdet::cascade_rcnn/cascade-mask-rcnn_r50_fpn_1x_coco.py"

dataset_type = "CocoDataset"
data_root = "coco_cell/"

classes = ("class1", "class2", "class3", "class4")

metainfo = dict(
    classes=classes,
    palette=[
        (220, 20, 60),
        (119, 11, 32),
        (0, 0, 142),
        (0, 60, 100),
    ],
)

num_classes = 4
backend_args = None


model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type="Shared2FCBBoxHead",
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2],
                ),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type="CrossEntropyLoss",
                    use_sigmoid=False,
                    loss_weight=1.0,
                ),
                loss_bbox=dict(
                    type="SmoothL1Loss",
                    beta=1.0,
                    loss_weight=1.0,
                ),
            ),
            dict(
                type="Shared2FCBBoxHead",
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1],
                ),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type="CrossEntropyLoss",
                    use_sigmoid=False,
                    loss_weight=1.0,
                ),
                loss_bbox=dict(
                    type="SmoothL1Loss",
                    beta=1.0,
                    loss_weight=1.0,
                ),
            ),
            dict(
                type="Shared2FCBBoxHead",
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067],
                ),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type="CrossEntropyLoss",
                    use_sigmoid=False,
                    loss_weight=1.0,
                ),
                loss_bbox=dict(
                    type="SmoothL1Loss",
                    beta=1.0,
                    loss_weight=1.0,
                ),
            ),
        ],
        mask_head=dict(
            type="FCNMaskHead",
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=num_classes,
            loss_mask=dict(
                type="CrossEntropyLoss",
                use_mask=True,
                loss_weight=1.0,
            ),
        ),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type="nms", iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type="nms", iou_threshold=0.5),
            max_per_img=300,
            mask_thr_binary=0.5,
        ),
    ),
)


train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="Resize",
        scale=(1024, 1024),
        keep_ratio=True,
    ),
    dict(
        type="RandomFlip",
        prob=0.5,
        direction="horizontal",
    ),
    dict(
        type="RandomFlip",
        prob=0.5,
        direction="vertical",
    ),
    dict(type="PackDetInputs"),
]


test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(
        type="Resize",
        scale=(1024, 1024),
        keep_ratio=True,
    ),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="PackDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
        ),
    ),
]


train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations/train.json",
        data_prefix=dict(img="images/train/"),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
        pipeline=train_pipeline,
        backend_args=backend_args,
    ),
)


val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations/val.json",
        data_prefix=dict(img="images/val/"),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)


test_dataloader = val_dataloader


val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "annotations/val.json",
    metric=["bbox", "segm"],
    classwise=True,
    backend_args=backend_args,
)


test_evaluator = val_evaluator


train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=15,
    val_interval=1,
)


val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")


optim_wrapper = dict(
    _delete_=True,
    type="AmpOptimWrapper",
    optimizer=dict(
        type="AdamW",
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    ),
    clip_grad=dict(
        max_norm=5.0,
        norm_type=2,
    ),
)


param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=500,
    ),
    dict(
        type="CosineAnnealingLR",
        eta_min=1e-6,
        begin=0,
        end=15,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]


default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=5,
        save_best="coco/segm_mAP_50",
        rule="greater",
    ),
    logger=dict(
        type="LoggerHook",
        interval=20,
    ),
)


log_processor = dict(
    type="LogProcessor",
    window_size=20,
    by_epoch=True,
)


load_from = (
    "https://download.openmmlab.com/mmdetection/v2.0/"
    "cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco/"
    "cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth"
)


work_dir = "work_dirs/cascade_cell"
