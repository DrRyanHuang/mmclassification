# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import List, Optional, Union
from pycocotools.coco import COCO as _COCO

from mmcls.registry import DATASETS
from .base_dataset import expanduser
from .categories import COCO_CATEGORITES
from .multi_label import MultiLabelDataset


@DATASETS.register_module()
class COCO(MultiLabelDataset):
    """`COCO2017 <https://cocodataset.org/#download>`_ Dataset.

    After decompression, the dataset directory structure is as follows:

    COCO dataset directory: ::

        COCO (data_root)/
        ├── train2017 (data_prefix['img_path'])
        │   ├── xxx.jpg
        │   ├── xxy.jpg
        │   └── ...
        ├── val2017 (data_prefix['img_path'])
        │   ├── xxx.jpg
        │   ├── xxy.jpg
        │   └── ...
        └── annotations (directory contains COCO annotation JSON file)

    Extra iscrowd label is in COCO annotations, we will use
    `gt_label_crowd` to record the crowd labels in each sample
    and corresponding evaluation should take care of this field
    to calculate metrics. Usually, crowd labels are reckoned as
    negative in defaults.

    Args:
        data_prefix (str): the prefix of data path for COCO dataset.
        ann_file (str): coco annotation file path.
        test_mode (bool): ``test_mode=True`` means in test phase.
            It determines to use the training set or test set.
        metainfo (dict, optional): Meta information for dataset, such as
            categories information. Defaults to None.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """  # noqa: E501

    METAINFO = {'classes': COCO_CATEGORITES}

    def __init__(self,
                 data_prefix: str,
                 ann_file: str,
                 test_mode: bool = False,
                 metainfo: Optional[dict] = None,
                 **kwargs):
        if isinstance(data_prefix, str):
            data_prefix = dict(img_path=expanduser(data_prefix))
            
        assert 'val2017' in data_prefix or 'train2017' in data_prefix
        
        data_root = os.path.dirname(data_prefix)
        self.data_prefix = data_prefix
        self.ann_file = ann_file

        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            metainfo=metainfo,
            data_prefix=data_prefix,
            test_mode=test_mode,
            **kwargs)


    def _get_labels_from_coco(self, img_id):
        """Get gt_labels and labels_crowd from COCO object."""
        
        info = self.coco.loadImgs([img_id])[0]
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        labels = [self.cat2label[ann['category_id']] for ann in ann_info]
        labels_crowd = [
                self.cat2label[ann['category_id']] for ann in ann_info
                if ann['iscrowd']
            ]
        
        labels, labels_crowd = set(labels), set(labels_crowd)
        img_path = info['file_name']

        return list(labels), list(labels_crowd), img_path


    def load_data_list(self):
        """Load images and ground truth labels."""
        data_list = []
        
        self.coco = _COCO(self.ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.getCatIds(catNms=self.METAINFO['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.getImgIds()

        for img_id in self.img_ids:
            
            labels, labels_crowd, img_path = self._get_labels_from_coco(img_id)
            
            img_path = os.path.join(self.data_prefix['img_path'], img_path)

            info = dict(
                img_path=img_path,
                gt_label=labels,
                gt_label_crowd=labels_crowd)
            data_list.append(info)

        return data_list

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [
            f'Prefix of images: \t{self.data_prefix["img_path"]}',
            f'Path of annotation file: \t{self.ann_file}',
        ]

        return body
