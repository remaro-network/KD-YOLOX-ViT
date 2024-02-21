#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # ---------------- YOLOX with ViT Layer ---------------- #
        
        self.vit = False
        
        # ---------------- Knowledge Distillation config ---------------- #
        
        #KD set to True activate add the KD loss to the ground truth loss
        self.KD = True
        
        #KD_Online set to False recquires the teacher FPN logits saved to the "folder_KD_directory" folder
        #Then the student training will use the teacher FPN logits
        #Otherwise, if KD_Online set to True the student use the online data augmentation and does not recquire saved teacher FPN logits
        self.KD_online = True
        
        #KD_Teacher_Inference set to True save the FPN logits before using offline KD
        
        #folder_KD_directory is the folder where the teacher FPN logits are saved
        self.folder_KD_directory = "KD-FPN-Images/"
        
        if self.KD and not self.KD_online:
            # ---------------- dataloader config ---------------- #

            # To disable multiscale training, set the value to 0.
            self.multiscale_range = 0

            # --------------- transform config ----------------- #
            # prob of applying mosaic aug
            self.mosaic_prob = 0
            # prob of applying mixup aug
            self.mixup_prob = 0
            # prob of applying hsv aug
            self.hsv_prob = 0
            # prob of applying flip aug
            self.flip_prob = 0.0
            # rotation angle range, for example, if set to 2, the true range is (-2, 2)
            self.degrees = 0.0
            # translate range, for example, if set to 0.1, the true range is (-0.1, 0.1)
            self.translate = 0
            self.mosaic_scale = (1, 1)
            # apply mixup aug or not
            self.enable_mixup = False
            self.mixup_scale = (1, 1)
            # shear angle range, for example, if set to 2, the true range is (-2, 2)
            self.shear = 0
            
        # Define yourself dataset path
        self.data_dir = "datasets/COCO/SWDD/"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"

        self.num_classes = 2#71

        self.max_epoch = 400
        self.data_num_workers = 4
        self.eval_interval = 10
