#!/bin/bash

TRAIN_DIR="/home/c4ndypuff/Documentos/end2end-all-conv-master_copia/ddsm_train/TrainP"
VAL_DIR="/home/c4ndypuff/Documentos/end2end-all-conv-master_copia/ddsm_train/ValP"
TEST_DIR="/home/c4ndypuff/Documentos/end2end-all-conv-master_copia/ddsm_train/TestP"
RESUME_FROM="/home/c4ndypuff/Documentos/end2end-all-conv-master_copia/ddsm_train/ddsm_resnet50_s10_[512-512-1024]x2.h5"
BEST_MODEL="/home/c4ndypuff/Documentos/end2end-all-conv-master_copia/ddsm_train/DDSM_settings3_lrch_5.h5"
FINAL_MODEL="NOSAVE"
export NUM_CPU_CORES=4

python image_clf_train.py \
	--no-patch-model-state \
	--resume-from $RESUME_FROM \
    --img-size 1152 896 \
    --no-img-scale \
    --rescale-factor 1 \
	--featurewise-center \
    --featurewise-mean 44.33 \
    --no-equalize-hist \
    --patch-net resnet50 \
    --block-type resnet \
    --top-depths 512 512 \
    --top-repetitions 2 2 \
    --bottleneck-enlarge-factor 2 \
    --no-add-heatmap \
    --avg-pool-size 7 7 \
    --add-conv \
    --no-add-shortcut \
    --hm-strides 1 1 \
    --hm-pool-size 5 5 \
    --fc-init-units 64 \
    --fc-layers 2 \
    --batch-size 4 \
    --train-bs-multiplier 0.5 \
	--augmentation \
	--class-list neg pos \
	--nb-epoch 0 \
    --all-layer-epochs 30 \
    --load-val-ram \
    --no-load-train-ram \
    --optimizer adam \
    --weight-decay 0.0005 \
    --hidden-dropout 0.0 \
    --weight-decay2 0.0005 \
    --hidden-dropout2 0.1 \
    --init-learningrate 0.0001 \
    --all-layer-multiplier 0.01 \
	--lr-patience 2 \
	--es-patience 10 \
	--auto-batch-balance \
    --pos-cls-weight 1.0 \
	--neg-cls-weight 1.0 \
	--best-model $BEST_MODEL \
	--final-model $FINAL_MODEL \
	$TRAIN_DIR $VAL_DIR $TEST_DIR
