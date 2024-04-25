
#!/bin/bash

TRAIN_DIR="/home/c4ndypuff/Documentos/end2end-all-conv-master_copia/ddsm_train/CBIS_train"
VAL_DIR="/home/c4ndypuff/Documentos/end2end-all-conv-master_copia/ddsm_train/CBIS_val"
TEST_DIR="/home/c4ndypuff/Documentos/end2end-all-conv-master_copia/ddsm_train/CBIS_test"
RESUME_FROM="/home/c4ndypuff/Documentos/end2end-all-conv-master_copia/ddsm_train/ddsm_resnet50_s10_[512-512-1024]x2.h5"
BEST_MODEL="/home/c4ndypuff/Documentos/end2end-all-conv-master_copia/ddsm_train/CBISDDSM_resnet_nlr.2.h5"
FINAL_MODEL="NOSAVE"
export NUM_CPU_CORES=4

python image_clf_train.py \
    --no-patch-model-state \
    --resume-from $RESUME_FROM \
    --img-size 1152 896 \
    --no-img-scale \
    --rescale-factor 1 \
    --featurewise-center \
    --featurewise-mean 44.3 \
    --no-equalize-hist \
    --batch-size 2 \
    --train-bs-multiplier 0.5 \
    --augmentation \
    --class-list neg pos \
    --nb-epoch 0 \
    --all-layer-epochs 50 \
    --load-val-ram \
    --no-load-train-ram \
    --optimizer adam \
    --weight-decay 0.001 \
    --hidden-dropout 0.0 \
    --weight-decay2 0.01 \
    --hidden-dropout2 0.0 \
    --init-learningrate 0.0001 \
    --all-layer-multiplier 0.01 \
    --es-patience 10 \
    --auto-batch-balance \
    --best-model $BEST_MODEL \
    --final-model $FINAL_MODEL \
    $TRAIN_DIR $VAL_DIR $TEST_DIR
