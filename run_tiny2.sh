# =========
# TinyImageNet
# =========
# baseline
declare -a MODELS=("resnet152" "wideresnet50" "wideresnet101" "resnext50_32x4d" "resnext101_32x8d")
for MODEL in ${MODELS[@]}; do
    echo $MODEL
    python main_new.py --data_type tinyimagenet --classifier_type $MODEL  &&
    # single addition
    python main_new.py --PSKD --data_type tinyimagenet --classifier_type $MODEL  &&
    python main_new.py --BYOT --data_type tinyimagenet --classifier_type $MODEL  &&
    python main_new.py --DML --data_type tinyimagenet --classifier_type $MODEL  &&
    # double addition
    python main_new.py --PSKD --BYOT --data_type tinyimagenet --classifier_type $MODEL  &&
    python main_new.py --BYOT --DML --data_type tinyimagenet --classifier_type $MODEL  &&
    python main_new.py --PSKD --DML --data_type tinyimagenet --classifier_type $MODEL  &&
    # triple addition
    python main_new.py --PSKD --BYOT --DML --data_type tinyimagenet --classifier_type $MODEL  --BYOT_from_k_block 3 &&
    # ablation study
    # BYOT block variation (defualt: 3)
    python main_new.py --PSKD --BYOT --DML --data_type tinyimagenet --classifier_type $MODEL  --BYOT_from_k_block 1 &&
    python main_new.py --PSKD --BYOT --DML --data_type tinyimagenet --classifier_type $MODEL  --BYOT_from_k_block 2 &&
    python main_new.py --PSKD --BYOT --data_type tinyimagenet --classifier_type $MODEL  --BYOT_from_k_block 1 &&
    python main_new.py --PSKD --BYOT --data_type tinyimagenet --classifier_type $MODEL  --BYOT_from_k_block 2 &&
    python main_new.py --BYOT --DML --data_type tinyimagenet --classifier_type $MODEL  --BYOT_from_k_block 1 &&
    python main_new.py --BYOT --DML --data_type tinyimagenet --classifier_type $MODEL  --BYOT_from_k_block 2
    # run with model variation (lighter model):           NOTE: default: 4 resnet blocks in resnet_BYOT
    # python main_new.py --PSKD --BYOT --DML --BYOT_from_k_block 3 --data_type tinyimagenet --classifier_type $MODEL  --num_resnet_blocks 3
    # python main_new.py --PSKD --BYOT --DML --BYOT_from_k_block 2 --data_type tinyimagenet --classifier_type $MODEL  --num_resnet_blocks 3
    # python main_new.py --PSKD --BYOT --DML --BYOT_from_k_block 1 --data_type tinyimagenet  --classifier_type $MODEL --num_resnet_blocks 3
done