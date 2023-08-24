# =========
# CIFAR-100
# =========
declare -a MODELS=("resnext50_32x4d")
DATASET=cifar100
for MODEL in ${MODELS[@]}; do
    echo $MODEL
    # python main_new.py --data_type $DATASET --classifier_type $MODEL
    # single addition
    python main_new.py --PSKD --data_type $DATASET --classifier_type $MODEL  
    python main_new.py --BYOT --data_type $DATASET --classifier_type $MODEL  
    python main_new.py --DML --data_type $DATASET --classifier_type $MODEL  
    # double addition
    python main_new.py --PSKD --BYOT --data_type $DATASET --classifier_type $MODEL  
    python main_new.py --BYOT --DML --data_type $DATASET --classifier_type $MODEL  
    python main_new.py --PSKD --DML --data_type $DATASET --classifier_type $MODEL  
    # triple addition
    python main_new.py --PSKD --BYOT --DML --data_type $DATASET --classifier_type $MODEL   --BYOT_from_k_block 3 

    # ablation study
    # BYOT block variation (default: 3)
    # python main_new.py --PSKD --BYOT --DML --data_type $DATASET --classifier_type $MODEL  --BYOT_from_k_block 1 
    # python main_new.py --PSKD --BYOT --DML --data_type $DATASET --classifier_type $MODEL  --BYOT_from_k_block 2 
    # python main_new.py --PSKD --BYOT --data_type $DATASET --classifier_type $MODEL --BYOT_from_k_block 1 
    # python main_new.py --PSKD --BYOT --data_type $DATASET --classifier_type $MODEL  --BYOT_from_k_block 2 
    # python main_new.py --BYOT --DML --data_type $DATASET --classifier_type $MODEL  --BYOT_from_k_block 1 
    # python main_new.py --BYOT --DML --data_type $DATASET --classifier_type $MODEL  --BYOT_from_k_block 2 

done