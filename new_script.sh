### RESNET scripts

# running==============================start: this server
    # Resnet-18 plain
    python main_new.py --data_type cifar100 --classifier_type ResNetBeMyOwnTeacher18

    # resnet18 PSKD + DML + BYOT
    python main_new.py --PSKD --BYOT --DML --data_type cifar100 --classifier_type ResNetBeMyOwnTeacher18   --BYOT_from_k_block 1

    # resnet18 PSKD + BYOT
    python main_new.py --PSKD --BYOT --data_type cifar100 --classifier_type ResNetBeMyOwnTeacher18   --BYOT_from_k_block 1

    # ablation: resnet18
    # python main_new.py --PSKD --BYOT --DML --data_type cifar100 --classifier_type ResNetBeMyOwnTeacher18   --BYOT_from_k_block 1 # 1,2,3 - final (default)
    python main_new.py --PSKD --BYOT --DML --data_type cifar100 --classifier_type ResNetBeMyOwnTeacher18   --BYOT_from_k_block 2 # 2,3 - final - running


# running==============================start: my server: cs5701
    # ablation: resnet18
    python main_new.py --PSKD --BYOT --DML --data_type cifar100 --classifier_type ResNetBeMyOwnTeacher18   --BYOT_from_k_block 3 # 3 - final

    # python main_new.py --PSKD --BYOT --DML --data_type cifar100 --classifier_type ResNetBeMyOwnTeacher18   --BYOT_from_k_block 1 --num_resnet_blocks 4 # (default)
    python main_new.py --PSKD --BYOT --DML --data_type cifar100 --classifier_type ResNetBeMyOwnTeacher18   --BYOT_from_k_block 1 --num_resnet_blocks 3
    python main_new.py --PSKD --BYOT --DML --data_type cifar100 --classifier_type ResNetBeMyOwnTeacher18   --BYOT_from_k_block 1 --num_resnet_blocks 2

    # running==============================start: my server: cs5702
    # resnet50 BYOT
    python main_new.py --BYOT --data_type cifar100 --classifier_type ResNetBeMyOwnTeacher50   --BYOT_from_k_block 1

    #  resnet50 DML
    python main_new.py --DML --data_type cifar100 --classifier_type ResNetBeMyOwnTeacher50  
    # resnet50 PSKD + DML + BYOT
    python main_new.py --PSKD --BYOT --DML --data_type cifar100 --classifier_type ResNetBeMyOwnTeacher50   --BYOT_from_k_block 1

    # resnet50 PSKD + DML
    python main_new.py --PSKD --DML --data_type cifar100 --classifier_type ResNetBeMyOwnTeacher50  

    # resnet50 PSKD + BYOT
    python main_new.py --PSKD --BYOT --data_type cifar100 --classifier_type ResNetBeMyOwnTeacher50   --BYOT_from_k_block 1



# TODO
    # resenext50 DML
    python main_new.py --DML --data_type cifar100 --classifier_type resnext50_32x4d

    # resenext50 BYOT + DML
    python main_new.py --BYOT --DML --data_type cifar100 --classifier_type resnext50_32x4d   --BYOT_from_k_block 1

