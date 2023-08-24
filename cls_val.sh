# classifier 1/4 - 4/4, ensemble results (val only)

# resnet18
python main_new.py --val_only --classifier_type ResNetBeMyOwnTeacher18 --data_type cifar100 --batch_size 128 --end_epoch 1 --resume PATH

python main_new.py --val_only  --classifier_type ResNetBeMyOwnTeacher18 --data_type cifar100 --batch_size 128 --end_epoch 1 --resume /root/term_project/latest_version_3/experiments/_2023-5-28-15-25-48_BYOT_from_k_block_1_num_resnet_blocks_4_cifar100_ResNetBeMyOwnTeacher18_PSKD_False_DML_False_BYOT_False_DMLonoutput_False/model/checkpoint_best.pth


# resnet50
python main_new.py --val_only --resume /root/term_project/latest_version_3/experiments/completed/other_machine/2023-5-21-12-2-1_BYOT_from_k_block_1__num_resnet_blocks_4_cifar100_ResNetBeMyOwnTeacher50_PSKD_False_DML_False_BYOT_False_DMLonoutput_False/model/checkpoint_best.pth --classifier_type ResNetBeMyOwnTeacher50 --data_type cifar100 --batch_size 128 --end_epoch 1

python main_new.py --val_only --classifier_type ResNetBeMyOwnTeacher50 --data_type cifar100 --batch_size 128 --end_epoch 1 --resume PATH

# resnext50
python main_new.py --val_only --resume /root/term_project/latest_version_3/experiments/completed/2023-5-24-2-57-20_BYOT_from_k_block_1__num_resnet_blocks_4_cifar100_resnext50_32x4d_PSKD_True_DML_True_BYOT_True_DMLonoutput_False/model/checkpoint_best.pth --classifier_type resnext50_32x4d --data_type cifar100 --batch_size 128 --end_epoch 1



