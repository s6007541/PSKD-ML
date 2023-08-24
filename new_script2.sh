### ResNeXt50 scripts



### ResNeXt50 DML
python main_new.py  --DML --data_type cifar100 --classifier_type resnext50_32x4d

### ResNeXt50 PSKD + DML + BYOT
python main_new.py --PSKD --BYOT --DML --data_type cifar100 --classifier_type resnext50_32x4d