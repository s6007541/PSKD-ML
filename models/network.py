from models import pyramid, pyramid_shake_drop, preact_resnet, densenet_cifar, resnext, resnet_imagenet
from models import resnet_BYOT as BYOT
from models import resnet_additional as BYOT_ADD
from utils.color import Colorer


C = Colorer.instance()

def get_network(args):
    ################################
    #Declare instance for Clasifier#
    ################################
    # if args.data_type == 'cifar100':
    #     if args.classifier_type == 'PyramidNet':
    #         net = pyramid.PyramidNet(dataset = 'cifar100', depth=200, alpha=240, num_classes=100,bottleneck=True)
    #     elif args.classifier_type == 'PyramidNet_SD':
    #         net = pyramid_shake_drop.PyramidNet_ShakeDrop(dataset = 'cifar100', depth=200, alpha=240, num_classes=100,bottleneck=True)
    #     elif args.classifier_type == 'ResNet18':
    #         net = preact_resnet.CIFAR_ResNet18_preActBasic(num_classes=100)
    #     elif args.classifier_type == 'ResNet101':
    #         net = preact_resnet.CIFAR_ResNet101_Bottle(num_classes=100)
    #     elif args.classifier_type == 'DenseNet121':
    #         net = densenet_cifar.CIFAR_DenseNet121(num_classes=100, bias=True)
    #     elif args.classifier_type == 'ResNeXt':
    #         net = resnext.CifarResNeXt(cardinality=8, depth=29, nlabels=100, base_width=64, widen_factor=4)
    #     # ==============
    #     # our addition
    #     # ==============
    #     elif args.classifier_type == "ResNetBeMyOwnTeacher18":
    #         net = BYOT.multi_resnet18_kd(num_classes=100, num_resnet_blocks=args.num_resnet_blocks)
    #     elif args.classifier_type == "ResNetBeMyOwnTeacher50":
    #         net = BYOT.multi_resnet50_kd(num_classes=100)
    #     else:
    #         raise NotImplementedError
        

    # if args.data_type == 'imagenet':
    #     if args.classifier_type == 'ResNet152':
    #         net = resnet_imagenet.ResNet(dataset = 'imagenet', depth=152, num_classes=1000, bottleneck=True)
    #     else:
    #         raise NotImplementedError
    # if args.data_type == 'tinyimagenet':
    #     if args.classifier_type == 'ResNet152':
    #         net = resnet_imagenet.ResNet(dataset='tinyimagenet', depth=152, num_classes=200, bottleneck=True)
        
    if args.data_type in ['cifar100', 'tinyimagenet']:
        if args.classifier_type == "ResNetBeMyOwnTeacher18":
            net = BYOT.multi_resnet18_kd(num_classes=100, num_resnet_blocks=args.num_resnet_blocks)
        elif args.classifier_type == "ResNetBeMyOwnTeacher50":
            """ what we use """
            net = BYOT.multi_resnet50_kd(num_classes=100)
        # elif args.classifier_type == "resnet18":
            # net = BYOT_ADD.resnet18()
        elif args.classifier_type == "resnet34":
            net = BYOT_ADD.resnet34()
        # elif args.classifier_type == "resnet50":
            # net = BYOT_ADD.resnet50()
        elif args.classifier_type == "resnet101":
            net = BYOT_ADD.resnet101()
        elif args.classifier_type == "resnet152":
            net = BYOT_ADD.resnet152()
        elif args.classifier_type == "wideresnet50":
            net = BYOT_ADD.wide_resnet50_2()
        elif args.classifier_type == "wideresnet101":
            net = BYOT_ADD.wide_resnet101_2()
        elif args.classifier_type == "resnext50_32x4d":
            net = BYOT_ADD.resnext50_32x4d()
        elif args.classifier_type == "resnext101_32x8d":
            net = BYOT_ADD.resnext101_32x8d()
            
        else:
            raise NotImplementedError
    print(C.underline(C.yellow("[Info] Building model: {}".format(args.classifier_type))))


    return net