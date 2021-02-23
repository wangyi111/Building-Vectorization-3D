from models.modeling.backbone import resnet, xception, drn, mobilenet, xception65, xception_dense_feature

def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'xception65':
        return xception65.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'xception_dense_feature':
        return xception_dense_feature.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError
