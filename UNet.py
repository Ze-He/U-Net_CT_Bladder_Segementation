# encoding: utf-8
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, concatenate


class UNet(Model):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder1 = Sequential([
            Conv2D(64, kernel_size=3, strides=1, padding='valid', activation='relu'),  # 向下编码第一层，2个3x3卷积核向下卷积
            Conv2D(64, kernel_size=3, strides=1, padding='valid', activation='relu')
        ])
        self.encoder2 = Sequential([
            MaxPooling2D(pool_size=2),  # 最大池化，缩减尺寸
            Conv2D(128, kernel_size=3, strides=1, padding='valid', activation='relu'),  # 向下编码第二层
            Conv2D(128, kernel_size=3, strides=1, padding='valid', activation='relu')
        ])
        self.encoder3 = Sequential([
            MaxPooling2D(pool_size=2),
            Conv2D(256, kernel_size=3, strides=1, padding='valid', activation='relu'),  # 向下编码第三层
            Conv2D(256, kernel_size=3, strides=1, padding='valid', activation='relu')
        ])
        self.encoder4 = Sequential([
            MaxPooling2D(pool_size=2),
            Conv2D(512, kernel_size=3, strides=1, padding='valid', activation='relu'),  # 向下编码第四层
            Conv2D(512, kernel_size=3, strides=1, padding='valid', activation='relu')

        ])
        self.encoder5 = Sequential([
            MaxPooling2D(pool_size=2),
            Conv2D(1024, kernel_size=3, strides=1, padding='valid', activation='relu'),  # 向下编码第五层
            Conv2D(1024, kernel_size=3, strides=1, padding='valid', activation='relu')
        ])
        self.decoder1 = Sequential([
            Conv2DTranspose(512, kernel_size=2, strides=2, padding='same', activation='relu'),  # 向上编码第一层
        ])
        self.decoder2 = Sequential([
            Conv2D(512, kernel_size=3, strides=1, padding='valid', activation='relu'),
            Conv2D(512, kernel_size=3, strides=1, padding='valid', activation='relu'),
            Conv2DTranspose(256, kernel_size=2, strides=2, padding='same', activation='relu'),  # 向上编码第二层

        ])
        self.decoder3 = Sequential([
            Conv2D(256, kernel_size=3, strides=1, padding='valid', activation='relu'),
            Conv2D(256, kernel_size=3, strides=1, padding='valid', activation='relu'),
            Conv2DTranspose(128, kernel_size=2, strides=2, padding='same', activation='relu'),  # 向上编码第三层

        ])
        self.decoder4 = Sequential([
            Conv2D(128, kernel_size=3, strides=1, padding='valid', activation='relu'),
            Conv2D(128, kernel_size=3, strides=1, padding='valid', activation='relu'),
            Conv2DTranspose(64, kernel_size=2, strides=2, padding='same', activation='relu'),  # 向上编码第四层
        ])
        self.decoder5 = Sequential([
            Conv2D(64, kernel_size=3, strides=1, padding='valid', activation='relu'),
            Conv2D(64, kernel_size=3, strides=1, padding='valid', activation='relu'),
            Conv2D(1, kernel_size=1, strides=1, padding='valid', activation='sigmoid')
        ])

    def call(self, inputs):
        # 向下编码
        layer1 = self.encoder1(inputs)
        layer2 = self.encoder2(layer1)
        layer3 = self.encoder3(layer2)
        layer4 = self.encoder4(layer3)
        layer5 = self.encoder5(layer4)
        # 向上编码
        layer6 = self.decoder1(layer5)
        layer4_crop = layer4[:, 4:60, 4:60, :]      # 这里需要对张量进行剪切
        temp1 = concatenate([layer4_crop, layer6])  # 复制 + 拼接， 第一层
        layer7 = self.decoder2(temp1)

        layer3_crop = layer3[:, 16:120, 16:120, :]
        temp2 = concatenate([layer3_crop, layer7])  # 复制 + 拼接， 第二层
        layer8 = self.decoder3(temp2)

        layer2_crop = layer2[:, 40:240, 40:240, :]
        temp3 = concatenate([layer2_crop, layer8])  # 复制 + 拼接， 第三层
        layer9 = self.decoder4(temp3)

        layer1_crop = layer1[:, 88:480, 88:480, :]
        temp4 = concatenate([layer1_crop, layer9])  # 复制 + 拼接， 第四层
        output = self.decoder5(temp4)

        return output
