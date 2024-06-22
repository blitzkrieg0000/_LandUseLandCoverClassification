from torchvision.transforms import v2 as tranformsv2


class NormalizeSentinel2Transform(object):
    def __call__(self, sample):
        #? Sentinel-2 verilerini [0, 1] aralığına normalize etmek için 10000'e bölme işlemi yapılır
        return sample / 10000.0


TRANSFORM_IMAGE = tranformsv2.Compose([
    NormalizeSentinel2Transform()
])
