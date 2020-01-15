f =
    conv_stem
    act1
    blocks
    conv_head
    act2
    global_pool
    flatten
    classifier

conv_stem = NTKConv(c(), c(round(4 * h)), k=5, s=2, p=2, bias=False)  # 16x16
act1 = SwishJit()
blocks =
    DepthwiseSeparableConv(c(), c(round(2 * h)), k=5, s=1, p=2)
    InvertedResidual(c(), c(round(3 * h)), k=5, s=2, p=2, exp_ratio=3.0)  # 8x8
    InvertedResidual(c(), c(), k=5, s=1, p=2, exp_ratio=3.0)
    InvertedResidual(c(), c(round(5 * h)), k=5, s=2, p=2, exp_ratio=3.0)  # 4x4
    InvertedResidual(c(), c(), k=5, s=1, p=2, exp_ratio=3.0)

conv_head = NTKConv(c(), c(round(20 * h)), k=1, s=1, bias=False)
act2 = SwishJit()
global_pool = nn.AdaptiveAvgPool2d(1)
classifier = NTKLinear(c(), 1, bias=True)
