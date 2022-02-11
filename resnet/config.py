from torch import cuda, nn

batchsize = 256
iteration = 5
device = 'cuda' if cuda.is_available() else 'cpu'
learning_rate = 0.1
criterion = nn.CrossEntropyLoss()

transform_param = {
    'short_resize': 256,
    'long_resize' : 480,
    'random_crop' : 224,
    'pixel_mean'  : (0.5, 0.5, 0.5),
    'pixel_std'   : (0.5, 0.5, 0.5),
}