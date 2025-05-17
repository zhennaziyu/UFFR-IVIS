from .mini_imagenet import MiniImageNet
from .cub import CUB
from .tiered_imagenet import TieredImageNet
from .cifarfs import CIFARFS
from .fc100 import FC100
from .Food2K import Food_2K
dataset_dict = {
    'MiniImageNet': MiniImageNet,
    'CUB': CUB,
    'TieredImageNet': TieredImageNet,
    'CIFAR-FS': CIFARFS,
    'FC100': FC100,
    'Food2K': Food_2K
}
