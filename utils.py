import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms(args, dts):
    if dts == 'Aircraft':
        MEAN = [0.4785, 0.5100, 0.5338]
        STD = [0.1743, 0.1731, 0.1973]
    elif dts == 'ImageNet':
        MEAN = [0.4807, 0.4485, 0.3980]
        STD = [0.2016, 0.1979, 0.1976]
    elif dts == 'dtd':
        MEAN = [0.5276, 0.4714, 0.4234]
        STD = [0.1220, 0.1221, 0.1194]
    elif dts == 'birds':
        MEAN = [0.4860, 0.4997, 0.4319]
        STD = [0.1489, 0.1481, 0.1597]
    elif dts == 'flower':
        MEAN = [0.4860, 0.4997, 0.4319]
        STD = [0.1489, 0.1481, 0.1597]
    elif dts == 'omniglot':
        MEAN = [0.9195, 0.9195, 0.9195]
        STD = [0.2210, 0.2210, 0.2210]

    if args.dataset == 'omniglot':
        train_transform = transforms.Compose([
            transforms.Resize([84, 84]),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomCrop(84, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

        if args.cutout:
            train_transform.transforms.append(Cutout(args.cutout_length))
        
        valid_transform = transforms.Compose([
            transforms.Resize([84, 84]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        return train_transform, valid_transform
    else: 
        train_transform = transforms.Compose([
            transforms.Resize([84, 84]),
            transforms.RandomCrop(84, padding=20),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

        if args.cutout:
            train_transform.transforms.append(Cutout(args.cutout_length))
        
        valid_transform = transforms.Compose([
            transforms.Resize([84, 84]),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        return train_transform, valid_transform


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def _data_transforms_cifar100(args):
    CIFAR_MEAN = [0.5071, 0.4867, 0.4408]
    CIFAR_STD = [0.2675, 0.2565, 0.2761]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def _data_transforms_mnist(args):
    MEAN = [0.13066051707548254]
    STD = [0.30810780244715075]

    if args.dataset == 'Omniglot':
        size = 28

    train_transform = transforms.Compose([
        transforms.RandomCrop(size, padding=3),
        transforms.RandomHorizontalFlip(),
        transforms.Resize([84, 84]),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_transform, valid_transform


def _data_transforms_mnist(args):
    MEAN = [0.13066051707548254]
    STD = [0.30810780244715075]

    train_transform = transforms.Compose([
        transforms.RandomCrop(28, padding=3),
        transforms.RandomHorizontalFlip(),
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        #transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_transform, valid_transform


def _data_transforms_omniglot(args):
    MEAN = [0.13066051707548254]
    STD = [0.30810780244715075]

    train_transform = transforms.Compose([
        #transforms.RandomCrop(28, padding=3),
        # transforms.RandomHorizontalFlip(),
        # transforms.Resize([32,32]),
        transforms.ToTensor(),
        #transforms.Normalize(MEAN, STD),
        #transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_transform, valid_transform


def _data_transforms_fashion_mnist(args):
    MEAN = [0.28604063146254594]
    STD = [0.35302426207299326]

    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(
            0.1, 0.1), scale=(0.9, 1.1), shear=0.1),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_transform, valid_transform


def _data_transforms_svhn(args):
    SVHN_MEAN = [0.4380, 0.4440, 0.4730]
    SVHN_STD = [0.1751, 0.1771, 0.1744]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(SVHN_MEAN, SVHN_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(SVHN_MEAN, SVHN_STD),
    ])
    return train_transform, valid_transform


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1.-drop_prob
        mask = Variable(torch.cuda.FloatTensor(
            x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
