from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

class DatasetMnist:
    def __repr__(self):
        return "Loading MNIST Dataset"

    def __str__(self):
        return "Loading MNIST Dataset"

    def __init__(self, data_path, *, batch_size = 32 , shuffle = True,transformations, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None, generator=None):
        self.data_path = data_path
        self.train_set = datasets.MNIST(root = self.data_path,train = True,download = True, transform = transformations.apply_transforms(train = True))
        self.test_set = datasets.MNIST(root = self.data_path,train = False,download = True, transform = transformations.apply_transforms(train = False))
        self.params = {
            'shuffle': shuffle,
            'batch_size': batch_size,
            'sampler': sampler,
            'batch_sampler':batch_sampler,
            'collate_fn':collate_fn,
            'drop_last':drop_last,
            'timeout':timeout,
            'worker_init_fn':worker_init_fn,
            'multiprocessing_context':multiprocessing_context,
            'generator':generator,
            'num_workers': num_workers,
            'pin_memory': pin_memory
        }

    def load_data(self):
        return DataLoader(self.train_set,**self.params),DataLoader(self.test_set,**self.params)
    
class DatasetCifar10:
    def __repr__(self):
        return "Loading CIFAR 10 Dataset"

    def __str__(self):
        return "Loading CIFAR 10 Dataset"

    def __init__(self, data_path,transformations,*, batch_size = 32 , shuffle = True, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None, generator=None):
        self.data_path = data_path
        self.train_set = datasets.CIFAR10(root = self.data_path,train = True,download = True, transform = transformations.apply_transforms(train = True))
        self.test_set = datasets.CIFAR10(root = self.data_path,train = False,download = True, transform = transformations.apply_transforms(train = False))
        self.params = {
            'shuffle': shuffle,
            'batch_size': batch_size,
            'sampler': sampler,
            'batch_sampler':batch_sampler,
            'collate_fn':collate_fn,
            'drop_last':drop_last,
            'timeout':timeout,
            'worker_init_fn':worker_init_fn,
            'multiprocessing_context':multiprocessing_context,
            'generator':generator,
            'num_workers': num_workers,
            'pin_memory': pin_memory
        }

    def load_data(self):
        return DataLoader(self.train_set,**self.params),DataLoader(self.test_set,**self.params)