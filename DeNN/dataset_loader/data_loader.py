from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

class DatasetMnist:
    """
        This class loads MNIST dataset with applied transformations

        # Functions:

            __repr__:

                This is a representation function, It returns the printable representation of the object.

            __str__:

                It returns useful string representation of the object.

            __init__:

                This is the constructor of DataMnist class< It initializes dataset and applied transformations over it.
            
            load_data:

                This function returns the generated datasaet.
    """
    def __repr__(self):
        return "Loading MNIST Dataset"

    def __str__(self):
        return "Loading MNIST Dataset"

    def __init__(self, data_path:str, *, batch_size = 32, shuffle = True,transformations, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None, generator=None):
        """
            This function initializes the dataset based on the values provided as parameters

            # Param:

                data_path: Root directory of dataset where MNIST/processed/training.pt and MNIST/processed/testing.pt exist.

                batch_size (int, optional): how many samples per batch to load (default: ``1``).

                shuffle (bool, optional): set to ``True`` to have the data reshuffled at every epoch (default: ``False``).

                transformations:  A function/transform that takes in an PIL image and returns a transformed version. E.g, transforms.RandomCrop                
        
                sampler (Sampler, optional): defines the strategy to draw samples from the dataset. If specified, ``shuffle`` must be False.
                
                batch_sampler (Sampler, optional): like sampler, but returns a batch ofindices at a time. Mutually exclusive with :attr:`batch_size`,:attr:`shuffle`, :attr:`sampler`, and :attr:`drop_last`.
        
                num_workers (int, optional): how many subprocesses to use for dataloading. 0 means that the data will be loaded in the main process.(default: ``0``)
                
                collate_fn (callable, optional): merges a list of samples to form a mini-batch.
                
                pin_memory (bool, optional): If ``True``, the data loader will copy tensors into CUDA pinned memory before returning them.  If your data elements are a custom type, or your ``collate_fn`` returns a batch that is a custom type
                                             see the example below.
                
                drop_last (bool, optional): set to ``True`` to drop the last incomplete batch, if the dataset size is not divisible by the batch size. If ``False`` and the size of dataset is not divisible by the batch size, then the last batch
                                            will be smaller. (default: ``False``)

                timeout (numeric, optional): if positive, the timeout value for collecting a batch from workers. Should always be non-negative. (default: ``0``)
                
                worker_init_fn (callable, optional): If not ``None``, this will be called on each worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as input, after seeding and before data loading. (default: ``None``)
        """
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
        """
            This function returns the generated dataset

            # Returns:

                Dataset,Dataset : Generated test and train dataset
        """
        return DataLoader(self.train_set,**self.params),DataLoader(self.test_set,**self.params)
    
class DatasetCifar10:

    def __repr__(self):
        return "Loading CIFAR 10 Dataset"

    def __str__(self):
        return "Loading CIFAR 10 Dataset"

    def __init__(self, data_path,transformations,*, batch_size = 32 , shuffle = True, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None, generator=None):
        """
            This function initializes the dataset based on the values provided as parameters

            # Param:

                data_path: Root directory of dataset where CIFAR10/processed/train.pt and CIFAR10/processed/test.pt exist.

                batch_size (int, optional): how many samples per batch to load (default: ``1``).

                shuffle (bool, optional): set to ``True`` to have the data reshuffled at every epoch (default: ``False``).

                transformations:  A function/transform that takes in an PIL image and returns a transformed version. E.g, transforms.RandomCrop                
        
                sampler (Sampler, optional): defines the strategy to draw samples from the dataset. If specified, ``shuffle`` must be False.
                
                batch_sampler (Sampler, optional): like sampler, but returns a batch ofindices at a time. Mutually exclusive with :attr:`batch_size`,:attr:`shuffle`, :attr:`sampler`, and :attr:`drop_last`.
        
                num_workers (int, optional): how many subprocesses to use for dataloading. 0 means that the data will be loaded in the main process.(default: ``0``)
                
                collate_fn (callable, optional): merges a list of samples to form a mini-batch.
                
                pin_memory (bool, optional): If ``True``, the data loader will copy tensors into CUDA pinned memory before returning them.  If your data elements are a custom type, or your ``collate_fn`` returns a batch that is a custom type
                                             see the example below.
                
                drop_last (bool, optional): set to ``True`` to drop the last incomplete batch, if the dataset size is not divisible by the batch size. If ``False`` and the size of dataset is not divisible by the batch size, then the last batch
                                            will be smaller. (default: ``False``)

                timeout (numeric, optional): if positive, the timeout value for collecting a batch from workers. Should always be non-negative. (default: ``0``)
                
                worker_init_fn (callable, optional): If not ``None``, this will be called on each worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as input, after seeding and before data loading. (default: ``None``)
        """
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
        """
            This function returns the generated dataset

            # Returns:

                Dataset,Dataset : Generated test and train dataset
        """
        return DataLoader(self.train_set,**self.params),DataLoader(self.test_set,**self.params)