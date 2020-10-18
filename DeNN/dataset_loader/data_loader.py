from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from .util import download_and_extract_archive
import os, glob
from PIL import Image
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

class TINYIMAGENET:
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    filename = 'tiny-imagenet-200.zip'
    dataset_folder_name = 'tiny-imagenet-200'

    EXTENSION = 'JPEG'
    NUM_IMAGES_PER_CLASS = 500
    CLASS_LIST_FILE = 'wnids.txt'
    VAL_ANNOTATION_FILE = 'val_annotations.txt'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if download and (not os.path.isdir(os.path.join(self.root, self.dataset_folder_name))):
            self.download()

        self.split_dir = 'train' if train else 'val'
        self.split_dir = os.path.join(
            self.root, self.dataset_folder_name, self.split_dir)
        self.image_paths = sorted(glob.iglob(os.path.join(
            self.split_dir, '**', '*.%s' % self.EXTENSION), recursive=True))

        self.target = []
        self.labels = {}

        # build class label - number mapping
        with open(os.path.join(self.root, self.dataset_folder_name, self.CLASS_LIST_FILE), 'r') as fp:
            self.label_texts = sorted([text.strip()
                                       for text in fp.readlines()])
        self.label_text_to_number = {
            text: i for i, text in enumerate(self.label_texts)}

        # build labels for NUM_IMAGES_PER_CLASS images
        if train:
            for label_text, i in self.label_text_to_number.items():
                for cnt in range(self.NUM_IMAGES_PER_CLASS):
                    self.labels[f'{label_text}_{cnt}.{self.EXTENSION}'] = i

        # build the validation dataset
        else:
            with open(os.path.join(self.split_dir, self.VAL_ANNOTATION_FILE), 'r') as fp:
                for line in fp.readlines():
                    terms = line.split('\t')
                    file_name, label_text = terms[0], terms[1]
                    self.labels[file_name] = self.label_text_to_number[label_text]

        self.target = [self.labels[os.path.basename(
            filename)] for filename in self.image_paths]

    def download(self):
        download_and_extract_archive(
            self.url, self.root, filename=self.filename)

    def __getitem__(self, index):
        filepath = self.image_paths[index]
        img = Image.open(filepath)
        img = img.convert("RGB")
        target = self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
    def __len__(self):
        return len(self.image_paths)

class DatasetTinyImageNet:

    def __repr__(self):
        return "Loading TinyImageNet Dataset"

    def __str__(self):
        return "Loading TinyImageNet Dataset"

    def __init__(self, data_path,transformations,*, batch_size = 32 , shuffle = True, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None, generator=None):
        """
            This function initializes the dataset based on the values provided as parameters

            # Param:

                data_path: Root directory of dataset where TinyImageNet/processed/train.pt and TinyImageNet/processed/test.pt exist.

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
        self.train_set = TINYIMAGENET(root = self.data_path,train = True,download = True, transform = transformations.apply_transforms(train = True))
        self.test_set = TINYIMAGENET(root = self.data_path,train = False,download = True, transform = transformations.apply_transforms(train = False))
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