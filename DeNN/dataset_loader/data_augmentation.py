from abc import ABC,abstractmethod
import torchvision.transforms as T
import albumentations as A
import albumentations.pytorch.transforms as AT
import numpy as np

class BaseDataAugmentation(ABC):
    """
    This class is used to apply augmentation operations over the dataset.

    # Function:

        __repr__:

            This is a representation function, It returns the printable representation of the object.

        __str__:

            It returns useful string representation of the object.

        train_augmentation:

           Abstract function to apply train transformations

        test_augmentation:
            
            Abstract function to apply train transformations

        apply_transformations:

            This function is used to apply transformations

    """
    def __repr__(self):
        return "Base Data Augmentation Class"
    
    def __str__(self):
        return "Base Data Augmentation Class"

    @abstractmethod
    def train_augmentation(self):
        """
            This is an abstract method. The classs inheriting this class must write its own implementation of this method
        """
        pass
    
    @abstractmethod
    def test_augmentation(self):
        """
            This is an abstract method. The classs inheriting this class must write its own implementation of this method
        """
        pass

    def apply_transforms(self,train):
        """
            This function takes in train as a parameter and decides whether the generated data is of training set or test set.
        """
        return self.train_augmentation() if train else self.test_augmentation()
    
class AlbumentationTransforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        img = np.array(img)

        return self.transforms(image=img)['image']

class DataAugmentationMnist(BaseDataAugmentation):
    """
    This class is for Data Augmentation of MNIST Dataset

    # Function:

        __repr__:

            This is a representation function, It returns the printable representation of the object.

        __str__:

            It returns useful string representation of the object.

        train_augmentation:

            train transformation over dataset

        test_augmentation:
        
            test transformation over dataset

    """
    def __repr__(self):
        return "Data Augmentation Applied over MNIST Dataset"
    
    def __str__(self):
        return "Data Augmentation Applied over MNIST Dataset"

    def train_augmentation(self):
        """
            Training Augmentation applied on to the train dataset
        """
        return T.Compose([T.ToTensor(),T.Normalize((0.1307,), (0.3081,))])

    def test_augmentation(self):
        """
            Testing Augmentation applied on to the Test dataset
        """
        return T.Compose([T.ToTensor(),T.Normalize((0.1307,), (0.3081,))])

class DataAugmentationCifar10(BaseDataAugmentation):
    """
    This class is for Data Augmentation of CIFAR Dataset

    # Function:

        __repr__:

            This is a representation function, It returns the printable representation of the object.

        __str__:

            It returns useful string representation of the object.

        train_augmentation:

            train transformation over dataset

        test_augmentation:
            
            test transformation over dataset
    """
    def __repr__(self):
        return "Data Augmentation Applied over CIFAR 10 Dataset"
    
    def __str__(self):
        return "Data Augmentation Applied over CIFAR 10 Dataset"

    def train_augmentation(self):
        """
            Training Augmentation applied on to the train dataset
        """
        return T.Compose([T.RandomCrop(32, padding=4),T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    def test_augmentation(self):
        """
            Testing Augmentation applied on to the Test dataset
        """
        return T.Compose([T.ToTensor(),T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
class DataAugmentationCifar10_album(BaseDataAugmentation):
    """
    This class is for Data Augmentation of CIFAR Dataset

    # Function:

        __repr__:

            This is a representation function, It returns the printable representation of the object.

        __str__:

            It returns useful string representation of the object.

        train_augmentation:

            train transformation over dataset

        test_augmentation:
            
            test transformation over dataset
    """
    def __repr__(self):
        return "Data Augmentation Applied over CIFAR 10 Dataset"
    
    def __str__(self):
        return "Data Augmentation Applied over CIFAR 10 Dataset"

    def train_augmentation(self):
        """
            Training Augmentation applied on to the train dataset
        """
        return AlbumentationTransforms(A.Compose([A.PadIfNeeded(min_height=36, min_width=36),A.RandomCrop(32, 32),A.HorizontalFlip(),A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),A.Cutout(num_holes=4),AT.ToTensor()]))

    def test_augmentation(self):
        """
            Testing Augmentation applied on to the Test dataset
        """
        return AlbumentationTransforms(A.Compose([A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),AT.ToTensor()]))


class TinyImageNetAlbumentations(BaseDataAugmentation):
    """
    This class is for Data Augmentation of TinyImageNet Dataset

    # Function:

        __repr__:

            This is a representation function, It returns the printable representation of the object.

        __str__:

            It returns useful string representation of the object.

        train_augmentation:

            train transformation over dataset

        test_augmentation:
            
            test transformation over dataset
    """
    def __repr__(self):
        return "Data Augmentation Applied over TinyImageNet Dataset"
    
    def __str__(self):
        return "Data Augmentation Applied over TinyImageNet Dataset"

    def train_augmentation(self):
        """
            Training Augmentation applied on to the train dataset
        """
        return AlbumentationTransforms(A.Compose([A.RandomCrop(64, 64),A.Rotate((-30.0, 30.0)),A.HorizontalFlip(),A.Normalize(mean=[0.4802, 0.4481, 0.3975],std=[0.2302, 0.2265, 0.2262]),A.Cutout(num_holes=4),AT.ToTensor()]))

    def test_augmentation(self):
        """
            Testing Augmentation applied on to the Test dataset
        """
        return AlbumentationTransforms(A.Compose([A.Normalize(mean=[0.4802, 0.4481, 0.3975],std=[0.2302, 0.2265, 0.2262]),AT.ToTensor()]))


class Dataset_Mean_and_std():
    
    def mean_and_std(self,dataset):
        if dataset == "cifar10":
            return (0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)
        if dataset == "tinyimagenet":
            return (0.4802, 0.4481, 0.3975),(0.2302, 0.2265, 0.2262)