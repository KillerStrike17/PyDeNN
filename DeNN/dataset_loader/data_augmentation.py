from abc import ABC,abstractmethod
import torchvision.transforms as T

class BaseDataAugmentation(ABC):
    
    def __repr__(self):
        return "Base Data Augmentation Class"
    
    def __str__(self):
        return "Base Data Augmentation Class"

    @abstractmethod
    def train_augmentation(self):
        pass
    
    @abstractmethod
    def test_augmentation(self):
        pass

    def apply_transforms(self,train):
        return self.train_augmentation() if train else self.test_augmentation()

class DataAugmentationMnist(BaseDataAugmentation):
    def __repr__(self):
        return "Data Augmentation Applied over MNIST Dataset"
    
    def __str__(self):
        return "Data Augmentation Applied over MNIST Dataset"

    def train_augmentation(self):
        return T.Compose([T.ToTensor(),T.Normalize((0.1307,), (0.3081,))])

    def test_augmentation(self):
        return T.Compose([T.ToTensor(),T.Normalize((0.1307,), (0.3081,))])

class DataAugmentationCifar10(BaseDataAugmentation):
    def __repr__(self):
        return "Data Augmentation Applied over CIFAR 10 Dataset"
    
    def __str__(self):
        return "Data Augmentation Applied over CIFAR 10 Dataset"

    def train_augmentation(self):
        return T.Compose([T.RandomCrop(32, padding=4),T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    def test_augmentation(self):
        return T.Compose([T.ToTensor(),T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
