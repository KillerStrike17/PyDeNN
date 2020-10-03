from DeNN.base import BaseTrainer
from tqdm import tqdm
import torch

class Trainer(BaseTrainer):
    """
    
        # Function:
        
            __init__:

                Here we initialize train_loader and test_loader

            train_step:

                This function is to perform back propogation over train_step

            test_step:

                This function is used to evaluate the test dataset

        
    """
    def __init__(self, model, optimizer, device, train_loader, test_loader,epochs,criteria,scheduler):
        super().__init__(model, device, epochs)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criteria = criteria
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train_step(self):
        loss_data = []
        accuracy_data = []
        self.model.train()

        train_loss = 0
        correct = 0
        processed = 0
        pbar = tqdm(self.train_loader)

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criteria(output, target)
            loss.backward()
            self.optimizer.step()
            pred = output.argmax(dim=1,keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx} Accuracy = {correct/len(self.train_loader.dataset)}')
            processed += len(data)
            accuracy_data.append(100*correct/processed)
            loss_data.append(loss.data.cpu().numpy().item())
        return loss_data,accuracy_data

    def test_step(self):
        loss_data = []
        accuracy_data = []
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criteria(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))
        accuracy_data.append(100*correct/len(self.test_loader.dataset))
        loss_data.append(test_loss)
        return loss_data,accuracy_data,test_loss