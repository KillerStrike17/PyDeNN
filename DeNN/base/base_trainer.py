class BaseTrainer:
    
    def __init__(self, model, optimizer,device,epochs):

        self.model = model
        self.device = device
        self.epochs = epochs

    def _train_epoch(self, epoch: int):
        raise NotImplementedError

    def _test_epoch(self, epoch: int):
        raise NotImplementedError
    
    def train(self):

        train_accuracy = []
        test_accuracy = []
        train_loss = []
        test_loss = []

        for epoch in range(self.epochs):
            train_results = self._train_epoch(epoch)
            test_results = self._test_epoch(epoch)
            train_accuracy.extend(train_results[0])
            train_loss.extend(train_results[1])
            test_accuracy.extend(test_results[0])
            test_loss.extend(test_results[1])
        

        return ((train_accuracy,train_loss),(test_accuracy,test_loss))