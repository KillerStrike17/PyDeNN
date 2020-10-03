import torch
class BaseTrainer:
    """
        This is the base trainer class,

        Functions:
            __repr__:

                This is a representation function, It returns the printable representation of the object.

            __str__:

                It returns useful string representation of the object.

            __init__:
                
                This is the constructor if the class, it takes in three parameters model, device and epochs
                and assigns it value to local variable
            
            train_step:
                
                This is the training step of each epoch.

            test_step:
                
                This is the training step of each epoch.

            train:

                This is the train function. It runs for epochs defined in the constructor and 
                calls test_step and train_step function and stores thier results in variables.
            
    """
    def __repr__(self):
        return "Base Trainer Class"
    
    def __str__(self):
        return "Base Trainer Class"

    def __init__(self, model,device:str,epochs:int,scheduler = None):
        """
            # Params: 
                
                model: It is the model achitecture which is being used.
                
                device: It contains the device information over which the model is running
                
                epochs: It is the total number of epochs a model runs
        """
        self.model = model
        self.device = device
        self.epochs = epochs
        self.scheduler = scheduler

    def train_step(self):
        """
            # Param:

                None
                
            # Raises:

                This function has to be implemented in the class inherting the base trainer class
                If not implemented it raises a NotImplemented Error.
        """
        raise NotImplementedError

    def test_step(self):
        """
            # Param:

                None
                
            # Raises:

                This function has to be implemented in the class inherting the base trainer class
                If not implemented it raises a NotImplemented Error.
        """
        raise NotImplementedError
    
    def train(self)->(tuple,tuple):
        """
            # Param:

                None
                
            # Returns:

                This function returns the train_accuracy, train_loss, test_accuracy and test_loss function
        
        """
        train_accuracy = []
        test_accuracy = []
        train_loss = []
        test_loss = []

        for _ in range(self.epochs):
            print("Epoch:",_+1)
            train_results = self.train_step()
            test_results = self.test_step()
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                val_loss = test_results[2]
                self.scheduler.step(val_loss)
            train_accuracy.extend(train_results[1])
            train_loss.extend(train_results[0])
            test_accuracy.extend(test_results[1])
            test_loss.extend(test_results[0])
        

        return ((train_accuracy,train_loss),(test_accuracy,test_loss))