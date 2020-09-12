import torch.nn as nn

class BaseModel(nn.Module):
    """
        This is the base class model class. Every architecture in this library inherits this class
        
        Functions:
            
            __repr__: This function is used to update the representation of the BaseModel class

            __str__: This function is used to update the representation of the BaseModel class if printed.

            __init__: 

                Parameters: 
                        None

            forward:
            
                Parameters: *args

                This function returns the forward pass of the model
                If this function is not implemented it raises a Not implemented error. This function takes in all the input and stores it under args variable. Then that input is 
                used to perform the forward pass.

    """
    def __repr__(self):
        return "Base Model Function "
    
    def __str__(self):
        return "Base Model Function "

    def __init__(self):
        """
            Calling the init function of nn.module class and importing all its settings of nn.Module class
        """

        super().__init__()
    
    def forward(self,*args):
        """
            This function is for the definition of forward pass for the NN Model. It has to be overwritten 
            from the class inherting this class
        """

        raise "Not Implemented"