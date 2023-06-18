
# %load stepbystep/v0.py
### Fron: Deep_Learning with Pythorch Step By Step. Volume II. Daniel Voigt Godoy
### https://github.com/dvgodoy/PyTorchStepByStep 
 
import numpy as np
import datetime
import torch
torch.backends.cudnn.benchmark = False ## To ensure Torch choice the same algorith all along the process and help with repetability. 
import matplotlib.pyplot as plt
from scr import dataLoader as DL
from torch.utils.tensorboard import SummaryWriter
from scr import util as U
import logging
trainLogger = logging.getLogger(__name__)
plt.style.use('fivethirtyeight')

class models_trainer(object):
    def __init__(self, model, loss_fn, optimizer, metric, init_func = None, *params, **kwargs):
        self.model = model
        if init_func is not None:
            self.init_all(self.model, init_func, *params, **kwargs) 
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metric_fn = metric
        self.device = U.checkDevice()
        print(f" Active device :  {self.device}")
        print(f"Actual metric function {self.metric_fn}")
        self.model.to(self.device)
        self.writer = None
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = [] 
        self.test_losses = []
        self.test_metrics = [] 
        self.total_epochs = 0
        
    def set_loaders(self, 
                    train_loader: DL.customDataloader, 
                    val_loader:DL.customDataloader = None,
                    test_loader:DL.customDataloader = None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def getModel(self):
        return self.model   
    
    def _make_train_step_fn(self):
        def perform_train_step_fn(x, y):
            self.model.train()
            yhat = self.model(x)
            loss = self.loss_fn(yhat, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            item  = loss.item()
            # print(f"Training loss {item}")
            return item
        return perform_train_step_fn
    
    def _make_val_step_fn(self):
        def perform_val_step_fn(x, y):
            self.model.eval()
            yhat = self.model(x)
            loss = self.loss_fn(yhat, y)
            item  = loss.item()
            #print(f"Validation loss {item}")
            return item
        return perform_val_step_fn
            
    def _computeLossMeanPerMiniBatch(self, validation=False):
        '''
        The mini-batch can be used with both loaders
        The argument `validation`defines which loader and 
        corresponding step function is going to be used
        '''
        mini_batch_losses = []
        if validation:
            data_loader = self.val_loader
            step_fn = self._make_val_step_fn()
        else:
            data_loader = self.train_loader
            step_fn = self._make_train_step_fn()
        if data_loader is None:
            return None
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device) #.float()   ## Add .float() to avoid type conflict
            mini_batch_loss = step_fn(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)
        loss = np.mean(mini_batch_losses)
        return loss
    
    def _computeLossMeanTestSet(self):
        '''
        The mini-batch can be used with both loaders
        The argument `validation`defines which loader and 
        corresponding step function is going to be used
        '''
        mini_batch_losses = []
        step_fn = self._make_val_step_fn()
        dataLoader = self.test_loader
        for x_batch, y_batch in dataLoader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device).float()   ## Add .float() to avoid type conflict
            mini_batch_loss = step_fn(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)
        return np.mean(mini_batch_losses)
    
    def _computeMetricMiniBatch(self, data_loader)-> float:
        '''
        Return de mean per batch of the metric in self.metric  
        '''
        miniBathcMetric = []
        for x_batch, y_batch in data_loader:
            metric=[]
            for x,y in zip(x_batch, y_batch):
                yHat = self.predict2(x)
                y_item= y.detach().cpu().numpy() if torch.is_tensor(y) else y.squeeze().detach().cpu()
                metric.append(self.metric_fn(yHat, y_item))
            ItemMetric = np.mean(metric)
            miniBathcMetric.append(ItemMetric)
        return round(np.mean(miniBathcMetric),7)

    def train(self, n_epochs, seed=42):
        self.set_seed(seed)
        for epoch in range(n_epochs):
            print(f"Epoch {epoch} ........ -> of {n_epochs}")
            with U.timeit():
                self.total_epochs += 1
                loss = self._computeLossMeanPerMiniBatch(validation=False)
                self.train_losses.append(loss)
                with torch.no_grad():
                    # Performs evaluation using mini-batches
                    val_loss = self._computeLossMeanPerMiniBatch(validation=True)
                    self.val_losses.append(val_loss)
                    valMetric = self._computeMetricMiniBatch(self.val_loader)
                    self.val_metrics.append(valMetric)
                    if self.test_loader is not None:
                        test_loss = self._computeLossMeanTestSet()
                        self.test_losses.append(test_loss)
                        testMetric = self._computeMetricMiniBatch(self.test_loader)
                        self.test_metrics.append(testMetric)
                    
            trainLogger.info(f"Epoch_{epoch}: trainLoss = {loss}: valLoss = {val_loss}: valMetric = {valMetric}; test_loss = {test_loss}; testMetric = {testMetric}")
        
        self.call_logs(n_epochs)
       
        return self.train_losses, self.val_losses, self.test_losses 

    def save_checkpoint(self, filename):
        # Builds dictionary with all elements for resuming training
        checkpoint = {'epoch': self.total_epochs,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'loss': self.train_losses,
                      'val_loss': self.val_losses,
                      'validation_metric': self.val_metrics}
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        # Loads dictionary
        checkpoint = torch.load(filename)
        # Restore state for model and optimizer
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_epochs = checkpoint['epoch']
        self.train_losses = checkpoint['loss']
        self.val_losses = checkpoint['val_loss']
        self.model.train() # always use TRAIN for resuming training   

    def predict(self, x)-> np.array:
        '''
        @Return: A 0-1 mask as np.array. 
        '''
        self.model.eval()
        y_hat_tensor = self.model(torch.unsqueeze(x,0).to(self.device))
        self.model.train()
        # Detaches it, brings it to CPU and back to Numpy
        mask = torch.round(torch.sigmoid(y_hat_tensor)).to(torch.int32)
        return mask.detach().cpu().numpy()

    def predict2(self, x)-> np.array:
        '''
        @Return: A 0-1 mask as np.array. 
        '''
        y_hat_tensor = self.model(torch.unsqueeze(x,0).to(self.device))
        # Detaches it, brings it to CPU and back to Numpy
        mask = torch.round(y_hat_tensor).detach().cpu().numpy()
        return mask
    
    def plot_losses(self):
        print("I'm into plot Losess function")
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.train_losses, label='Training Loss', c='b')
        plt.plot(self.val_losses, label='Validation Loss', c='r')
        plt.plot(self.test_losses, label='Test Loss', c='g')
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        fig.show()
           
    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False    
        torch.manual_seed(seed)
        np.random.seed(seed)
        
    def to(self, device):
        '''
        This method allows the user to specify a different device
        It sets the corresponding attribute (to be used later in
        the mini-batches) and sends the model to the device
        '''
        try:
            self.device = device
            self.model.to(self.device)
        except RuntimeError:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Couldn't send it to {device}, sending it to {self.device} instead.")
            self.model.to(self.device)
    
    def set_tensorboard(self, name, folder='runs'):
        # This method allows the user to define a SummaryWriter to interface with TensorBoard
        suffix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.writer = SummaryWriter(f'{folder}/{name}_{suffix}')

    def init_all(self, model, init_func, *params, **kwargs):
        '''
        Methode to initialize the model's parameters, according a function <init_func>,
            and the necessary arguments <*params, **kwargs>. 
        Change the type of <p> in <if type(p) == torch.nn.Conv2d:> for a different behavior. 
        '''
        for p in model.parameters():
            if type(p) == (torch.nn.Conv2d or torch.nn.Conv1d) :
                init_func(p, *params, **kwargs)
    
    def call_logs(self, n_epochs):
        '''
        Add logs as needed. 
        '''
        trainLogger.info(f"Train losses after {n_epochs} epochs : {self.train_losses}")
        trainLogger.info(f"Validation losses after {n_epochs} epochs : {self.val_losses}")
        trainLogger.info(f"Validation metrics in {n_epochs} epochs : {self.val_metrics}")
        trainLogger.info(f"Test losses after {n_epochs} epochs : {self.test_losses}")
        trainLogger.info(f"Test metrics in {n_epochs} epochs : {self.test_metrics}")
        

    