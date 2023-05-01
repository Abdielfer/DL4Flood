
# %load stepbystep/v0.py
### Fron: Deep_Learning with Pythorch Step By Step. Volume II. Daniel Voigt Godoy
### https://github.com/dvgodoy/PyTorchStepByStep 

import numpy as np
import datetime
import torch
from torch.nn.functional import sigmoid
import matplotlib.pyplot as plt
from scr import dataLoader as DL
from torch.utils.tensorboard import SummaryWriter
from scr import losses

plt.style.use('fivethirtyeight')

class models_trainer(object):
    def __init__(self, model, loss_fn, optimizer, metric):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metric_fn = metric
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.train_loader = None
        self.val_loader = None
        self.writer = None
        self.losses = []
        self.val_losses = []
        self.val_metrics = []  
        self.total_epochs = 0
        self.train_step_fn = self._make_train_step_fn()
        self.val_step_fn = self._make_val_step_fn()

    def set_loaders(self, train_loader, val_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader

    def getModel(self):
        return self.model   
    
    def _make_train_step_fn(self):
        def perform_train_step_fn(x, y):
            self.model.train()
            self.optimizer.zero_grad()
            yhat = self.model(x)
            loss = self.loss_fn(yhat, y)
            loss.backward()
            self.optimizer.step()
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
            
    def _trainMiniBatch(self, validation=False):
        '''
        The mini-batch can be used with both loaders
        The argument `validation`defines which loader and 
        corresponding step function is going to be used
        '''
        if validation:
            data_loader = self.val_loader
            step_fn = self.val_step_fn

        else:
            data_loader = self.train_loader
            step_fn = self.train_step_fn

        if data_loader is None:
            return None
    
        mini_batch_losses = []
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device).float()   ## Add .float() to avoid type conflict
            mini_batch_loss = step_fn(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)

        loss = np.mean(mini_batch_losses)
        return loss
    
    def _computeMetricPerMiniBatch(self, data_loader):
        '''
        Return de mean per batch of the metric in self.metric  
        '''
        miniBathcMetric = []
        for x_batch, y_batch in data_loader:
            metric=[]
            for x,y in zip(x_batch, y_batch):
                yHat = self.model(torch.unsqueeze(x,0))
                y_hat_sigmoid = torch.round(sigmoid(yHat)).to(torch.int32)
                y_item= y.numpy() if torch.is_tensor(y) else y.numpy().squeeze()
                y_hat_item = y_hat_sigmoid.detach().cpu().numpy().squeeze() if torch.is_tensor(y_hat_sigmoid) else y_hat_sigmoid.numpy().squeeze()
                metricPerImage = self.metric_fn(y_hat_item, y_item)
                metric.append(metricPerImage)    
            ItemMetric = round(np.mean(metric),2)
            miniBathcMetric.append(ItemMetric)
        metricMean = round(np.mean(miniBathcMetric),2)
        return metricMean
    
    def _computeMetricMiniBatch(self, data_loader):
        '''
        Return de mean per batch of the metric in self.metric  
        '''
        metrics = {}
        print(f"Actual metric {self.metric_fn}")
        miniBathcMetric = []
        for x_batch, y_batch in data_loader:
            metric=[]
            for x,y in zip(x_batch, y_batch):
                yHat = self.model(torch.unsqueeze(x,0))
                y_hat_item = int(sigmoid(yHat[0]))
                y_item= y.numpy() if torch.is_tensor(y) else y.numpy().squeeze()
                y_hat_item = y_hat_item.detach().numpy().squeeze() if torch.is_tensor(y_hat_item) else y_hat_item.numpy().squeeze()
                metric.append(self.metric_fn(y_hat_item, y_item))
            ItemMetric = np.mean(metric)
            print(f"ItemMetric : {ItemMetric}")
            miniBathcMetric.append(ItemMetric)
        print(f"miniBathcMetric : {miniBathcMetric} ")
        metricMean = np.mean(miniBathcMetric)
        metrics[str(self.metric_fn)] = metricMean
        return metricMean

    def train(self, n_epochs, seed=42):
        self.set_seed(seed)

        for epoch in range(n_epochs):
            print(f"Epoch {epoch} ........ ->")
            self.total_epochs += 1
            loss = self._trainMiniBatch(validation=False)
            print(f"train Loss = {loss}")
            self.losses.append(loss)
            with torch.no_grad():
                # Performs evaluation using mini-batches
                val_loss = self._trainMiniBatch(validation=True)
                print(f"val Loss = {val_loss}")
                self.val_losses.append(val_loss)
                metric = self._computeMetricPerMiniBatch(self.val_loader)
                self.val_metrics.append(metric)
                print(f"Metric(s) per minibatch = {metric}")

            # If a SummaryWriter has been set...
            if self.writer:
                scalars = {'training': loss}
                # metricSaclar = {'validation Metric': metric}
                if val_loss is not None:
                    scalars.update({'validation': val_loss})
                # Records both losses for each epoch under the main tag "loss"
                self.writer.add_scalars(main_tag='loss',
                                        tag_scalar_dict=scalars,
                                        global_step=epoch)
                self.writer.add_scalars(main_tag='metric',
                                        # tag_scalar_dict=metricSaclar,
                                        global_step=epoch)  
        self.plot_losses()
        if self.writer:
            # Closes the writer
            self.writer.close()

    def save_checkpoint(self, filename):
        # Builds dictionary with all elements for resuming training
        checkpoint = {'epoch': self.total_epochs,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'loss': self.losses,
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
        self.losses = checkpoint['loss']
        self.val_losses = checkpoint['val_loss']
        self.model.train() # always use TRAIN for resuming training   

    def predict(self, x):
        '''
        @Return: A 0-1 mask as np.array. 
        '''
        self.model.eval() 
        # Takes aNumpy input and make it a float tensor
        x_tensor = torch.as_tensor(x).float()
        # Send input to device and uses model for prediction
        y_hat_tensor = self.model(x_tensor.to(self.device))
        # Set it back to train mode
        self.model.train()
        # Detaches it, brings it to CPU and back to Numpy
        mask = torch.round(sigmoid(y_hat_tensor)).to(torch.int32)
        return mask.detach().cpu().numpy()

    def plot_losses(self):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.losses, label='Training Loss', c='b')
        plt.plot(self.val_losses, label='Validation Loss', c='r')
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        return fig
   
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

