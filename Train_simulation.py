from client_net import ClientNetwork
from new_dataset import data_preparing
from transmitter_simulation import Transmitter
import torch  
import torch.nn as nn 
import pandas as pd 
import numpy as np 

class CAT(nn.Module) : 
    def __init__(self , seq_len, dataset_name,batch_size ,test_size , target  , d_latent  , h , dropout ,cap_in_dim , lr) -> None:
        super().__init__()
        
        # Select device (GPU if available)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        # Number of input features depends on dataset
        if dataset_name == 'metavision' : 
            self.N = 4
        else:
            self.N =5
        
        # Initialising the client network
        self.network = ClientNetwork(self.N, d_latent, h, dropout, seq_len, cap_in_dim , lr).to(device)
        
        # Path to dataset file
        chartevents_path = "/content/drive/MyDrive/split_learning/CHARTEVENTS.csv"
        #chartevents_path = "./CHARTEVENTS.csv"

        # Read dataset
        df_chartevents = pd.read_csv(chartevents_path)

        # Prepare dataset loaders
        self.data = data_preparing(df_chartevents ,dataset_name , seq_len , test_size , target  ,batch_size)
        
        # Communication/transmission module
        self.transmittion = Transmitter(cap_in_dim , device)
        
        self.batch_size = batch_size 
        
        # Loss functions
        self.L1Loss= nn.L1Loss()
        self.loss_fn = nn.MSELoss()

    def fit(self , epochs ): 
        # Store training/testing loss across epochs
        history = {
            'loss_train' : [] , 
            'loss_test'  : []
        }
        
        for epoch in range(epochs) : 
            # Run one training epoch
            self.train_one_epoch()
            
            # Evaluate both training and test losses
            loss_train , loss_test = self.evaluate_one_epoch()
            
            print(f'''
            [epoch {epoch} / {epochs}    train_loss = {loss_train}    test_loss = {loss_test}]
            ''')
            
            # Convert tensors to values
            loss_test = loss_test.item()
            loss_train = loss_train.item()
            
            # Store results
            history['loss_test'].append(loss_test)
            history['loss_train'].append(loss_train)

        return history

    def train_one_epoch(self) :
        # Iterate through training batches
        for x , l , mask in self.data.train_loader :  
            # Send input through client model
            v, loss_client  = self.network(x.to(self.device)  , mask)
            
            # Transmit intermediate representation
            grad = self.transmittion.send_data(v , l , status='train')
            
            # Update client model using gradients
            self.network.train_one_batch(loss_client , v, grad.clone())
        return True

    def evaluate_one_epoch(self)  :
        # Compute training loss in evaluation mode
        loss_train = 0 
        number = 0 
        
        for x , l , mask in self.data.train_loader :  
            l = l.to(self.device)
            
            # Forward pass without training updates
            v  = self.network(x.to(self.device), mask , train=False)
            prediction = self.transmittion.send_data(v , l , status='test')
            
            # Accumulate weighted loss
            loss_train +=x.shape[0] * self.loss_fn(prediction.to(self.device) , l )
            number += x.shape[0]
        
        # Mean training loss
        loss_train = loss_train/number
        
        # Compute test loss
        loss_test = 0 
        number = 0 
        
        for x , l , mask in self.data.test_loader :  
            l = l.to(self.device)
            v   = self.network(x.to(self.device) , mask  ,train=False)
            prediction = self.transmittion.send_data(v, l , status='test')
            
            # Accumulate weighted loss
            loss_test +=x.shape[0] * self.loss_fn(prediction.to(self.device) , l )
            number += x.shape[0]
        
        # Mean test loss
        loss_test = loss_test/number 
        
        return loss_train , loss_test     

    def get_knowledge(self , CAT_object ) : 
        # Access all autoencoders from another CAT instance
        all_auto_encoders  = CAT_object.network.multi_autoEncoder.auto_encoders
        
        # For each feature choose best autoencoder
        for i in range(self.N) : 
            l1Loss = [] 
            
            # Evaluate each autoencoder for feature i
            for auto_endocer in  all_auto_encoders : 
                l1Loss.append(self.compute_autoEnccoder_loss(auto_endocer , i ))
            
            # Select minimum-loss autoencoder
            min_idx = torch.argmin(torch.stack(l1Loss))
            print(f'the feature {i} chooses the autocoder {min_idx}')
            
            # Load its weights to corresponding position in current model
            weights = all_auto_encoders[min_idx].state_dict()
            self.network.multi_autoEncoder.auto_encoders[i].load_state_dict(weights)
            
    def compute_autoEnccoder_loss(self, auto_encoder, i):
        # Compute reconstruction loss of one autoencoder on feature i
        total_loss = 0.0
        total_samples = 0  

        for x, _, mask in self.data.train_loader:
            x = x.to(self.device)
            mask = mask.to(self.device)
            
            b, seq_len, _ = x.shape
            
            # Select feature i (flatten for AE input)
            inp = x[:, :, i].reshape(-1, 1)  # (b*seq_len, 1)
            
            # Forward pass through autoencoder
            _, decoder_out = auto_encoder(inp)
            
            # Reshape decoder output
            decoder_out = decoder_out.reshape(b, seq_len)
            
            # Apply mask to exclude padded positions
            masked_inp = inp.reshape(b, seq_len) * mask
            masked_out = decoder_out * mask
            
            # Compute L1 loss
            batch_loss = self.L1Loss(masked_inp, masked_out)
            
            total_loss += batch_loss.item() * b
            total_samples += b
        
        # Average loss over samples
        mean_loss = total_loss / total_samples
        return torch.tensor(mean_loss)
