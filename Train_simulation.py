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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        if dataset_name == 'metavision' : 
            self.N = 4
        else:
            self.N =5
        self.network = ClientNetwork(self.N, d_latent, h, dropout, seq_len, cap_in_dim , lr).to(device)
        chartevents_path = "/content/drive/MyDrive/split_learning/CHARTEVENTS.csv"
        #chartevents_path = "./CHARTEVENTS.csv"

        df_chartevents = pd.read_csv(chartevents_path)
        self.data = data_preparing(df_chartevents ,dataset_name , seq_len , test_size , target  ,batch_size)
        self.transmittion = Transmitter(cap_in_dim , device)
        self.batch_size = batch_size 
        self.L1Loss= nn.L1Loss()
        self.loss_fn = nn.MSELoss()
    def fit(self , epochs ): 
        history = {
            'loss_train' : [] , 
            'loss_test'  : []
        }
        for epoch in range(epochs) : 
            self.train_one_epoch()
            loss_train , loss_test = self.evaluate_one_epoch()
            print(f'''
            [epoch {epoch} / {epochs}    train_loss = {loss_train}    test_loss = {loss_test}]
            ''')
            loss_test =loss_test.item()
            loss_train =loss_train.item()
            history['loss_test'].append(loss_test)
            history['loss_train'].append(loss_train)

        return history

    def train_one_epoch(self) :
        for x , l , mask in self.data.train_loader :  
            v, loss_client  = self.network(x.to(self.device)  , mask)
            grad = self.transmittion.send_data(v , l , status='train')
            self.network.train_one_batch(loss_client , v, grad.clone())
        return True
    def evaluate_one_epoch(self)  :
        loss_train = 0 
        number = 0 
        for x , l , mask in self.data.train_loader :  
            l = l.to(self.device)
            v  = self.network(x.to(self.device), mask , train=False)
            prediction = self.transmittion.send_data(v , l , status='test')
            loss_train +=x.shape[0] * self.loss_fn(prediction.to(self.device) , l )
            number += x.shape[0]
        loss_train = loss_train/number
        loss_test = 0 
        number = 0 
        for x , l , mask in self.data.test_loader :  
            l = l.to(self.device)
            v   = self.network(x.to(self.device) , mask  ,train=False)
            prediction = self.transmittion.send_data(v, l , status='test')
            loss_test +=x.shape[0] * self.loss_fn(prediction.to(self.device) , l )
            number += x.shape[0]
        loss_test = loss_test/number 
        return loss_train , loss_test     
    def get_knowledge(self , CAT_object ) : 
        all_auto_encoders  = CAT_object.network.multi_autoEncoder.auto_encoders
        for i in range(self.N) : 
            l1Loss = [] 
            for auto_endocer in  all_auto_encoders : 
                l1Loss.append(self.compute_autoEnccoder_loss(auto_endocer , i ))
            min_idx = torch.argmin(torch.stack(l1Loss))
            print(f'the feature {i} chooses the autocoder {min_idx}')
            weights = all_auto_encoders[min_idx].state_dict()
            self.network.MultiAutoEncoder.autoEncoders[i].load_state_dict(weights)
            
    def compute_autoEnccoder_loss(self , auto_endocer , i ) : 
        loss =0
        number = 0  
        for x , _ in self.data.load_train : 
            a = x.shape[0]
            inp = x[: , 1 ,: ,i]
            _ , decoder_out  = auto_endocer(inp)
            loss += a * self.L1Loss(inp,decoder_out)
            number += a 
        loss = loss / number
        return loss 






