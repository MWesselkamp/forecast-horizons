import torch
import torch.cuda
import torch.nn as nn
import numpy as np
import zarr

import pytorch_lightning as L
import torchmetrics

from torch import tensor

from misc.helpers import r2_score_multi
from pytorch_lightning.utilities import grad_norm


class MLPregressor(L.LightningModule):

    """
    See also: https://github.com/da-ewanp/ai-land/blob/main/ai_land/model.py
    """
    
    def __init__(self, input_clim_dim,
                 input_met_dim,
                 input_state_dim,
                 hidden_dim,
                 output_dim,
                 output_diag_dim,
                 batch_size,
                 learning_rate,
                 lookback,
                 rollout,
                 dropout, 
                 weight_decay,
                 device,
                 loss = 'mse', 
                 activation = nn.ReLU(), 
                 targets = ['swvl1',
                         'swvl2',
                         'swvl3',
                         'stl1',
                         'stl2',
                         'stl3',
                         'snowc',
                         ],
                 db_path = ''):

        super().__init__()
        
        self.my_device = device
        
        ds = zarr.open(db_path)
        fistdiff_idx = [list(ds["variable"]).index(x) for x in targets]
        self.ds_data_std = tensor(ds.data_stdevs[fistdiff_idx]).to(device = self.my_device)
        self.ds_mean = tensor(ds.data_1stdiff_means[fistdiff_idx]).to(device = self.my_device)/self.ds_data_std
        self.ds_std = tensor(ds.data_1stdiff_stdevs[fistdiff_idx]).to(device = self.my_device)/self.ds_data_std
        
        self.save_hyperparameters(ignore=['criterion', 'activation'])
        
        self.input_clim_dim = input_clim_dim
        self.input_met_dim = input_met_dim
        self.input_state_dim = input_state_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_diag_dim = output_diag_dim
        self.batch_size = batch_size
        self.num_layers = len(hidden_dim)
        self.activation = activation
        self.learning_rate = learning_rate
        self.output_dim = output_dim
        self.lookback = lookback
        self.rollout = rollout
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.criterion = nn.MSELoss() if loss == 'mse' else nn.SmoothL1Loss()
        self.targ_lst = targets

        self.val_acc = torchmetrics.MeanAbsoluteError()
        self.test_acc = torchmetrics.MeanAbsoluteError()
        
        self.network = self.dense()  

        print("Device: ", self.device)

        #self.targ_idx_full = np.array([24, 25, 26, 27, 28, 29, 30])
        
    def transform(self, x, mean, std):
        x_norm = (x - mean) / (std + 1e-5)
        return x_norm

    def dense(self):
        layers = nn.Sequential()
        layers.add_module(f'input', nn.Linear(self.input_clim_dim+self.input_met_dim+self.input_state_dim, self.hidden_dim[0]))
        layers.add_module(f'activation0', self.activation)
        layers.add_module(f'dropout0', nn.Dropout(self.dropout))
        if self.num_layers > 1:
            for i in range(self.num_layers-1):
                layers.add_module(f'hidden{i}', nn.Linear(self.hidden_dim[i], self.hidden_dim[i + 1]))
                layers.add_module(f'activation{i}', self.activation)
                layers.add_module(f'dropout{i}', nn.Dropout(self.dropout))
        layers.add_module('output', nn.Linear(self.hidden_dim[-1], self.output_dim))
        
        return layers

    def forward(self, x_clim, x_met, x_state):

        x = torch.cat((x_clim, x_met, x_state), dim = -1)
        prediction = self.network(x)

        return prediction

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def training_step(self, train_batch, batch_idx):

        x_clim, x_met, x_state, y = train_batch
        
        logits = self.forward(x_clim, x_met, x_state)
        loss = self.criterion(self.transform(logits, self.ds_mean, self.ds_std),
                             self.transform(y, self.ds_mean, self.ds_std))
        self.log('train_loss_logit', loss, on_step=False, on_epoch=True, sync_dist=True)
        #train_loss = torch.zeros(1, dtype=x.dtype, device=self.device, requires_grad=False)
 
        x_state_rollout = x_state.clone()
        y_rollout = y.clone()
        #y_rollout_diag = y_diag.clone()

        # iterate over lead time.
        for step in range(self.rollout):
            # x = [batch_size=8, lookback (7) + rollout (3) = 10, n_feature = 37]
            x0 = x_state_rollout[:, step, :, :].clone()  # select input with lookback.
            y_hat = self.forward(x_clim[:, step, :, :], x_met[:, step, :, :], x0)  # prediction at rollout step
            y_rollout[:, step, :, :] = y_hat   # overwrite y with prediction.
            if step < self.rollout-1:
                x_state_rollout[:,step + 1,:, :] = x_state_rollout[:, step , :, :].clone() + y_hat # target at next step
            
            
        train_step_loss = self.criterion(self.transform(y_rollout, self.ds_mean, self.ds_std),
                                self.transform(y, self.ds_mean, self.ds_std))
                
        # Compute loss for each variable separately and accumulate
        variable_losses = torch.zeros(len(self.targ_lst), dtype=y.dtype, device=self.device, requires_grad=False)
        for idx, var_idx in enumerate(self.targ_lst):
            var_loss = self.criterion(self.transform(y_rollout, self.ds_mean, self.ds_std)[:, :, :, idx],
                                  self.transform(y, self.ds_mean, self.ds_std)[:, :, :, idx])
            variable_losses[idx] = var_loss

        #variable_losses /= self.rollout
        train_step_loss /= self.rollout  # average rollout loss

        self.log('train_loss_step', train_step_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_loss', loss + train_step_loss, on_step=False, on_epoch=True, sync_dist=True)
        for idx, v_loss in enumerate(variable_losses):
            self.log(f'train_loss_var_{idx}', v_loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss + train_step_loss

    def validation_step(self, val_batch, batch_idx):

        x_clim, x_met, x_state, y = val_batch  # IF x with lookback, y without -> len(x) > len(y)
        
        logits = self.forward(x_clim, x_met, x_state)
        loss = self.criterion(self.transform(logits, self.ds_mean, self.ds_std),
                             self.transform(y, self.ds_mean, self.ds_std))
        self.log('val_loss_logit', loss, on_step=False, on_epoch=True, sync_dist=True)
   
        x_state_rollout = x_state.clone()
        y_rollout = y.clone()
        #y_rollout_diag = y_diag.clone()

        # iterate over lead time.
        for step in range(self.rollout):
            # x = [batch_size=8, lookback (7) + rollout (3) = 10, n_feature = 37]
            x0 = x_state_rollout[:, step, :, :].clone()  # select input with lookback.
            y_hat = self.forward(x_clim[:, step, :, :], x_met[:, step, :, :], x0)  # prediction at rollout step
            y_rollout[:, step, :, :] = y_hat   # overwrite y with prediction.
            if step < self.rollout-1:
                x_state_rollout[:,step + 1,:, :] = x_state_rollout[:, step , :, :].clone() + y_hat # target at next step
            
            
        val_step_loss = self.criterion(self.transform(y_rollout, self.ds_mean, self.ds_std),
                                self.transform(y, self.ds_mean, self.ds_std))
        val_step_loss /= self.rollout # average rollout loss
                
        # Compute loss for each variable separately and accumulate
        variable_losses = torch.zeros(len(self.targ_lst), dtype=y.dtype, device=self.device, requires_grad=False)
        for idx, var_idx in enumerate(self.targ_lst):
            var_loss = self.criterion(self.transform(y_rollout, self.ds_mean, self.ds_std)[:, :, :, idx],
                                  self.transform(y, self.ds_mean, self.ds_std)[:, :, :, idx])
            variable_losses[idx] = var_loss

        print("Loss:", val_step_loss)

        r2 = r2_score_multi(self.transform(y_rollout, self.ds_mean, self.ds_std).cpu(),
                            self.transform(y, self.ds_mean, self.ds_std).cpu())

        self.log('val_loss_step', val_step_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_loss', loss + val_step_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_r**2', r2, on_step=False, on_epoch=True, sync_dist=True)
        for idx, v_loss in enumerate(variable_losses):
            self.log(f'val_loss_var_{idx}', v_loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss + val_step_loss

    def forecast(self, x_clim, x_met, x_state):
        
        preds = states.clone().to(self.device)
        lead_time = x_state.shape[0]
        
        self.eval()
        
        for t in range(lead_time - 1):
            
            if t % 1000 == 0:
                print(f"on step {t}...")
                
            with torch.no_grad():
                logits = self.forward(torch.cat((x_clim, x_met[[t]], preds[[t]]), axis=-1))
                preds[t+1, ...] = preds[t, ...] + logits
                
        return preds


class LSTMregressor(L.LightningModule):

    def __init__(self, 
                 input_clim_dim ,
                 input_met_dim,
                 input_state_dim,
                 lookback, 
                 lead_time, 
                 device,
                 batch_size=6,
                 dropout = 0.0, 
                 weight_decay = 0.001, 
                 learning_rate = 1e-3, 
                 num_layers_en = 2, 
                 num_layers_de = 2, 
                 hidden_size_en = 64,
                 hidden_size_de = 64,
                 loss = 'mse', 
                 use_dlogits = True,
                 fields_embedding = False,
                 embed_cell_states = True,
                 transfer = "linear",
                 transform = 'zscoring',
                 targets = ['swvl1',
                         'swvl2',
                         'swvl3',
                         'stl1',
                         'stl2',
                         'stl3',
                         'snowc',
                         ],
                 db_path=''):

        super().__init__() 

        self.my_device = device
        
        ds = zarr.open(db_path)
        fistdiff_idx = [list(ds["variable"]).index(x) for x in targets]
        self.ds_data_std = tensor(ds.data_stdevs[fistdiff_idx]).to(device = self.my_device)
        self.ds_data_max = tensor(ds.data_maxs[fistdiff_idx]).to(device = self.my_device)
        if transform == 'zscoring':
            self.ds_mean = tensor(ds.data_1stdiff_means[fistdiff_idx]).to(device = self.my_device)/self.ds_data_std
            self.ds_std = tensor(ds.data_1stdiff_stdevs[fistdiff_idx]).to(device = self.my_device)/self.ds_data_std
        elif transform == 'max':
            self.ds_mean = tensor(ds.data_1stdiff_means[fistdiff_idx]).to(device = self.my_device)/self.ds_data_max
            self.ds_std = tensor(ds.data_1stdiff_stdevs[fistdiff_idx]).to(device = self.my_device)/self.ds_data_max

        self.save_hyperparameters(ignore=['criterion', 'activation', 'db_path','device'])
        self.input_clim_dim = input_clim_dim
        self.input_met_dim = input_met_dim
        self.input_state_dim = input_state_dim
        self.lookback = lookback
        self.batch_size = batch_size
        self.lead_time = lead_time
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.activation = nn.ReLU()
        self.transfer_activation = nn.Tanh()
        self.output_activation = nn.Identity() if transform == 'zscoring' else nn.ReLU()
        self.num_layers_en = num_layers_en
        self.num_layers_de = num_layers_de
        self.hidden_size_en = hidden_size_en
        self.hidden_size_de = hidden_size_de
        self.use_dlogits = use_dlogits
        self.embed_cell_states = embed_cell_states
        self.transform = self.z_transform if transform == 'zscoring' else self.z_transform
    
        if loss == 'mse':
            self.criterion = nn.MSELoss()
        elif loss == 'smoothl1':
            self.criterion = nn.SmoothL1Loss()
        else:
            print("Dont know loss function")
            
        self.second_criterion = nn.MSELoss()
        self.val_acc = torchmetrics.MeanAbsoluteError()
        self.test_acc = torchmetrics.MeanAbsoluteError()

        self.targ_lst = targets

        self.fields_embedding = self.dense_fields_enconding() if fields_embedding else nn.Identity()
        self.hidden_encoder = nn.Linear(in_features=self.input_state_dim*self.lookback, out_features=self.hidden_size_en)
        self.cell_encoder = nn.Linear(in_features=self.input_state_dim*self.lookback, out_features=self.hidden_size_en)
        self.hidden_activation = self.activation
        self.cell_activation = self.activation
        
        self.lstm_encoder = nn.LSTM(input_size=(self.input_clim_dim + self.input_met_dim),
                                    hidden_size=self.hidden_size_en,
                                    num_layers=self.num_layers_en,
                                    dropout=self.dropout,
                                    batch_first=True)
        
        self.hidden_adapter = self.linear_transfer() if transfer == "linear" else self.dense_transfer() 
        self.cell_adapter = self.linear_transfer() 
        self.hidden2_activation = self.transfer_activation
        self.cell2_activation = self.transfer_activation
        
        self.lstm_decoder = nn.LSTM(input_size=(self.input_clim_dim + self.input_met_dim),
                                    hidden_size=self.hidden_size_de,
                                    num_layers=self.num_layers_de,
                                    dropout=self.dropout,
                                    batch_first=True)
        
        self.mlp_decoder = nn.Linear(in_features=self.hidden_size_de, out_features=self.input_state_dim)

    def dense_fields_encoding(self):
        
        layers = nn.Sequential()
        layers.add_module('ce_fc0', nn.Linear(self.input_clim_dim, self.hidden_emb[0]))
        layers.add_module('ce_activation0', self.activation)
        layers.add_module('ce_dropout0', nn.Dropout(self.dropout))
        layers.add_module('ce_output', nn.Linear(self.hidden_emb[0], self.input_clim_dim))
        
        return layers
        
    def initialize_hidden(self, y_h):

        h_en = self.hidden_encoder(y_h)
        h_en = self.hidden_activation(h_en)
        h_en = h_en.unsqueeze(dim=0).repeat(self.num_layers_en, 1, 1).contiguous().to(device=self.my_device)

        return h_en

    def initialize_cell(self, y_h):

        if self.embed_cell_states:
            c_en = self.cell_encoder(y_h)
            c_en = self.cell_activation(c_en)
            c_en = c_en.unsqueeze(dim=0).repeat(self.num_layers_en, 1, 1).contiguous().to(device=self.my_device)
        else:
            c_en = torch.zeros(self.h_en.size(), dtype=h_en.dtype).contiguous().to(device=self.my_device)
            
        return c_en

    def transfer_hidden(self, hn):
        
        h_de = self.hidden2_activation(self.hidden_adapter(hn[-1,:,:].unsqueeze(dim=0)))
        h_de = h_de.repeat(self.num_layers_de, 1, 1).contiguous().to(device=self.my_device)
        
        return h_de

    def transfer_cell(self, cn):

        c_de = self.cell2_activation(self.cell_adapter(cn[-1,:,:].unsqueeze(dim=0))) 
        c_de = c_de.repeat(self.num_layers_de, 1, 1).contiguous().to(device=self.my_device)
        
        return c_de

    def linear_transfer(self):
        return nn.Linear(in_features=self.hidden_size_en, out_features=self.hidden_size_de)

    def dense_transfer(self):
        dense = nn.Sequential()
        dense.add_module('ht_fc0', nn.Linear(self.hidden_size_en, self.hidden_size_en))
        dense.add_module('ht_activation0', nn.Tanh())
        dense.add_module('ht_fc1', nn.Linear(self.hidden_size_en, self.hidden_size_en))
        dense.add_module('ht_activation1', nn.Tanh())
        dense.add_module('ht_output', nn.Linear(self.hidden_size_en, self.hidden_size_de))
        
        return dense
        
    def forward(self, X_static_h, X_static_f, X_met_h, X_met_f, y_h):

        X_static_h = self.fields_embedding(X_static_h)
        X_static_f = self.fields_embedding(X_static_f)
        
        X_h = torch.cat((X_static_h, X_met_h), dim = -1)
        X_f = torch.cat((X_static_f, X_met_f), dim = -1)
        
        # size of spatial subsample for output reshape.
        spatial_points = X_f.size(2)
        
        # Swap temporal and spatial dimension
        X_h = X_h.permute(0, 2, 1, 3)
        X_f = X_f.permute(0, 2, 1, 3)
        y_h = y_h.permute(0, 2, 1, 3)

        # Merge spatial dimension into batch
        if not X_h.is_contiguous():
            X_h = X_h.contiguous()
        if not X_f.is_contiguous():
            X_f = X_f.contiguous()
        if not y_h.is_contiguous():
            y_h = y_h.contiguous()
            
        X_h = X_h.view(X_h.size(0)*X_h.size(1), X_h.size(2), X_h.size(3))
        X_f = X_f.view(X_f.size(0)*X_f.size(1), X_f.size(2), X_f.size(3))
        y_h = y_h.view(y_h.size(0)*y_h.size(1), y_h.size(2), y_h.size(3))
    
        # Flatten historic states for hidden embedding
        y_h = y_h.view(y_h.shape[0], -1)
        
        # Transfer to hidden and cell states
        self.h_en = self.initialize_hidden(y_h)
        self.c_en = self.initialize_cell(y_h)
                
        # LSTM encoder network
        out , (hn, cn) = self.lstm_encoder(X_h, (self.h_en, self.c_en))
       
        # Transfer hidden and cell to deencoder network
        h_de = self.transfer_hidden(hn)
        c_de = self.transfer_cell(cn)

        out, (h_de, c_de) = self.lstm_decoder(X_f, (h_de, c_de))

        # Forecast head projects output
        out = self.mlp_decoder(out)
        
        print("out shape:", out.shape)
        
        # Reshape and permute out back into original
        out = out.view(self.batch_size, spatial_points, X_f.size(1), self.input_state_dim)
        out = out.permute(0, 2, 1, 3)
        
        out = self.output_activation(out)
        
        return out

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss_total'}
    
    def z_transform(self, x, **kwargs):
        x_norm = (x - kwargs['means']) / (kwargs['stds'] + 1e-5)
        return x_norm
    
    #def max_transform(self, x, **kwargs):
    #    x_norm = (x / kwargs['maxs']) 
    #    return x_norm
    
    def training_step(self, train_batch, batch_idx):

        X_static_h, X_static_f,X_met_h, X_met_f, y_h, y_f, y_inc_h, y_inc_f = train_batch

        logits = self.forward(X_static_h, X_static_f, X_met_h, X_met_f, y_h)
        logit_loss = self.criterion(logits,y_f)
        print("Train logit loss:", logit_loss)
        
        if self.use_dlogits:
            d_logits = logits[:,1:,:,:] - logits[:,:-1,:,:]
            d_y_f = y_inc_f[:,:-1,:,:]
            # Removed feature transform for direct states prediction.
            d_logit_loss = self.criterion(
                self.transform(d_logits, means = self.ds_mean, stds = self.ds_std), # maxs = self.ds_data_max
                self.transform(d_y_f, means = self.ds_mean, stds = self.ds_std))
            print("Train d_logit loss:", d_logit_loss)

        # Compute loss for each variable separately and accumulate
        variable_losses = torch.zeros(len(self.targ_lst), dtype=y_f.dtype, device=self.device, requires_grad=False)
        variable_rmse = torch.zeros(len(self.targ_lst), dtype=y_f.dtype, device=self.device, requires_grad=False)
        for idx, var_idx in enumerate(self.targ_lst):
            var_loss = self.criterion(logits[..., idx],y_f[..., idx])
            var_rmse = torch.sqrt(self.second_criterion(logits[..., idx], y_f[..., idx]))
            variable_losses[idx] = var_loss
            variable_rmse[idx] = var_rmse
        
        rmse = torch.sqrt(self.second_criterion(logits, y_f))
        #sum up logits and dlogits loss if computed
        total_loss = logit_loss + d_logit_loss if self.use_dlogits else logit_loss
                                                 
        self.log('train_loss_logit', logit_loss , on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_loss_total', total_loss , on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_rmse', rmse, on_step=False, on_epoch=True, sync_dist=True)
        for idx, v_loss in enumerate(variable_losses):
            self.log(f'train_loss_var_{idx}', v_loss, on_step=False, on_epoch=True, sync_dist=True)

        del X_static_h, X_static_f, X_met_h, X_met_f, y_h, y_f
        torch.cuda.empty_cache()
        
        return total_loss

    def validation_step(self, val_batch, batch_idx):

        X_static_h, X_static_f,X_met_h, X_met_f, y_h, y_f, y_inc_h, y_inc_f = val_batch

        logits = self.forward(X_static_h, X_static_f, X_met_h, X_met_f, y_h)
        logit_loss = self.criterion(logits,y_f)
        print("Val logit loss:", logit_loss)
        
        if self.use_dlogits:
            d_logits = logits[:,1:,:,:] - logits[:,:-1,:,:]
            d_y_f = y_inc_f[:,:-1,:,:]
            # Removed feature transform for direct states prediction.
            d_logit_loss = self.criterion(
                self.transform(d_logits, means = self.ds_mean, stds = self.ds_std),
                self.transform(d_y_f, means = self.ds_mean, stds = self.ds_std))
            print("Val d_logit loss:", d_logit_loss)
   
        # Compute loss for each variable separately and accumulate
        variable_losses = torch.zeros(len(self.targ_lst), dtype=y_f.dtype, device=self.device, requires_grad=False)
        variable_rmse = torch.zeros(len(self.targ_lst), dtype=y_f.dtype, device=self.device, requires_grad=False)
        for idx, var_idx in enumerate(self.targ_lst):
            var_loss = self.criterion(logits[..., idx],y_f[..., idx])
            var_rmse = torch.sqrt(self.second_criterion(logits[..., idx], y_f[..., idx]))
            variable_losses[idx] = var_loss
            variable_rmse[idx] = var_rmse

        #r2 = r2_score_multi(logits.cpu(), y_f.cpu())
        rmse = torch.sqrt(self.second_criterion(logits, y_f))
           
        total_loss = logit_loss + d_logit_loss if self.use_dlogits else logit_loss
                                                 
        self.log('val_loss_logit', logit_loss , on_step=False, on_epoch=True, sync_dist=True) 
        self.log('val_loss_total', total_loss , on_step=False, on_epoch=True, sync_dist=True)
        #self.log('val_r**2', r2, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_rmse', rmse, on_step=False, on_epoch=True, sync_dist=True)
        for idx, v_loss in enumerate(variable_losses):
            self.log(f'val_loss_var_{idx}', v_loss, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f'val_rmse_var_{idx}', var_rmse, on_step=False, on_epoch=True, sync_dist=True)

        del X_static_h, X_static_f, X_met_h, X_met_f, y_h, y_f
        torch.cuda.empty_cache()
        
        return total_loss
        
    def forecast(self, clim_feats, met_feats, states):
        
        lead_time = met_feats.shape[0] - self.lookback
        
        self.eval()
        
        with torch.no_grad():
    
            y_h = states[:self.lookback, ...].unsqueeze(0)
            X_static = clim_feats.expand(self.lookback + lead_time, -1, -1)
            X_static_h = X_static[:self.lookback, ...].unsqueeze(0)
            X_met_h = met_feats[:self.lookback, ...].unsqueeze(0)

            X_static_f = X_static[self.lookback:(self.lookback + lead_time), ...].unsqueeze(0)
            X_met_f = met_feats[self.lookback:(self.lookback + lead_time), ...].unsqueeze(0)

            preds = self.forward(X_static_h, X_static_f, X_met_h, X_met_f, y_h) 
            
        return preds