import numpy as np
import os
import torch
from dataloader.dataset import BasicDataset, Collator
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from config import alphabet
from optim.loss import LabelSmoothingLoss
from model.seq2seq import Seq2Seq
from torch.utils.data import DataLoader, random_split
from utils import batch_to_device, compute_accuracy
from tool.translate import translate
import time
from tool.logger import Logger
from model.vocab import Vocab


class Trainer(object):
    def __init__(self, data_file='./Data/train_data.txt', batch_size=10, num_iters=200000, weight_path='./weights/auto_correct.pth'):
        self.batch_size = batch_size
        self.num_iters = num_iters
        self.device = ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.weight_path = weight_path
        self.valid_every = 3000
        self.print_every = 200
        self.lr = 0.0001
        self.logger = Logger('./train.log')
        
        # create vocab model
        self.vocab = Vocab(alphabet)
        
        # create model
        self.model = Seq2Seq(len(alphabet), encoder_hidden=256, decoder_hidden=256)
        self.model = self.model.to(device=self.device)
        
        # load pretrain weights
        self.load_weights(weight_path)

        # create loss function
        self.criterion = LabelSmoothingLoss(len(alphabet), 0).cuda(0)
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-09)
        self.scheduler = OneCycleLR(self.optimizer, max_lr=self.lr, total_steps=self.num_iters, pct_start=0.1)

        # create dataset
        dataset = BasicDataset(label_file=data_file)

        # split train and val dataset
        split_ratio = 0.99
        n_train = int(len(dataset) * split_ratio)
        n_val = len(dataset) - n_train
        train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
        print('The number of train data: ', n_train)
        print('The number of val data: ', n_val)

        # create train and val dataloader
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=Collator(), shuffle=True, num_workers=8, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=Collator(), shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
        

    def train(self):
        total_loss = 0
        best_acc = 0
        global_step = 0
        data_iter = iter(self.train_loader)

        for _ in range(self.num_iters):
            self.model.train()
            
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)
            
            global_step += 1
            start = time.time()
            loss = self.train_step(batch)
            end = time.time()
            total_loss += loss

            if global_step % self.print_every == 0:
                info = 'step: {:06d}, train_loss: {:.4f}, gpu_time: {}'.format(global_step, total_loss / self.print_every, end - start)
                print(info)
                self.logger.log(info)
                total_loss = 0
                

            if global_step % self.valid_every == 0:
                # validate 
                val_loss = self.validate()
                acc_full_seq, acc_per_char = self.precision()
               
                if acc_full_seq > best_acc:
                    best_acc = acc_full_seq
                    torch.save(self.model.state_dict(), self.weight_path)
                    
                print("==============================================================================")
                info = "val_loss: {:.4f}, full_seq_acc: {:.4f}, char_acc: {:.4f}".format(val_loss, acc_full_seq, acc_per_char)
                print(info)
                self.logger.log(info)
                print("==============================================================================")

    def validate(self):
        self.model.eval()
        total_loss = []

        with torch.no_grad():
            for batch in self.val_loader:
                texts, tgt_input, tgt_output = batch
                texts, tgt_input, tgt_output = batch_to_device(texts, tgt_input, tgt_output, self.device)
                outputs = self.model(texts, tgt_input)
                outputs = outputs.flatten(0, 1)
                tgt_output = tgt_output.flatten()
                loss = self.criterion(outputs, tgt_output)
                total_loss.append(loss.item())
                del outputs,
                del loss,
                
        val_loss = np.mean(total_loss)
        self.model.train()
         
        return val_loss
    
    def train_step(self, batch):
        # get the inputs
        texts, tgt_input, tgt_output = batch
        texts, tgt_input, tgt_output = batch_to_device(texts, tgt_input, tgt_output, self.device)

        # zero the parameter gradients
        self.optimizer.zero_grad()
        
        # forward + backward + optimize + scheduler
        outputs = self.model(texts, tgt_input)
        outputs = outputs.flatten(0, 1)
        tgt_output = tgt_output.flatten()
        loss = self.criterion(outputs, tgt_output)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()
        self.scheduler.step()
        
        loss_item = loss.item()
        
        return loss_item
    
    def predict(self):
        pred_sents = []
        actual_sents = []

        for batch in self.val_loader:
            texts, tgt_input, tgt_output = batch
            texts, tgt_input, tgt_output = batch_to_device(texts, tgt_input, tgt_output, self.device)

            translated_sentence = translate(texts, self.model)

            pred_sent = self.vocab.batch_decode(translated_sentence.tolist())
            actual_sent = self.vocab.batch_decode(tgt_output.tolist())

            pred_sents.extend(pred_sent)
            actual_sents.extend(actual_sent)

        return pred_sents, actual_sents

    def precision(self):
        pred_sents, actual_sents  = self.predict()

        acc_full_seq = compute_accuracy(actual_sents, pred_sents, mode='full_sequence')
        acc_per_char = compute_accuracy(actual_sents, pred_sents, mode='per_char')

        return acc_full_seq, acc_per_char

    def load_weights(self, filename):
        state_dict = torch.load(filename, map_location=torch.device(self.device))

        for name, param in self.model.named_parameters():
            if name not in state_dict:
                print('{} not found'.format(name))
            elif state_dict[name].shape != param.shape:
                print(
                    '{} missmatching shape, required {} but found {}'.format(name, param.shape, state_dict[name].shape))
                del state_dict[name]

        self.model.load_state_dict(state_dict, strict=False)

    def save_weights(self, filename):
        path, _ = os.path.split(filename)
        os.makedirs(path, exist_ok=True)

        torch.save(self.model.state_dict(), filename)
    
Trainer().train()
