import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchtext.legacy import data
import torchtext
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab
from collections import Counter
import re
import io
import os
import nltk
import numpy as np
import math
import matplotlib.pyplot as plt

#Initializations
epochs = 15
batch_size = 128
vocabulary_size = 5160
encoder_hidden_size = 256
decoder_hidden_size = 256
embedded_size = 300
latent_size = 100
encoder_layers = 2
decoder_layers = 4
rec_coef = 8
kld_coef = 0.001
lr = 0.0001

#Special word tokens
unk_token = "<unk>"
pad_token = "<pad>"
start_token = "<sos>"
end_token = "<eos>"

#Convert sentence to words
def make_tokens(sentence):                   
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    tokenized_sentence = tokenizer.tokenize(sentence)
    return tokenized_sentence

#Dataset class with function to prepare dataset
class MyDataset(data.Dataset):
    def __init__(self, path, text_field, **kwargs):
        fields = [('text', text_field)]
        examples = []
        with open(path, 'r',encoding='utf-8') as f:
            for text in f:
                examples.append(data.Example.fromlist([text], fields))
        super(MyDataset, self).__init__(examples, fields, **kwargs)
    @classmethod
    def splits(cls, text_field, train='train', **kwargs):
        return super(MyDataset, cls).splits(text_field=text_field, train=train, **kwargs)

#Move to cuda
def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

#Encoder class
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_layers = encoder_layers
        self.lstm = nn.LSTM(input_size=embedded_size, hidden_size=encoder_hidden_size, num_layers=encoder_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=0.3)

    def init_hidden(self, batch_size):
        h_init = torch.zeros(2*self.encoder_layers, batch_size, self.encoder_hidden_size)
        c_init = torch.zeros(2*self.encoder_layers, batch_size, self.encoder_hidden_size)
        self.hidden = (to_cuda(h_init), to_cuda(c_init))

    def forward(self, x):
        batch_size, sentence_size, embedded_size = x.size()
        self.init_hidden(batch_size)
        _, (self.hidden, _) = self.lstm(x, self.hidden)	            
        self.hidden = self.dropout(self.hidden)
        self.hidden = self.hidden.view(self.encoder_layers, 2, batch_size, self.encoder_hidden_size)
        self.hidden = self.hidden[-1]	                            
        hidden_output = torch.cat(list(self.hidden), dim=1)	               
        return hidden_output

#Decoder class
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_layers = decoder_layers
        self.latent_size = latent_size
        self.lstm = nn.LSTM(input_size=embedded_size+latent_size, hidden_size=decoder_hidden_size, num_layers=decoder_layers, batch_first=True)
        self.fc = nn.Linear(decoder_hidden_size, vocabulary_size)
        self.dropout = nn.Dropout(p=0.3)

    def init_hidden(self, batch_size):
        h_init = torch.zeros(self.decoder_layers, batch_size, self.decoder_hidden_size)
        c_init = torch.zeros(self.decoder_layers, batch_size, self.decoder_hidden_size)
        self.hidden = (to_cuda(h_init), to_cuda(c_init))

    def forward(self, x, z, decoder_hidden = None):
        batch_size, sentence_size, embedded_size = x.size()
        z = torch.cat([z]*sentence_size, 1).view(batch_size, sentence_size, self.latent_size)	
        x = torch.cat([x,z], dim=2)	                                    

        if decoder_hidden is None:	                                    
            self.init_hidden(batch_size)
        else:					                                    
            self.hidden = decoder_hidden

        output, self.hidden = self.lstm(x, self.hidden)
        output = self.dropout(output)
        output = self.fc(output)

        return output, self.hidden	                              

#VAE class
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedded_size)
        self.encoder = Encoder()
        self.find_mean = nn.Linear(2*encoder_hidden_size, latent_size)
        self.find_log_variance = nn.Linear(2*decoder_hidden_size, latent_size)
        self.decoder = Decoder()
        self.latent_size = latent_size

    def forward(self, x, decoder_input, z = None, decoder_hidden = None):
        if z is None:	                                               
            batch_size, sentence_size = x.size()
            x = self.embedding(x)	                                   
            encoder_hidden_output = self.encoder(x)
            
            mean_out = self.find_mean(encoder_hidden_output)	                       
            log_variance = self.find_log_variance(encoder_hidden_output)	               
            z = to_cuda(torch.randn([batch_size, self.latent_size]))	           
            z = mean_out + z*torch.exp(0.5*log_variance)	                           
            kld = -0.5*torch.sum(log_variance-mean_out.pow(2)-log_variance.exp()+1, 1).mean()
        else:
            kld = None                                                 

        decoder_input = self.embedding(decoder_input)	                                

        output, decoder_hidden = self.decoder(decoder_input, z, decoder_hidden)
        return output, decoder_hidden, kld

train_loss_list = []
val_loss_list = []
train_KL_list = []
val_KL_list = []

save_path = "data/saved_models/vae_model.tar"
if not os.path.exists("data/saved_models"):
    os.makedirs("data/saved_models")

os.environ["CUDA_VISIBLE_DEVICES"] = str(1)    

text_field = data.Field(init_token=start_token, eos_token=end_token, lower=True, tokenize=make_tokens, batch_first=True)
train_data, val_data = MyDataset.splits(path="", train = "train.txt", test="test.txt", text_field=text_field)
text_field.build_vocab(train_data, val_data, max_size=vocabulary_size-4, vectors = 'glove.6B.300d')
vocab = text_field.vocab
train_iter, val_iter = data.BucketIterator.splits((train_data, val_data), batch_size=batch_size, sort_key = lambda x: len(x.text), repeat = False, device = torch.device('cuda'))
    

vae = VAE()
weight_matrix = vocab.vectors
vae.embedding.weight.data.copy_(weight_matrix)            
vae = to_cuda(vae)

optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

def train_batch(x, decoder_input, step, train = True):
    output, _, kld = vae(x, decoder_input, None, None)
    output = output.view(-1, vocabulary_size)	               
    x = x.contiguous().view(-1)	                           
    rec_loss = F.cross_entropy(output, x)
    loss = rec_coef*rec_loss + kld_coef*kld
    if train == True:	                                   
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return rec_loss.item(), kld.item()

def training():
    start_epoch = step = 0
    for epoch in range(start_epoch, epochs):
        vae.train()
        train_rec_loss = []
        train_kl_loss = []
        for batch in train_iter:
            x = batch.text 	                             
            decoder_input = x
            rec_loss, kl_loss = train_batch(x, decoder_input, step, train=True)
            train_rec_loss.append(rec_loss)
            train_kl_loss.append(kl_loss)
            step += 1

        vae.eval()
        valid_rec_loss = []
        valid_kl_loss = []
        for batch in val_iter:
            x = batch.text
            decoder_input = x
            with torch.autograd.no_grad():
                rec_loss, kl_loss = train_batch(x, decoder_input, step, train=False)
            valid_rec_loss.append(rec_loss)
            valid_kl_loss.append(kl_loss)

        train_rec_loss = np.mean(train_rec_loss)
        train_kl_loss = np.mean(train_kl_loss)
        valid_rec_loss = np.mean(valid_rec_loss)
        valid_kl_loss = np.mean(valid_kl_loss)

        print("Epoch -> ", epoch)
        print("Train data -> Reconstruction loss = ", train_rec_loss,", KL divergence = ", train_kl_loss)
        print("Validation data -> Reconstruction loss = ", valid_rec_loss,", KL divergence = ", valid_kl_loss)
        train_loss_list.append(train_rec_loss)
        train_KL_list.append(train_kl_loss)
        val_loss_list.append(valid_rec_loss)
        val_KL_list.append(valid_kl_loss)
        if epoch%5==0:
            torch.save({
                'epoch': epoch + 1,
                'vae_dict': vae.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step
            }, save_path)

def generate_sentence(input):
    checkpoint = torch.load(save_path)
    vae.load_state_dict(checkpoint['vae_dict'])
    vae.eval()
    del checkpoint
    inp = torch.tensor([[vocab.stoi[i] for i in input.split()]])
    inp = to_cuda(inp)
    output, _, kld = vae(inp, inp, None, None)
    probs = F.softmax(output[0], dim=1)
    final_out = torch.multinomial(probs,1)
    str = ""
    for i in final_out:
        next_word = vocab.itos[i.item()] 
        str += next_word + " "
    print(str)

if __name__ == '__main__':
    generate_sentence("Enter sentence")
   
    
