# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:59:14 2020

@author: HQ Xie
"""
import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np
from utils import SNR_to_noise, initNetParams, train_step, val_step, train_mi
from dataset import EurDataset, collate_data
from src.transceiver import DeepSC
from src.mutual_info import Mine
from torch.utils.data import DataLoader
from tqdm import tqdm
from LPlot import plot_loss #L add on 2022-01-18

parser = argparse.ArgumentParser()#L: set variables
#parser.add_argument('--data-dir', default='data/train_data.pkl', type=str)
parser.add_argument('--vocab-file', default='data/europarl/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='saved_models/deepsc-Rayleigh', type=str)
parser.add_argument('--channel', default='Rayleigh', type=str, help = 'Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=40, type=int)#LLL change it for test on 2022-01-17 origin:80


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    """
    L: set a seed so that the results are consistent
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def validate(epoch, args, net):
    """
    L: Validate (development) dataset is to verify the network is not over fitted

    -input-
    epoch: number of epoch (not the amount) just for count
    args: all the arguments
    net: network structure e.g. DeepSC

    -output-
    total/len(test_iterator): average loss

    """
    test_eur = EurDataset('test') #L: define a object in class "EurDataset" and input the test dataset from europarl doc
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)#L: use the DataLoader function of pytorch loading the dataset
    net.eval()# L: just add it when you are going to evaluate your data
    pbar = tqdm(test_iterator) #L: A Progress Bar for Python
    total = 0 #L: sum of loss
    with torch.no_grad(): # save memory
        for sents in pbar:
            sents = sents.to(device)
            loss = val_step(net, sents, sents, 0.1, pad_idx,
                             criterion, args.channel)#L: deepsc encoder-channel encoder-channel-channel decoder-deepsc decoder

            total += loss #L: count loss
            pbar.set_description(
                'Epoch: {}; Type: VAL; Loss: {:.5f}'.format(
                    epoch + 1, loss #description for Progress Bar
                )
            )

    return total/len(test_iterator)


def train(epoch, args, net, mi_net=None):
    """
        L: training with/without mutual information estimation

        -input-
        epoch: number of epoch (not the amount) just for count
        args: all the arguments defined at the beginning
        net: network structure e.g. DeepSC
        mi_net: whether consider mutual information estimation

        -output-
        none

        """
    train_eur= EurDataset('train')#L: define a object in class "EurDataset" and input the train dataset from europarl doc
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    pbar = tqdm(train_iterator)

    noise_std = np.random.uniform(SNR_to_noise(5), SNR_to_noise(10), size=(1))
    #for lllsentsfordebug in pbar:
    #    lllsentsfordebug = lllsentsfordebug.to(device)

    for sents in pbar:
        sents = sents.to(device)

        if mi_net is not None:# L: train with Transformer where the loss function is mutual information
            mi = train_mi(net, mi_net, sents, 0.1, pad_idx, mi_opt, args.channel)
            loss = train_step(net, sents, sents, 0.1, pad_idx,
                              optimizer, criterion, args.channel, mi_net)
            pbar.set_description(
                'Epoch: {};  Type: Train; Loss: {:.5f}; MI {:.5f}'.format(
                    epoch + 1, loss, mi
                )
            )
        else:
            loss = train_step(net, sents, sents, noise_std[0], pad_idx,
                              optimizer, criterion, args.channel)
            pbar.set_description(
                'Epoch: {};  Type: Train; Loss: {:.5f}'.format(
                    epoch + 1, loss
                )
            )


if __name__ == '__main__':#L:similar to the main()in c language
    # setup_seed(10)
    args = parser.parse_args()
    args.vocab_file = '' + args.vocab_file#LLL change the data address at 2022-01-14 4:05pm origin:/import/antennas/Datasets/hx301/
    """ preparing the dataset """
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]


    """ define optimizer and loss function """
    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                        num_vocab, num_vocab, args.d_model, args.num_heads,
                        args.dff, 0.1).to(device)# L: include Transformer (?not sure
    mi_net = Mine().to(device)#L: establish mi_net and let GPU to take the computation work
    criterion = nn.CrossEntropyLoss(reduction = 'none')#L: reduction means no reduction will be applied, construct the CrossEntropyLoss
    optimizer = torch.optim.Adam(deepsc.parameters(),
                                 lr=2e-3, betas=(0.9, 0.98), eps=1e-8, weight_decay = 5e-4)#L: learning rate changed from "lr=1e-4"
    mi_opt = torch.optim.Adam(mi_net.parameters(), lr=1e-3)
    #opt = NoamOpt(args.d_model, 1, 4000, optimizer)
    initNetParams(deepsc)
    record_loss = []# LLL add on 2022-01-17
    for epoch in range(args.epochs):
        start = time.time()
        record_acc = 10

        train(epoch, args, deepsc,mi_net)#L: train without mutual information. add training mutual information 2022-0515 orgin:train(epoch, args, deepsc)
        avg_acc = validate(epoch, args, deepsc)#L: dev/val

        # Record the costs LLL add on 2022-01-17
        record_loss.append(avg_acc) # LLL add on 2022-01-17

        if avg_acc < record_acc:
            if not os.path.exists(args.checkpoint_path):
                os.makedirs(args.checkpoint_path)
            with open(args.checkpoint_path + '/checkpoint_{}.pth'.format(str(epoch + 1).zfill(2)), 'wb') as f:
                torch.save(deepsc.state_dict(), f)
            record_acc = avg_acc

    #record_loss = [] #LLL add # on 2022-01-17

    plot_loss(args.epochs, np.array(record_loss))#LLL add on 2022-01-18
