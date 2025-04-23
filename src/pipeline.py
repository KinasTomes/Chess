#!/usr/bin/env python

from alpha_net import ChessNet, train
from MCTS_chess import MCTS_self_play
import os
import pickle
import numpy as np
import torch
import torch.multiprocessing as mp

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Ensure required directories exist
ensure_dir("./model_data")
ensure_dir("./datasets/iter0")
ensure_dir("./datasets/iter1")
ensure_dir("./datasets/iter2")

if __name__=="__main__":
    for iteration in range(10):
        print(f"\n{'='*50}")
        print(f"Starting Iteration {iteration + 1}/10")
        print(f"{'='*50}\n")
        
        # Runs MCTS
        print(f"\nRunning MCTS for iteration {iteration + 1}...")
        net_to_play="current_net_trained8_iter1.pth.tar"
        mp.set_start_method("spawn",force=True)
        net = ChessNet()
        cuda = torch.cuda.is_available()
        if cuda:
            net.cuda()
        net.share_memory()
        net.eval()
        current_net_filename = os.path.join("./model_data/",\
                                        net_to_play)
        checkpoint = torch.load(current_net_filename)
        net.load_state_dict(checkpoint['state_dict'])
        processes1 = []
        for i in range(6):
            p1 = mp.Process(target=MCTS_self_play,args=(net,50,i))
            p1.start()
            processes1.append(p1)
        for p1 in processes1:
            p1.join()
        print(f"Finished MCTS for iteration {iteration + 1}")
            
        # Runs Net training
        print(f"\nStarting training for iteration {iteration + 1}...")
        net_to_train="current_net_trained8_iter1.pth.tar"; save_as="current_net_trained8_iter1.pth.tar"
        # gather data
        data_path = "./datasets/iter0/"
        datasets = []
        for idx,file in enumerate(os.listdir(data_path)):
            filename = os.path.join(data_path,file)
            with open(filename, 'rb') as fo:
                datasets.extend(pickle.load(fo, encoding='bytes'))
        data_path = "./datasets/iter1/"
        for idx,file in enumerate(os.listdir(data_path)):
            filename = os.path.join(data_path,file)
            with open(filename, 'rb') as fo:
                datasets.extend(pickle.load(fo, encoding='bytes'))
        data_path = "./datasets/iter2/"
        for idx,file in enumerate(os.listdir(data_path)):
            filename = os.path.join(data_path,file)
            with open(filename, 'rb') as fo:
                datasets.extend(pickle.load(fo, encoding='bytes'))
        datasets = np.array(datasets)
        
        mp.set_start_method("spawn",force=True)
        net = ChessNet()
        cuda = torch.cuda.is_available()
        if cuda:
            net.cuda()
        net.share_memory()
        net.train()
        current_net_filename = os.path.join("./model_data/",\
                                        net_to_train)
        checkpoint = torch.load(current_net_filename)
        net.load_state_dict(checkpoint['state_dict'])
        
        processes2 = []
        for i in range(6):
            p2 = mp.Process(target=train,args=(net,datasets,0,200,i))
            p2.start()
            processes2.append(p2)
        for p2 in processes2:
            p2.join()
        # save results
        torch.save({'state_dict': net.state_dict()}, os.path.join("./model_data/",\
                                        save_as))
        print(f"Finished training for iteration {iteration + 1}")