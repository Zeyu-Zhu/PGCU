import matplotlib.pyplot as plt
import numpy as np
import torch
import os


class Evaluate():
    
    def __init__(self, exp_type, data_type, device):
        self.path = 'result/'+exp_type+'/'+data_type+'exp'
        self.data_type = data_type
        self.exp_type = exp_type
        self.device = device
        self.best_o = 1
        self.best_g = 1
        i = 0
        while True:
            folder = os.path.exists(self.path + str(i))
            if folder is False:
                self.path = self.path + str(i) + '/'
                os.makedirs(self.path)
                break
            i += 1
            
    def visualize(self, o_train, o_test, g_train, g_test, pannet, pannet_pgcu):
        # Save the best model on training datasets
        if self.best_o > o_train[-1]:
            self.best_o = o_train[-1]
            torch.save(pannet, self.path+self.exp_type+'.pkl')
        if self.best_g > o_train[-1]:
            self.best_g = o_train[-1]
            torch.save(pannet_pgcu, self.path+self.exp_type+'_pgcu.pkl')
        
        text_o = self.exp_type
        test_g = self.exp_type + ' with PGCU'
        
        index = np.arange(len(o_train))
        index_ = np.arange(0, len(o_train), 10)
        
        plt.figure(1)
        plt.grid(color='#7d7f7c', linestyle='-.')
        plt.plot(index, o_train, 'c', linewidth=1.5, label="train "+text_o)
        plt.plot(index_, o_test, '2c--', linewidth=1.5, label="test "+text_o)
        plt.plot(index, g_train, 'r', linewidth=1.5, label="train "+test_g)
        plt.plot(index_, g_test, '2r--', linewidth=1.5, label="test "+test_g)
        plt.title('Loss:'+self.exp_type+'&&'+self.exp_type+'withPGCU')
        plt.xlabel('epoch')
        plt.ylabel('MSE')
        plt.legend(loc=1)
        
        ylim = {'GF2':1e-2, 'WV2':5e-4, 'WV3':1e-2}
        plt.ylim(0, ylim[self.data_type[0:3]])
        plt.savefig(self.path + 'loss.jpg', dpi=300)
        plt.clf()
