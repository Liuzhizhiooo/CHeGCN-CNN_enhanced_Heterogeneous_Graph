# -*- coding: utf-8 -*-
import torch
import time


class BasicModule(torch.nn.Module):
    '''
    provide the save and load functions for model
    '''

    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name=str(type(self))

    def load(self, path, optimizer):
        if torch.cuda.is_available():
            snapshot = torch.load(path)
        else:
            snapshot = torch.load(path, map_location="cpu")

        # 弹栈
        model_state = snapshot.pop('model_state', snapshot)
        optimizer_state = snapshot.pop('optimizer_state', None)

        if model_state is not None:
            print("load model")
            self.load_state_dict(model_state)
            print(f'model{path} loaded')

        if optimizer is not None and optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        
        return snapshot


    def save(self, optimizer, name=None, **kwargs):
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')

        model_state = None
        optimizer_state = None
        model_state = self.state_dict()
        if optimizer is not None:
            optimizer_state = optimizer.state_dict()
        torch.save(
            dict(model_state=model_state,
                 optimizer_state=optimizer_state,
                 **kwargs
                ),
            name
        )
        print(f"\nmodel {name} saved! \n")
