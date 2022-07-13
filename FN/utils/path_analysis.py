import random
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

def get_paras(net):
    paras=[]
    for name,parameters in net.named_parameters():
        paras.append(parameters)
    return paras

def get_active_neurons4(net,sample):
    neurons=[]
    def hook(module,input,output):
        neurons.append(output.data.detach().clone())
    handle1=net.fc1.register_forward_hook(hook)
    handle2=net.fc2.register_forward_hook(hook)
    handle3=net.fc3.register_forward_hook(hook)
    handle4=net.fc4.register_forward_hook(hook)
    net(x=torch.tensor(sample,dtype=torch.float32))
    handle1.remove()
    handle2.remove()
    handle3.remove()
    handle4.remove()
    return neurons

def get_contrib4(paras,neurons):
    contrib_list=[]
    for i in range(3):
        i=i
        contrib=neurons[i]*paras[2*i+2]
        contrib_list.append(contrib.detach().clone())
    return contrib_list

def get_path_set4(net,sample,GAMMA=0.9):
    active_neuron_indice=[[],[],[],[]]
    path_set=set()
    neurons=get_active_neurons4(net,sample)
    paras=get_paras(net)
    contrib_list=get_contrib4(paras,neurons)
    active_neuron_indice[3].append(torch.argmax(neurons[3]).item())
    for i in range(3):
        L=3-i
        for j in active_neuron_indice[L]:
            s=torch.sort(contrib_list[L-1][j],descending=True)
            sum=0
            for k in range(len(contrib_list[L-1][j])):
                sum+=s.values[k].item()
                active_neuron_indice[L-1].append(s.indices[k].item())
                path_set.add((L,s.indices[k].item(),j))
                if(sum>=GAMMA*neurons[L][j].item()):
                    break
    return path_set



def sample_sort(net, train_dataset, THETA=1e-3, GAMMA=0.9):
    net=net.cpu()
    # THETA = 1e-3
    path_set_list=[]
    for i in (range(len(train_dataset))):
        path_set=get_path_set4(net,train_dataset[i][0],GAMMA=GAMMA)
        path_set_list.append(path_set)
    v=pd.value_counts(path_set_list).rename_axis('pathset').reset_index(name='counts')
#     v_list.append(v)
    t=tuple(v[v.counts<=max(v.counts[0]*THETA,1)].pathset)
    adv_data_idx=[]
    for i in range(len(path_set_list)):
        if path_set_list[i] in t:
            adv_data_idx.append(i)
    print("frac:{}".format(len(adv_data_idx)/len(train_dataset)))
    return adv_data_idx

v_list=[]
def sample_sort_test(net, train_dataset, THETA=1e-3, GAMMA=0.9):
    net=net.cpu()
    # THETA = 1e-3
    path_set_list=[]
    for i in (range(len(train_dataset))):
        path_set=get_path_set4(net,train_dataset[i][0],GAMMA=GAMMA)
        path_set_list.append(path_set)
    v=pd.value_counts(path_set_list).rename_axis('pathset').reset_index(name='counts')
    v_list.append(v)
    t=tuple(v[v.counts<=max(v.counts[0]*THETA,1)].pathset)
    adv_data_idx=[]
    for i in range(len(path_set_list)):
        if path_set_list[i] in t:
            adv_data_idx.append(i)
    print("frac:{}".format(len(adv_data_idx)/len(train_dataset)))
    return adv_data_idx

def get_adv(train_dataset,adv_data_idx,BATCH_SIZE=128):
    x_t_adv, y_t_adv, l_t_adv, s_t_adv = (None,None,None,None)
    for i in range(len(train_dataset)):
        if i in adv_data_idx:
            a,b,c,d=train_dataset[i]
            x_t_adv = a.unsqueeze(0) if x_t_adv is None else torch.cat((x_t_adv,a.unsqueeze(0)),0)
            y_t_adv = b.unsqueeze(0) if y_t_adv is None else torch.cat((y_t_adv,b.unsqueeze(0)),0)
            l_t_adv = c.unsqueeze(0) if l_t_adv is None else torch.cat((l_t_adv,c.unsqueeze(0)),0)
            s_t_adv = d.unsqueeze(0) if s_t_adv is None else torch.cat((s_t_adv,d.unsqueeze(0)),0)
    x_t_benign, y_t_benign, l_t_benign, s_t_benign = (None,None,None,None)
    for i in range(len(train_dataset)):
        if i not in adv_data_idx:
            a,b,c,d=train_dataset[i]
            x_t_benign = a.unsqueeze(0) if x_t_benign is None else torch.cat((x_t_benign,a.unsqueeze(0)),0)
            y_t_benign = b.unsqueeze(0) if y_t_benign is None else torch.cat((y_t_benign,b.unsqueeze(0)),0)
            l_t_benign = c.unsqueeze(0) if l_t_benign is None else torch.cat((l_t_benign,c.unsqueeze(0)),0)
            s_t_benign = d.unsqueeze(0) if s_t_benign is None else torch.cat((s_t_benign,d.unsqueeze(0)),0)

    adv_dataset = TensorDataset(x_t_adv, y_t_adv, l_t_adv, s_t_adv)
    adv_loader = DataLoader(dataset=adv_dataset, batch_size=BATCH_SIZE, shuffle=True)


    benign_dataset = TensorDataset(x_t_benign, y_t_benign, l_t_benign, s_t_benign)
    benign_loader = DataLoader(dataset=benign_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    return adv_loader,benign_loader

def get_adv_rand(train_dataset,adv_data_idx,BATCH_SIZE=128):
    adv_data_idx=random.choices(range(0,len(train_dataset)),k=len(adv_data_idx))
    x_t_adv, y_t_adv, l_t_adv, s_t_adv = (None,None,None,None)
    for i in range(len(train_dataset)):
        if i in adv_data_idx:
            a,b,c,d=train_dataset[i]
            x_t_adv = a.unsqueeze(0) if x_t_adv is None else torch.cat((x_t_adv,a.unsqueeze(0)),0)
            y_t_adv = b.unsqueeze(0) if y_t_adv is None else torch.cat((y_t_adv,b.unsqueeze(0)),0)
            l_t_adv = c.unsqueeze(0) if l_t_adv is None else torch.cat((l_t_adv,c.unsqueeze(0)),0)
            s_t_adv = d.unsqueeze(0) if s_t_adv is None else torch.cat((s_t_adv,d.unsqueeze(0)),0)
    x_t_benign, y_t_benign, l_t_benign, s_t_benign = (None,None,None,None)
    for i in range(len(train_dataset)):
        if i not in adv_data_idx:
            a,b,c,d=train_dataset[i]
            x_t_benign = a.unsqueeze(0) if x_t_benign is None else torch.cat((x_t_benign,a.unsqueeze(0)),0)
            y_t_benign = b.unsqueeze(0) if y_t_benign is None else torch.cat((y_t_benign,b.unsqueeze(0)),0)
            l_t_benign = c.unsqueeze(0) if l_t_benign is None else torch.cat((l_t_benign,c.unsqueeze(0)),0)
            s_t_benign = d.unsqueeze(0) if s_t_benign is None else torch.cat((s_t_benign,d.unsqueeze(0)),0)

    adv_dataset = TensorDataset(x_t_adv, y_t_adv, l_t_adv, s_t_adv)
    benign_dataset = TensorDataset(x_t_benign, y_t_benign, l_t_benign, s_t_benign)

    adv_loader = DataLoader(dataset=adv_dataset, batch_size=BATCH_SIZE, shuffle=True)
    benign_loader = DataLoader(dataset=benign_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return adv_loader,benign_loader