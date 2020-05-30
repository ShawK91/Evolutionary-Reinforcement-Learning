
from torch import nn
from torch.autograd import Variable
import random, pickle, copy, argparse
import numpy as np, torch, os
from torch import distributions
import torch.nn.functional as F

class Tracker(): #Tracker

    def __init__(self, save_folder, vars_string, project_string):
        self.vars_string = vars_string; self.project_string = project_string
        self.foldername = save_folder
        self.all_tracker = [[[],0.0,[]] for _ in vars_string] #[Id of var tracked][fitnesses, avg_fitness, csv_fitnesses]
        self.counter = 0
        self.conv_size = 1
        if not os.path.exists(self.foldername):
            os.makedirs(self.foldername)


    def update(self, updates, generation):
        """Add a metric observed

        Parameters:
            updates (list): List of new scoresfor each tracked metric
            generation (int): Current gen

        Returns:
            None
        """

        self.counter += 1
        for update, var in zip(updates, self.all_tracker):
            if update == None: continue
            var[0].append(update)

        #Constrain size of convolution
        for var in self.all_tracker:
            if len(var[0]) > self.conv_size: var[0].pop(0)

        #Update new average
        for var in self.all_tracker:
            if len(var[0]) == 0: continue
            var[1] = sum(var[0])/float(len(var[0]))

        if self.counter % 1 == 0:  # Save to csv file
            for i, var in enumerate(self.all_tracker):
                if len(var[0]) == 0: continue
                var[2].append(np.array([generation, var[1]]))
                filename = self.foldername + self.vars_string[i] + self.project_string
                np.savetxt(filename, np.array(var[2]), fmt='%.3f', delimiter=',')

def weights_init_(m, lin_gain=1.0, bias_gain=0.1):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=lin_gain)
        torch.nn.init.constant_(m.bias, bias_gain)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def hard_update(target, source):
    """Hard update (clone) from target network to source

        Parameters:
              target (object): A pytorch model
              source (object): A pytorch model

        Returns:
            None
    """

    for target_param, param in zip(target.parameters(), source.parameters()):

        target_param.data.copy_(param.data)


def soft_update(target, source, tau):
    """Soft update from target network to source

        Parameters:
              target (object): A pytorch model
              source (object): A pytorch model
              tau (float): Tau parameter

        Returns:
            None

    """

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def to_numpy(var):
    """Tensor --> numpy

    Parameters:
        var (tensor): tensor

    Returns:
        var (ndarray): ndarray
    """
    return var.data.numpy()

def to_tensor(ndarray, volatile=False, requires_grad=False):
    """numpy --> Variable

    Parameters:
        ndarray (ndarray): ndarray
        volatile (bool): create a volatile tensor?
        requires_grad (bool): tensor requires gradients?

    Returns:
        var (variable): variable
    """

    if isinstance(ndarray, list): ndarray = np.array(ndarray)
    return Variable(torch.from_numpy(ndarray).float(), volatile=volatile, requires_grad=requires_grad)

def pickle_obj(filename, object):
    """Pickle object

    Parameters:
        filename (str): folder to dump pickled object
        object (object): object to pickle

    Returns:
        None
    """

    handle = open(filename, "wb")
    pickle.dump(object, handle)

def unpickle_obj(filename):
    """Unpickle object from disk

    Parameters:
        filename (str): file from which to load and unpickle object

    Returns:
        obj (object): unpickled object
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)

def init_weights(m):
    """Initialize weights using kaiming uniform initialization in place

    Parameters:
        m (nn.module): Linear module from torch.nn

    Returns:
        None
    """
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def list_mean(l):
    """compute avergae from a list

    Parameters:
        l (list): list

    Returns:
        mean (float): mean
    """
    if len(l) == 0: return None
    else: return sum(l)/len(l)

def pprint(l):
    """Pretty print

    Parameters:
        l (list/float/None): object to print

    Returns:
        pretty print str
    """

    if isinstance(l, list):
        if len(l) == 0: return None
    else:
        if l == None: return None
        else: return '%.2f'%l


def flatten(d):
    """Recursive method to flatten a dict -->list

        Parameters:
            d (dict): dict

        Returns:
            l (list)
    """

    res = []  # Result list
    if isinstance(d, dict):
        for key, val in sorted(d.items()):
            res.extend(flatten(val))
    elif isinstance(d, list):
        res = d
    else:
        res = [d]
    return res

def reverse_flatten(d, l):
    """Recursive method to unflatten a list -->dict [Reverse of flatten] in place

        Parameters:
            d (dict): dict
            l (list): l

        Returns:
            None
    """

    if isinstance(d, dict):
        for key, _ in sorted(d.items()):

            #FLoat is immutable so
            if isinstance(d[key], float):
                d[key] = l[0]
                l[:] = l[1:]
                continue

            reverse_flatten(d[key], l)
    elif isinstance(d, list):
        d[:] = l[0:len(d)]
        l[:] = l[len(d):]

def load_all_models_dir(dir, model_template):
    """Load all models from a given directory onto a template

        Parameters:
            dir (str): directory
            model_template (object): Class template to load the objects onto

        Returns:
            models (list): list of loaded objects
    """

    list_files = os.listdir(dir)
    print(list_files)
    models = []
    for i, fname in enumerate(list_files):
        try:
            model_template.load_state_dict(torch.load(dir + fname))
            model_template.eval()
            models.append(copy.deepcopy(model_template))
        except:
            print(fname, 'failed to load')
    return models







