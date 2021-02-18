import os
import torch
import torch.nn as nn
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks
import numpy as np
import qtorch
from qtorch.quant import posit_quantize, float_quantize
from qtorch.quant import new_format_quantize, act_format_quantize, configurable_table_quantize
import math

act_table_global = []

def np_to_str(arr):
    res= list(map(lambda x: str(x), arr))
    return " ".join(res)
def filter_arr(arr, elem, tol=1e-5):
    filter_arr = []
    for item in arr : 
        if math.isclose(elem, item, abs_tol=tol):
            filter_arr.append(item)
    return filter_arr

weight_data  = np.array([])
act_data  = np.array([])
def my_quant_table (x, table):
    global weight_data
    temp = configurable_table_quantize(x,table, scale=1.0 )
    #numpy_data = temp.cpu().numpy()
    #print (numpy_data.shape)
    #weight_data = np.hstack([weight_data,numpy_data.flatten()])
    return temp
def my_quant (x):
    return x
    #return new_format_quantize(x)
    #return posit_quantize(x, nsize=8, es=1,scale=8.0)

def my_quant_16 (x):
    return x
    #return new_format_quantize(x)
    #return posit_quantize(x, nsize=16, es=1)
    
def linear_activation(input):
    global act_data
    global act_table_global 
    '''
    temp_input = input.cpu().detach().numpy()
    freq,bins = np.histogram(temp_input, bins=20)
    print ("---- activation hist ---- ")
    print (list(freq))
    print (list(bins))
    #return input
    f = open("act_freq.txt", "a")
    for item in freq:
        f.write("%d "% item)
    f.write("\n")
    f.close()

    f = open("act_bin.txt", "a")
    for item in bins:
        f.write("%f "% item)
    f.write("\n")
    f.close()    
    '''
    #temp = act_format_quantize(input)
    #minhhn acts only
    temp = configurable_table_quantize(input, act_table_global, scale= 1.0)
    return temp

def forward_pre_hook_linear(m, input):
    return (linear_activation(input[0]),) 

class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        if opt.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt, table=[], act_table = []):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            if (len(table) == 0):
                self.load_networks(load_suffix)
            else :
                self.load_networks_table(load_suffix, table = table, act_table = act_table)
        self.print_networks(opt.verbose)

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks_table(self, epoch, table=[] , act_table  = []):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        global act_table_global 
        if (len(table) == 0): 
            print ("null lookup table, something is wrong")
            exit(0)
        else: 
            print(table)
            torch_table = torch.tensor(table, dtype=torch.float)
            torch_table_act = torch.tensor(act_table, dtype=torch.float )
            act_table_global = torch_table_act
            print ("set torch table _act" )
            print (act_table_global)
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                #print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata



                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

                #print ("measuring stats weight")
                # params = np.array([])
                #f_freq = open("w_freq.txt", "a")
                #f_bins = open("w_bins.txt", "a")
                count = 0
                for layer in net.modules():
                    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                        print(layer)
                        if (count != 0 and count<=22):
                            layer.weight.data =  my_quant_table(layer.weight.data, torch_table)
                            if(len(act_table) > 0 ):
                                layer.register_forward_pre_hook(forward_pre_hook_linear)
                        else:
                            layer.weight.data =  my_quant_16(layer.weight.data)
                        #params =  layer.weight.data.cpu().detach().numpy().flatten()
                        #(n_s,bins) =  np.histogram(np.log2(np.absolute(params[params != 0])), bins=20)
                        #f_freq.write(np_to_str(n_s)+"\n")
                        #f_bins.write(np_to_str(bins)+"\n")
                        count = count +1
                        
                        #print ("hello ", count)
                #f_freq.close()
                #f_bins.close()
                # exit()
                print ("count layers ", count)
                print ("weight shape ", len(weight_data))

                #weight_table = [0.275395, 0.023438 ,0.076174, 0.12891,  0.42188 ]
                weight_table = sorted(table)
                print (weight_table)
                hist_positive = []
                hist_negative = []
                #print (weight_data[:100])
                for item in weight_table:
                   
                    hist_positive.append(len(filter_arr(weight_data, item, tol = min(weight_table)/2.0)))
                    hist_negative.append(len(filter_arr(weight_data, -item, tol = min(weight_table)/2.0)))
                    #hist_positive.append(len(weight_data[weight_data == item ]))
                    #hist_negative.append(len(weight_data[weight_data == -item]))
                print (sum(hist_positive), " ", sum(hist_negative), " ", sum(hist_positive) + sum(hist_negative))
                print (hist_positive)
                print (hist_negative)
                
                print ("done generating weight histogram ")
                

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata



                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

                print ("measuring stats weight")
                # params = np.array([])
                #f_freq = open("w_freq.txt", "a")
                #f_bins = open("w_bins.txt", "a")
                count = 0
                for layer in net.modules():
                    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                        print(layer)
                        if (count != 0 and count<=22):
                            layer.weight.data =  my_quant(layer.weight.data)
                        else:
                            layer.weight.data =  my_quant_16(layer.weight.data)
                        #params =  layer.weight.data.cpu().detach().numpy().flatten()
                        #(n_s,bins) =  np.histogram(np.log2(np.absolute(params[params != 0])), bins=20)
                        #f_freq.write(np_to_str(n_s)+"\n")
                        #f_bins.write(np_to_str(bins)+"\n")
                        count = count +1
                        
                        #print ("hello ", count)
                #f_freq.close()
                #f_bins.close()
                # exit()
                print ("count layers ", count)
                print ("done generating weight histogram ")


    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0

                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
