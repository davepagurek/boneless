import datetime
import numpy as np
import scipy.signal

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def args_to_str(args):
    """Convert cmd line args into a logdir string for experiment logging"""
    exp_name = '{}/'.format(args.prefix)
    exp_name += '{}/'.format("boneless")
    exp_name += '{}/'.format(args.loss_mode)
    exp_name += '{}/'.format(args.psi_mode)
    exp_name += 'gamma{}'.format(args.gamma)
    exp_name += '-lam{}'.format(args.lam)
    exp_name += '-pi_it{}'.format(args.train_pi_iters)
    exp_name += '-v_it{}'.format(args.train_v_iters)
    exp_name += '-steps{}'.format(args.steps_per_epoch)
    exp_name += '-pi_lr{}'.format(args.pi_lr)
    exp_name += '-v_lr{}'.format(args.v_lr)
    exp_name += '-v_lr{}'.format(args.v_lr)
    exp_name += '-seed{}'.format(args.seed)
    exp_name += '-clip{}'.format(args.clip_ratio)
    exp_name += '-{}'.format(args.suffix)
    exp_name += '-{}'.format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    return exp_name
