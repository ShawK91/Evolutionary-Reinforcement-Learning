import os
from torch.utils.tensorboard import SummaryWriter

class Parameters:
    def __init__(self, parser):
        """Parameter class stores all parameters for policy gradient

        Parameters:
            None

        Returns:
            None
        """

        #Env args
        self.env_name = vars(parser.parse_args())['env']
        self.frameskip = vars(parser.parse_args())['frameskip']

        self.total_steps = int(vars(parser.parse_args())['total_steps'] * 1000000)
        self.gradperstep = vars(parser.parse_args())['gradperstep']
        self.savetag = vars(parser.parse_args())['savetag']
        self.seed = vars(parser.parse_args())['seed']
        self.batch_size = vars(parser.parse_args())['batchsize']
        self.rollout_size = vars(parser.parse_args())['rollsize']

        self.hidden_size = vars(parser.parse_args())['hidden_size']
        self.critic_lr = vars(parser.parse_args())['critic_lr']
        self.actor_lr = vars(parser.parse_args())['actor_lr']
        self.tau = vars(parser.parse_args())['tau']
        self.gamma = vars(parser.parse_args())['gamma']
        self.reward_scaling = vars(parser.parse_args())['reward_scale']
        self.buffer_size = int(vars(parser.parse_args())['buffer'] * 1000000)
        self.learning_start = vars(parser.parse_args())['learning_start']

        self.pop_size = vars(parser.parse_args())['popsize']
        self.num_test = vars(parser.parse_args())['num_test']
        self.test_frequency = 1
        self.asynch_frac = 1.0  # Aynchronosity of NeuroEvolution

        #Non-Args Params
        self.elite_fraction = 0.2
        self.crossover_prob = 0.15
        self.mutation_prob = 0.90
        self.extinction_prob = 0.005  # Probability of extinction event
        self.extinction_magnituide = 0.5  # Probabilty of extinction for each genome, given an extinction event
        self.weight_magnitude_limit = 10000000
        self.mut_distribution = 1  # 1-Gaussian, 2-Laplace, 3-Uniform


        self.alpha = vars(parser.parse_args())['alpha']
        self.target_update_interval = 1
        self.alpha_lr = 1e-3

        #Save Results
        self.savefolder = 'Results/Plots/'
        if not os.path.exists(self.savefolder): os.makedirs(self.savefolder)
        self.aux_folder = 'Results/Auxiliary/'
        if not os.path.exists(self.aux_folder): os.makedirs(self.aux_folder)

        self.savetag += str(self.env_name)
        self.savetag += '_seed' + str(self.seed)
        self.savetag += '_roll' + str(self.rollout_size)
        self.savetag += '_pop' + str(self.pop_size)
        self.savetag += '_alpha' + str(self.alpha)


        self.writer = SummaryWriter(log_dir='Results/tensorboard/' + self.savetag)




