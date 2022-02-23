import numpy as np
from utils import get_idxs_per_relation
from mpi4py import MPI


ALL_MASKS = True


class GoalSampler:
    def __init__(self, args):
        self.num_rollouts_per_mpi = args.num_rollouts_per_mpi
        self.rank = MPI.COMM_WORLD.Get_rank()

        self.goal_dim = args.env_params['goal']
        self.relation_ids = get_idxs_per_relation(n=args.n_blocks)

        self.init_stats()

    def sample_goal(self, n_goals):
        """
        Sample n_goals goals to be targeted during rollouts
        evaluation controls whether or not to sample the goal uniformly or according to curriculum
        """
        return np.zeros((n_goals, self.goal_dim))

    # def update(self, episodes, t):
    #     """
    #     Update discovered goals list from episodes
    #     Update list of successes and failures for LP curriculum
    #     Label each episode with the last ag (for buffer storage)
    #     """
    #     all_episodes = MPI.COMM_WORLD.gather(episodes, root=0)

    #     if self.rank == 0:
    #         all_episode_list = [e for eps in all_episodes for e in eps]

    #         for e in all_episode_list:
    #             # Add last achieved goal to memory if first time encountered
    #             if str(e['ag_binary'][-1]) not in self.discovered_goals_str:
    #                 self.discovered_goals.append(e['ag_binary'][-1].copy())
    #                 self.discovered_goals_str.append(str(e['ag_binary'][-1]))

    #     self.sync()

    #     return episodes

    # def sync(self):
    #     self.discovered_goals = MPI.COMM_WORLD.bcast(self.discovered_goals, root=0)
    #     self.discovered_goals_str = MPI.COMM_WORLD.bcast(self.discovered_goals_str, root=0)

    # def build_batch(self, batch_size):
    #     goal_ids = np.random.choice(np.arange(len(self.discovered_goals)), size=batch_size)
    #     return goal_ids

    def init_stats(self):
        self.stats = dict()
        # Number of classes of eval
        self.stats['Eval_SR'] = []
        self.stats['Av_Rew'] = []
        self.stats['epoch'] = []
        self.stats['episodes'] = []
        self.stats['global_sr'] = []
        keys = ['goal_sampler', 'rollout', 'store', 'norm_update',
                'policy_train', 'eval', 'epoch', 'total']
        for k in keys:
            self.stats['t_{}'.format(k)] = []

    def save(self, epoch, episode_count, av_res, av_rew, global_sr, time_dict):
        self.stats['epoch'].append(epoch)
        self.stats['episodes'].append(episode_count)
        self.stats['global_sr'].append(global_sr)
        for k in time_dict.keys():
            self.stats['t_{}'.format(k)].append(time_dict[k])
        self.stats['Eval_SR'].append(av_res[0])
        self.stats['Av_Rew'].append(av_rew[0])
