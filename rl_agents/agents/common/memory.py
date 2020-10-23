import random
from collections.__init__ import namedtuple

from rl_agents.configuration import Configurable

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'terminal', 'info'))


class ReplayMemory(Configurable):
    """
        Container that stores and samples transitions.
    """

    def __init__(self, config=None, transition_type=Transition):
        super(ReplayMemory, self).__init__(config)
        self.capacity = int(self.config['memory_capacity'])
        self.transition_type = transition_type
        self.memory = []
        self.position = 0

    @classmethod
    def default_config(cls):
        return dict(memory_capacity=10000,
                    n_steps=1,
                    gamma=0.99)

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:  # calculate the capacity memory buffer
            self.memory.append(None)
            self.position = len(self.memory) - 1
        elif len(self.memory) > self.capacity:
            self.memory = self.memory[:self.capacity]  # 取出memory中的前[0:capacity]个数据-->去除后面的数据
        # Faster than append and pop
        self.memory[self.position] = self.transition_type(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, collapsed=True):
        """
            Sample a batch of transitions.

            If n_steps is greater than one, the batch will be composed of lists of successive transitions.
        :param batch_size: size of the batch
        :param collapsed: whether successive transitions must be collapsed into one n-step transition.
        :return: the sampled batch
        """
        # TODO: use agent's np_random for seeding
        if self.config["n_steps"] == 1:
            # Directly sample transitions
            return random.sample(self.memory, batch_size)  # from the memory buffer sample with random
        else:
            # Sample initial transition indexes
            indexes = random.sample(range(len(self.memory)), batch_size)  # 从经验池容量中随机采样batch_size大小的数据
            # Get the batch of n-consecutive-transitions starting from sampled indexes  从采样索引开始获取n-连续转换的批次
            all_transitions = [self.memory[i:i + self.config["n_steps"]] for i in indexes]
            # Collapse transitions  折叠转换  map(函数,迭代次数序列)  依据提供的函数对指定序列做映射
            return map(self.collapse_n_steps, all_transitions) if collapsed else all_transitions

    def collapse_n_steps(self, transitions):
        """
            Collapse n transitions <s,a,r,s',t> of a trajectory into one transition <s0, a0, Sum(r_i), sp, tp>.

            We start from the initial state, perform the first action, and then the return estimate is formed by
            accumulating the discounted rewards along the trajectory until a terminal state or the end of the
            trajectory is reached.
        :param transitions: A list of n successive transitions
        :return: The corresponding n-step transition
        """
        state, action, cumulated_reward, next_state, done, info = transitions[0]  # initial the memory buffer
        discount = 1
        for transition in transitions[1:]:
            if done:
                break
            else:
                _, _, reward, next_state, done, info = transition
                discount *= self.config['gamma']  # DISCOUNT decrease with the time iteration
                cumulated_reward += discount * reward    # 累积奖励 = 当前奖励 + 折扣奖励
        return state, action, cumulated_reward, next_state, done, info   # update memory buffer

    def __len__(self):
        return len(self.memory)

    def is_full(self):
        return len(self.memory) == self.capacity  # judge weather the memory is full

    def is_empty(self):
        return len(self.memory) == 0    # judge weather the memory is empty
