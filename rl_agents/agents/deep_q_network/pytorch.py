import logging
import torch

from rl_agents.agents.common.memory import Transition  # 转移函数
from rl_agents.agents.common.models import model_factory, size_model_config, trainable_parameters  # 加载各种模型
from rl_agents.agents.common.optimizers import loss_function_factory, optimizer_factory  # 加载损失函数以及优化器
from rl_agents.agents.common.utils import choose_device  # 选择设备CPU或者GPU
from rl_agents.agents.deep_q_network.abstract import AbstractDQNAgent  # 必须继承抽象类AbstractDQNAgent

logger = logging.getLogger(__name__)  # 日志加载项


class DQNAgent(AbstractDQNAgent):  # to definition your own network ,it must to inherit(继承) nn.Module class
    def __init__(self, env, config=None):
        super(DQNAgent, self).__init__(env, config)
        size_model_config(self.env, self.config["model"])
        self.value_net = model_factory(
            self.config["model"])  # the parameter 'type' in the default configuration by one external json file
        self.target_net = model_factory(self.config["model"])  # model selection
        """[multiplayerPerceptron|DuelingNetwork|ConvolutionalNetwork|EgoAttentionNetwork]"""
        self.target_net.load_state_dict(self.value_net.state_dict())
        """ model.state_dict() 存放训练过程中要学习的权重和偏置系数; load_state_dict() 加载网络结构名称和对应的参数"""
        self.target_net.eval()
        '''# 使用此命令后‘dropout'层和’batch Normalization‘才会进入evaluation状态(固定dropout和归一化层),固定两防止偏置参数变化--测试集'''
        logger.debug("Number of trainable parameters: {}".format(trainable_parameters(self.value_net)))
        """创建一条级别为debug的记录(调试)，其中存放value.net的训练参数"""
        self.device = choose_device(self.config["device"])  # GPU or CPU
        self.value_net.to(self.device)  # MOVE and/or casts (强制转换) parameters and buffers
        self.target_net.to(self.device)  # net.to 是为了实现模型在GPU、CPU之间的转换
        self.loss_function = loss_function_factory(
            self.config["loss_function"])  # choose loss function[l1 | l2 | smooth_l1 | bce]
        self.optimizer = optimizer_factory(self.config["optimizer"]["type"],  # choose the optimizer
                                           self.value_net.parameters(),  # [ADAM | RMS_PROP | RANGER:RAdam+Lookahead]
                                           **self.config["optimizer"])  # Root Mean Square Prop(均方根比例)
        # ADMA结合了momentum的动量进行梯度累积以及RMS_PRRP加速收敛
        '''前两者为自适应动量优化器，需要数据预热防止开始时候糟糕打局部优化效果'''
        self.steps = 0

        """ 优化器依据网络反向传播的梯度信息来更新和计算影响模型训练和模型输出的网络参数，使其逼近或者达到最优，最大或者最小化损失函数"""
    def step_optimizer(self, loss):
        # Optimize the model || each batch set the gradient initial to zero || prevent backward to accumulate gradient
        self.optimizer.zero_grad()  # 防止batch之间梯度相互关联
        loss.backward()  # backward to get the gradient
        for param in self.value_net.parameters():
            param.grad.data.clamp_(-1, 1)  # 将输入的梯度.data此数据张量夹紧到[-1,1]之间返回一个新的张量
        self.optimizer.step()  # update all parameters (weights matrix)

    def compute_bellman_residual(self, batch, target_state_action_value=None):
        # Compute concatenate the batch elements 计算合并批处理元素
        if not isinstance(batch.state, torch.Tensor):  # 认为子类是一种父类类型考虑inherit关系,type()不考虑,判断state是否是tensor
            # logger.info("Casting the batch to torch.tensor")
            state = torch.cat(tuple(torch.tensor([batch.state], dtype=torch.float))).to(
                self.device)  # cat:tensor splicing
            action = torch.tensor(batch.action, dtype=torch.long).to(self.device)
            reward = torch.tensor(batch.reward, dtype=torch.float).to(self.device)
            next_state = torch.cat(tuple(torch.tensor([batch.next_state], dtype=torch.float))).to(self.device)
            terminal = torch.tensor(batch.terminal, dtype=torch.bool).to(self.device)
            batch = Transition(state, action, reward, next_state, terminal, batch.info)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken  计算Q值,后选择采取的 action 列
        state_action_values = self.value_net(batch.state)
        state_action_values = state_action_values.gather(1, batch.action.unsqueeze(1)).squeeze(1)  # gather提取tensor元素
        """unsqueeze(i):在第i维增加一个维度，sequeeze(i):去除第i维度"""

        if target_state_action_value is None:
            with torch.no_grad():  # 上下文管理器，在该语句下的部分将不会跟踪梯度，不会backward
                # Compute V(s_{t+1}) for all next states.
                next_state_values = torch.zeros(batch.reward.shape).to(self.device)  # n_s_=reward大小的全零张量
                if self.config["double"]:
                    # Double Q-learning: pick best actions from policy network
                    _, best_actions = self.value_net(batch.next_state).max(1)  # 取出第一维,best_action
                    """max(1):返回下一状态值网络每一行最大值Q组成的一维数组--取出的是最大Q值，对应的Action组成的policy"""
                    # Double Q-learning: estimate action values from target network
                    best_values = self.target_net(batch.next_state).gather(1, best_actions.unsqueeze(1)).squeeze(1)
                else:
                    best_values, _ = self.target_net(batch.next_state).max(1)  # 切分成前后两个数组
                next_state_values[~batch.terminal] = best_values[~batch.terminal]
                # Compute the expected Q values      target_Q = r + GAMMA * Q(S')
                target_state_action_value = batch.reward + self.config["gamma"] * next_state_values

        # Compute loss        LOSS=( predict_Q - target_Q )
        loss = self.loss_function(state_action_values, target_state_action_value)
        return loss, target_state_action_value, batch

    def get_batch_state_values(self, states):
        values, actions = self.value_net(torch.tensor(states, dtype=torch.float).to(self.device)).max(1)
        return values.data.cpu().numpy(), actions.data.cpu().numpy()

    def get_batch_state_action_values(self, states):
        return self.value_net(torch.tensor(states, dtype=torch.float).to(self.device)).data.cpu().numpy()

    def save(self, filename):  # filename=permanent_folder / "latest.tar"
        state = {'state_dict': self.value_net.state_dict(),  # 保存训练好的值网络参数
                 'optimizer': self.optimizer.state_dict()}  # 优化器状态信息的字典
        torch.save(state, filename)  # torch.save(保存对象，类文件对象或一个保存文件的字符串4)
        return filename

    def load(self, filename):
        """ torch.load(类文件对象或一个保存文件的字符串,函数或者字典规定如何映射存储设备)"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.value_net.load_state_dict(checkpoint['state_dict'])  # 'state_dict': self.value_net.state_dict()
        self.target_net.load_state_dict(checkpoint['state_dict'])  # 'state_dict': self.value_net.state_dict()
        self.optimizer.load_state_dict(checkpoint['optimizer'])  # 'optimizer': self.optimizer.state_dict()
        return filename

    def initialize_model(self):
        self.value_net.reset()

    def set_writer(self, writer):
        super().set_writer(writer)
        model_input = torch.zeros((1, *self.env.observation_space.shape), dtype=torch.float, device=self.device) # [1,n]
        self.writer.add_graph(self.value_net, input_to_model=(model_input,)),
        self.writer.add_scalar("agent/trainable_parameters", trainable_parameters(self.value_net), 0)  # 标量的参数导入
