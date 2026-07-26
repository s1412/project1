import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import numpy as np
import copy
import random
import os
from typing import List, Tuple, Optional

try:
    from ID_TAP import Network
    from backpack import extend
except ImportError:
    print("Warning: Could not import Network from ID_TAP or backpack")
    Network = None

def is_lamp_dataset(lamp_type: int = None) -> bool:
    if lamp_type is None:
        lamp_type = int(os.environ.get('POHF_LAMP_TYPE', 0))

    return lamp_type in [4, 5, 8, 9, 10]

def get_input_dim_for_dataset(lamp_type: int = None, config: dict = None) -> int:
    try:
        from IDS_TAP_parameters import CONTEXTUAL_BANDIT_CONFIG
        contextual_enabled = CONTEXTUAL_BANDIT_CONFIG.get("enabled_for_lamp", True)
        contextual_dim = CONTEXTUAL_BANDIT_CONFIG.get("contextual_input_dim", 2048)
        standard_dim = CONTEXTUAL_BANDIT_CONFIG.get("standard_input_dim", 1024)
    except ImportError:
        contextual_enabled = True
        contextual_dim = 2048
        standard_dim = 1024

    if contextual_enabled and is_lamp_dataset(lamp_type):
        return contextual_dim
    else:
        return standard_dim

class Random:

    def __init__(self, input_dim, config=None):
        from IDS_TAP_parameters import EXPERIMENT_CONFIG, DATA_CONFIG

        self.total_iter = EXPERIMENT_CONFIG.get("total_iter", 100)
        self.n_init = EXPERIMENT_CONFIG.get("n_init", 10)
        self.times = DATA_CONFIG.get("times", 30)

        self.selection_count = self.total_iter - self.n_init
        self.selection_range = self.times

        print(f"🎲 Random algorithm initialization Complete")
        print(f"   Selection count: {self.selection_count} (total_iter={self.total_iter} - n_init={self.n_init})")
        print(f"   Selection range: {self.selection_range} (times={self.times})")

    def select_arm(self, items: List, history: Optional[List] = None) -> tuple:
        arm1 = random.randint(0, self.selection_range - 1)
        arm2 = random.randint(0, self.selection_range - 1)

        while arm2 == arm1:
            arm2 = random.randint(0, self.selection_range - 1)

        print(f"🎲 Random selection: arm1 {arm1}, arm2 {arm2} (range: 0-{self.selection_range-1})")
        return arm1, arm2

    def update(self, arm1_idx: int, arm2_idx: int, preference: int):
        pass

    def train_model(self, X1, X2, Y, incremental=False, weights=None):
        pass

class LinearModel(nn.Module):
    def __init__(self, input_dim, hidden_size=None, depth=None, dropout_rate=None, activation=None, init_params=None):
        super(LinearModel, self).__init__()

        try:
            from IDS_TAP_parameters import NETWORK_CONFIG
            default_hidden_size = NETWORK_CONFIG.get("hidden_size", 1024)
            default_depth = NETWORK_CONFIG.get("depth", 1)
            default_dropout_rate = NETWORK_CONFIG.get("dropout_rate", 0.1)
            default_activation = NETWORK_CONFIG.get("activation", "GELU")
        except ImportError:
            default_hidden_size = 1024
            default_depth = 1
            default_dropout_rate = 0.1
            default_activation = "GELU"

        hidden_size = hidden_size if hidden_size is not None else default_hidden_size
        depth = depth if depth is not None else default_depth
        dropout_rate = dropout_rate if dropout_rate is not None else default_dropout_rate
        activation = activation if activation is not None else default_activation

        if activation.upper() == "GELU":
            self.activate = nn.GELU()
        else:
            self.activate = nn.ReLU()

        self.dropout = nn.Dropout(p=dropout_rate)

        self.layer_list = nn.ModuleList()
        self.layer_list.append(nn.Linear(input_dim, hidden_size))
        for i in range(depth-1):
            self.layer_list.append(nn.Linear(hidden_size, hidden_size))
        self.layer_list.append(nn.Linear(hidden_size, 1))

        if init_params is None:
            for i in range(len(self.layer_list)):
                torch.nn.init.normal_(self.layer_list[i].weight, mean=0, std=1.0)
                torch.nn.init.normal_(self.layer_list[i].bias, mean=0, std=1.0)
        else:
            for i in range(len(self.layer_list)):
                self.layer_list[i].weight.data = init_params[i*2]
                self.layer_list[i].bias.data = init_params[i*2+1]

    def forward(self, x):
        y = x
        for i in range(len(self.layer_list)-1):
            y = self.activate(self.layer_list[i](y))
            y = self.dropout(y)
        y = self.layer_list[-1](y)
        return y

class LinearDuelingBandits:

    def __init__(self, input_dim, config=None):
        if config is None:
            from IDS_TAP_parameters import NETWORK_CONFIG, TRAINING_CONFIG, DEVICE_CONFIG, DATA_CONFIG
            network_config = NETWORK_CONFIG.copy()
            training_config = TRAINING_CONFIG
            device_config = DEVICE_CONFIG
            data_config = DATA_CONFIG
        else:
            network_config = config.get("network", {})
            training_config = config.get("training", {})
            device_config = config.get("device", {})
            data_config = config.get("data", {})

        device_type = device_config.get("device", "cuda:1")
        self.device = torch.device(device_type if torch.cuda.is_available() else "cpu")

        if self.device.type == 'cuda' and device_config.get("clear_cache", True):
            torch.cuda.empty_cache()
            print(f"LinearDuelingBandits - Using device: {self.device}, cleared CUDA cache")

        if input_dim is not None:
            self.input_dim = input_dim
        else:
            self.input_dim = get_input_dim_for_dataset(config=config)
            dataset_type = "LaMP(contextual)" if is_lamp_dataset() else "Standard"
            print(f"🎯 Using {dataset_type} mode embedding dimension: {self.input_dim}")

        network_config["input_dim"] = self.input_dim

        self.lamdba = 1.0
        self.nu = 1.0
        self.style = 'ucb'
        self.diagonalize = True

        hidden_size = network_config.get("hidden_size", None)
        depth = network_config.get("depth", None)
        dropout_rate = network_config.get("dropout_rate", None)
        activation = network_config.get("activation", None)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        self.func = LinearModel(input_dim, hidden_size, depth, dropout_rate, activation).to(self.device)
        self.init_state_dict = copy.deepcopy(self.func.state_dict())

        self.lr = training_config.get("learning_rate", 1e-3)
        self.epoch = training_config.get("epochs", 30)

        self.pair_embedding = torch.empty(2, 0, input_dim, device=self.device)
        self.reward = torch.empty(0, device=self.device, dtype=torch.int64)
        self.len = 0

        print(f"🎯 LinearDuelingBandits algorithm initialization Complete - input dimension: {input_dim}")

    def select_arm(self, items: List, history: Optional[List] = None) -> tuple:
        if len(items) == 0:
            return 0, 0
        if len(items) == 1:
            return 0, 0

        context = torch.stack([torch.tensor(item, device=self.device, dtype=torch.float32) for item in items])

        context_size = context.shape[0]
        batch_size = 300
        n_batchs = context_size // batch_size + int((context_size % batch_size) != 0)
        mu = []
        self.func.eval()

        for i in range(n_batchs):
            if i == n_batchs - 1:
                context_batch = context[(i*batch_size):]
            else:
                context_batch = context[(i*batch_size):((i+1)*batch_size)]

            mu_ = self.func(context_batch)
            mu.append(mu_.cpu())
        mu = torch.vstack(mu)

        greedy_arm = torch.argmax(mu.view(-1)).item()

        if history is not None and len(history) > 0:
            history_tensor = torch.tensor(history)
            grad_1 = context[history_tensor[:, 0]]
            grad_2 = context[history_tensor[:, 1]]
            feature = (grad_1 - grad_2).cpu()

            U = torch.matmul(feature.transpose(0, 1), feature)
            U = U + 1e-10 * torch.eye(U.shape[0])

            grad_arm_1 = context[greedy_arm]
            feature_arm_2 = (context - grad_arm_1).cpu()

            try:
                U_inv = torch.inverse(U)

                uncertainty = torch.sum(feature_arm_2 * (feature_arm_2 @ U_inv), dim=1)
                uncertainty = torch.sqrt(torch.clamp(uncertainty, min=1e-10))
            except:

                U_diag = U.diagonal() + 1e-10
                uncertainty = torch.sqrt(torch.sum(self.nu * feature_arm_2 * feature_arm_2 / U_diag, dim=1))

            uncertainty[greedy_arm] = -float('inf')
            ucb_arm = torch.argmax(uncertainty).item()
        else:
            sorted_idx = torch.argsort(mu.view(-1), descending=True)
            ucb_arm = sorted_idx[1].item()

        print(f"🎯 LinearDuelingBandits selection: greedy_arm {greedy_arm} (predicted: {mu[greedy_arm].item():.4f}), "
              f"exploration_arm {ucb_arm} (predicted: {mu[ucb_arm].item():.4f}, max info gain)")
        return greedy_arm, ucb_arm

    def update(self, arm1_idx: int, arm2_idx: int, preference: int):
        print(f"🎯 LinearDuelingBandits update: arm{arm1_idx} vs arm{arm2_idx}, preference: {preference}")
        pass

    def train_model(self, X1, X2, Y, incremental=False, weights=None):
        print(f"🎯 LinearDuelingBandits: Training model - data size: {Y.shape[0]}")

        if isinstance(X1, np.ndarray):
            X1 = torch.from_numpy(X1).float().to(self.device)
        else:
            X1 = X1.to(self.device)

        if isinstance(X2, np.ndarray):
            X2 = torch.from_numpy(X2).float().to(self.device)
        else:
            X2 = X2.to(self.device)

        if isinstance(Y, np.ndarray):
            Y = torch.from_numpy(Y).float().to(self.device)
        else:
            Y = Y.to(self.device)

        if weights is not None:
            if isinstance(weights, np.ndarray):
                W = torch.from_numpy(weights).float().to(self.device)
            else:
                W = weights.to(self.device)
        else:
            W = torch.ones(Y.shape[0], device=self.device)

        if self.init_state_dict is not None:
            self.func.load_state_dict(copy.deepcopy(self.init_state_dict))

        context = torch.stack([X1, X2], dim=0)
        if self.pair_embedding.shape[1] == 0:
            self.pair_embedding = context.to(self.device)
        else:
            self.pair_embedding = torch.cat((self.pair_embedding, context.to(self.device)), dim=1)
        self.reward = torch.cat((self.reward, Y.to(dtype=torch.int64)))
        self.len = self.pair_embedding.shape[1]

        optimizer = torch.optim.Adam(self.func.parameters(), lr=self.lr)
        self.func.train()

        for _ in range(self.epoch):
            self.func.zero_grad()
            optimizer.zero_grad()

            side_1 = self.pair_embedding[0].reshape(self.len, -1)
            side_2 = self.pair_embedding[1].reshape(self.len, -1)
            pred_1 = self.func(side_1)
            pred_2 = self.func(side_2)
            logits = (pred_1 - pred_2).reshape(-1)
            reward_ = self.reward.reshape(-1)

            per_sample_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, reward_.to(dtype=torch.float32), reduction='none')
            loss = (per_sample_loss * W).sum() / W.sum()

            loss.backward()
            optimizer.step()

        print(f"🎯 LinearDuelingBandits training Complete - final loss: {loss.item():.4f}")

class ENN(nn.Module):
    def __init__(self, input_dim, hidden_size=None, depth=None, dropout_rate=None, activation=None, init_params=None):
        super(ENN, self).__init__()

        try:
            from IDS_TAP_parameters import NETWORK_CONFIG, DOUBLETS_CONFIG
            doublets_network = DOUBLETS_CONFIG.get("network", {})

            default_hidden_size = doublets_network.get("hidden_size") or NETWORK_CONFIG.get("hidden_size", 1024)
            default_depth = doublets_network.get("depth") or NETWORK_CONFIG.get("depth", 1)
            default_dropout_rate = doublets_network.get("dropout_rate") or NETWORK_CONFIG.get("dropout_rate", 0.1)
            default_activation = doublets_network.get("activation") or NETWORK_CONFIG.get("activation", "GELU")

            self.use_kaiming_init = doublets_network.get("use_kaiming_init", True)
            self.ensemble_count = doublets_network.get("ensemble_count", 2)
        except ImportError:
            default_hidden_size = 1024
            default_depth = 1
            default_dropout_rate = 0.1
            default_activation = "GELU"
            self.use_kaiming_init = True
            self.ensemble_count = 2

        hidden_size = hidden_size if hidden_size is not None else default_hidden_size
        depth = depth if depth is not None else default_depth
        dropout_rate = dropout_rate if dropout_rate is not None else default_dropout_rate
        activation = activation if activation is not None else default_activation

        if activation.upper() == "GELU":
            self.activate = nn.GELU()
        else:
            self.activate = nn.ReLU()

        self.dropout = nn.Dropout(p=dropout_rate)

        self.layer_list = nn.ModuleList()
        self.layer_list.append(nn.Linear(input_dim, hidden_size))
        for i in range(depth-1):
            self.layer_list.append(nn.Linear(hidden_size, hidden_size))
        self.layer_list.append(nn.Linear(hidden_size, 1))

        if init_params is None:
            for layer in self.layer_list:
                if self.use_kaiming_init:
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                    nn.init.zeros_(layer.bias)
                else:
                    nn.init.normal_(layer.weight, mean=0, std=1.0)
                    nn.init.normal_(layer.bias, mean=0, std=1.0)
        else:
            for i in range(len(self.layer_list)):
                self.layer_list[i].weight.data = init_params[i*2]
                self.layer_list[i].bias.data = init_params[i*2+1]

        self.layer_list_10 = nn.ModuleList()
        for i in range(self.ensemble_count):
            torch.manual_seed(i + 1)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(i + 1)

            new_module = nn.ModuleList()
            new_module.append(nn.Linear(input_dim, hidden_size))
            for _ in range(depth-1):
                new_module.append(nn.Linear(hidden_size, hidden_size))
            new_module.append(nn.Linear(hidden_size, 1))

            for layer in new_module:
                if self.use_kaiming_init:
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                    nn.init.zeros_(layer.bias)
                else:
                    nn.init.normal_(layer.weight, mean=0, std=1.0)
                    nn.init.normal_(layer.bias, mean=0, std=1.0)

            self.layer_list_10.append(new_module)

        for param in self.layer_list.parameters():
            param.requires_grad = False

        init_method = "kaiming_normal_" if self.use_kaiming_init else "normal_(std=1.0)"
        print(f"🎰 ENN init: ensemble={self.ensemble_count}, init={init_method}, "
              f"dropout={dropout_rate}, hidden={hidden_size}, depth={depth}")

    def forward(self, x, idx):
        y = x
        for i in range(len(self.layer_list_10[idx])-1):
            y = self.activate(self.layer_list_10[idx][i](y))
            y = self.dropout(y)
        y = self.layer_list_10[idx][-1](y)
        return y

class POHFRandomPair:

    def __init__(self, input_dim, config=None):
        if config is None:
            from IDS_TAP_parameters import NETWORK_CONFIG, TRAINING_CONFIG, DEVICE_CONFIG, DATA_CONFIG, POHF_CONFIG
            network_config = NETWORK_CONFIG.copy()
            training_config = TRAINING_CONFIG
            device_config = DEVICE_CONFIG
            data_config = DATA_CONFIG
            pohf_config = POHF_CONFIG
        else:
            network_config = config.get("network", {})
            training_config = config.get("training", {})
            device_config = config.get("device", {})
            data_config = config.get("data", {})
            pohf_config = config.get("pohf", {})

        device_type = device_config.get("device", "cuda:1")
        self.device = torch.device(device_type if torch.cuda.is_available() else "cpu")

        if self.device.type == 'cuda' and device_config.get("clear_cache", True):
            torch.cuda.empty_cache()

        if input_dim is not None:
            self.input_dim = input_dim
        else:
            self.input_dim = get_input_dim_for_dataset(config=config)
        network_config["input_dim"] = self.input_dim

        self.version = pohf_config.get("version", "matrix")
        self.lamb = pohf_config.get("lambda", 1.0)
        self.nu = pohf_config.get("nu", 0.2)

        from ID_TAP import Network
        from backpack import extend
        self.func = extend(Network(self.input_dim, config=network_config)).to(self.device)

        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)
        self.init_model_weight = copy.deepcopy(self.func.state_dict())

        self.lr = training_config.get("learning_rate", 1e-3)
        self.epoch = training_config.get("epochs", 100)
        self.weight_decay = training_config.get("weight_decay", 1.0)

        optimizer_type = training_config.get("optimizer", "AdamW")
        if optimizer_type == "AdamW":
            self.optimizer_fn = optim.AdamW
        elif optimizer_type == "Adam":
            self.optimizer_fn = optim.Adam
        else:
            self.optimizer_fn = optim.AdamW

        self.optimizer = self.optimizer_fn(self.func.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        scheduler_type = training_config.get("scheduler_type", "CosineAnnealingLR")
        min_lr_ratio = training_config.get("min_lr_ratio", 0.01)

        if scheduler_type == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epoch,
                eta_min=self.lr * min_lr_ratio
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epoch,
                eta_min=self.lr * min_lr_ratio
            )

        max_params_for_matrix = pohf_config.get("max_params_for_matrix", 10000)

        if self.total_param > max_params_for_matrix:
            print(f"Warning: Too many parameters ({self.total_param} > {max_params_for_matrix}), forcing diag version to save memory")
            self.version = "diag"

        cov_init_enabled = pohf_config.get("cov_init_enabled", True)
        cov_init_value = pohf_config.get("cov_init_value", 0.01)

        if self.version == "diag":
            if cov_init_enabled:
                self.S = torch.ones(self.total_param, dtype=torch.float32, device=self.device) * cov_init_value
                self.Sinv = 1.0 / self.S.clamp(min=1e-16)
            else:
                self.S = self.lamb * torch.ones(self.total_param, dtype=torch.float32, device=self.device)
                self.Sinv = 1.0 / self.S
        elif self.version == "matrix":
            if self.total_param <= max_params_for_matrix:
                if cov_init_enabled:
                    self.S = torch.eye(self.total_param, dtype=torch.float32, device=self.device) * cov_init_value
                    self.Sinv = torch.eye(self.total_param, dtype=torch.float32, device=self.device) / cov_init_value
                else:
                    self.S = self.lamb * torch.eye(self.total_param, dtype=torch.float32, device=self.device)
                    self.Sinv = torch.inverse(self.S)
            else:
                print(f"Warning: Too many parameters for matrix version, switching to diag")
                self.version = "diag"
                if cov_init_enabled:
                    self.S = torch.ones(self.total_param, dtype=torch.float32, device=self.device) * cov_init_value
                    self.Sinv = 1.0 / self.S.clamp(min=1e-16)
                else:
                    self.S = self.lamb * torch.ones(self.total_param, dtype=torch.float32, device=self.device)
                    self.Sinv = 1.0 / self.S

        print(f"🎯 POHF-RandomPairalgorithm initialization Complete - input dimension: {input_dim}")

    def select_arm(self, items: List, history: Optional[List] = None) -> tuple:
        if len(items) == 0:
            return 0, 0
        if len(items) == 1:
            return 0, 0

        import random
        available_arms = list(range(len(items)))
        random_arm1, random_arm2 = random.sample(available_arms, 2)

        context = torch.stack([torch.tensor(item, device=self.device, dtype=torch.float32) for item in items])
        greedy_scores = self.calculate_scores_only(context.cpu().numpy())
        greedy_arm = torch.argmax(greedy_scores).item()

        del context
        torch.cuda.empty_cache()

        print(f"🎯 POHF-RandomPairselection: random_arm1 {random_arm1}, random_arm2 {random_arm2} (for training), "
              f"greedy_arm {greedy_arm} (score: {greedy_scores[greedy_arm].item():.4f}, for logging)")

        return random_arm1, random_arm2

    def calculate_greedy_score(self, items):
        import copy
        from backpack import backpack
        from backpack.extensions import BatchGrad

        current_state = copy.deepcopy(self.func.state_dict())

        self.func.load_state_dict(self.init_model_weight)
        self.func.eval()

        if isinstance(items, np.ndarray):
            items = torch.from_numpy(items).to(dtype=torch.float32, device=self.device)
        else:
            items = torch.from_numpy(np.array(items)).to(dtype=torch.float32, device=self.device)

        outputs_init = self.func(items)
        outputs_init = torch.sigmoid(outputs_init)

        self.func.zero_grad()
        with backpack(BatchGrad()):
            outputs_init.sum().backward()

        init_grads_batch = torch.cat(
            [p.grad_batch.flatten(1) for p in self.func.parameters() if hasattr(p, 'grad_batch') and p.grad_batch is not None],
            dim=1
        )

        self.func.load_state_dict(current_state)
        self.func.eval()

        with torch.no_grad():
            outputs_current = self.func(items)
            greedy_scores = torch.sigmoid(outputs_current).squeeze()

        return greedy_scores, init_grads_batch

    def calculate_scores_only(self, items):
        try:
            from IDS_TAP_parameters import POHF_CONFIG
            temperature = POHF_CONFIG.get("softmax_temperature", 1.0)
        except ImportError:
            temperature = 1.0

        if isinstance(items, np.ndarray):
            items = torch.from_numpy(items).to(dtype=torch.float32, device=self.device)
        else:
            items = torch.from_numpy(np.array(items)).to(dtype=torch.float32, device=self.device)

        self.func.eval()
        with torch.no_grad():
            outputs = self.func(items)
            outputs = outputs.squeeze()
            outputs = torch.softmax(outputs / temperature, dim=0)
            outputs = outputs.unsqueeze(-1)

        return outputs

    def update(self, arm1_idx: int, arm2_idx: int, preference: int):
        print(f"🎯 POHF-RandomPairupdate: arm{arm1_idx} vs arm{arm2_idx}, preference: {preference}")
        pass

    def restart_model(self, data_size):
        self.func.load_state_dict(self.init_model_weight)
        self.optimizer = self.optimizer_fn(self.func.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _reset_optimizer_only(self):
        from IDS_TAP_parameters import TRAINING_CONFIG

        self.optimizer = self.optimizer_fn(self.func.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        min_lr_ratio = TRAINING_CONFIG.get("min_lr_ratio", 0.1)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epoch, eta_min=self.lr * min_lr_ratio, last_epoch=-1
        )

    def save_query_start_weights(self):
        self._query_start_weights = copy.deepcopy(self.func.state_dict())

    def restore_to_query_start(self):
        if hasattr(self, '_query_start_weights') and self._query_start_weights is not None:
            self.func.load_state_dict(copy.deepcopy(self._query_start_weights))
        self._reset_optimizer_only()

    def train_model(self, X1, X2, Y, incremental=False, reset_to_query_start=False, weights=None):
        from torch.utils.data import TensorDataset, DataLoader
        import torch.nn.functional as F

        if reset_to_query_start:
            self.restore_to_query_start()
        elif incremental:
            self._reset_optimizer_only()
        else:
            self.restart_model(Y.shape[0])

        self.func.train()
        self.func.to(self.device)

        from IDS_TAP_parameters import TRAINING_CONFIG
        batch_size = TRAINING_CONFIG.get("batch_size", 4)
        gradient_clip_norm = TRAINING_CONFIG.get("gradient_clip_norm", 1.0)

        early_stopping = TRAINING_CONFIG.get("early_stopping", False)
        patience = TRAINING_CONFIG.get("early_stopping_patience", 5)
        min_delta = TRAINING_CONFIG.get("early_stopping_min_delta", 1e-4)

        X1_tensor = torch.tensor(X1, dtype=torch.float32)
        X2_tensor = torch.tensor(X2, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.float32)

        if weights is not None:
            W_tensor = torch.tensor(weights, dtype=torch.float32)
        else:
            W_tensor = torch.ones(Y.shape[0], dtype=torch.float32)

        dataset = TensorDataset(X1_tensor, X2_tensor, Y_tensor, W_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_loss, patience_counter = float('inf'), 0

        for epoch_idx in range(self.epoch):
            epoch_loss, batch_count = 0.0, 0
            for batch_X1, batch_X2, batch_Y, batch_W in dataloader:
                batch_X1, batch_X2, batch_Y, batch_W = batch_X1.to(self.device), batch_X2.to(self.device), batch_Y.to(self.device), batch_W.to(self.device)
                self.func.zero_grad()
                self.optimizer.zero_grad()

                pred_1 = self.func(batch_X1)
                pred_2 = self.func(batch_X2)

                per_sample_loss = F.binary_cross_entropy_with_logits(pred_1 - pred_2, batch_Y, reduction='none')
                loss = (per_sample_loss * batch_W).sum() / batch_W.sum()

                loss.backward()
                if gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.func.parameters(), max_norm=gradient_clip_norm)
                self.optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            self.scheduler.step()
            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0

            if early_stopping:
                if avg_loss < best_loss - min_delta:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

class POHFRandom:

    def __init__(self, input_dim, config=None):
        if config is None:
            from IDS_TAP_parameters import NETWORK_CONFIG, TRAINING_CONFIG, DEVICE_CONFIG, DATA_CONFIG, POHF_CONFIG
            network_config = NETWORK_CONFIG.copy()
            training_config = TRAINING_CONFIG
            device_config = DEVICE_CONFIG
            data_config = DATA_CONFIG
            pohf_config = POHF_CONFIG
        else:
            network_config = config.get("network", {})
            training_config = config.get("training", {})
            device_config = config.get("device", {})
            data_config = config.get("data", {})
            pohf_config = config.get("pohf", {})

        device_type = device_config.get("device", "cuda:1")
        self.device = torch.device(device_type if torch.cuda.is_available() else "cpu")

        if self.device.type == 'cuda' and device_config.get("clear_cache", True):
            torch.cuda.empty_cache()

        if input_dim is not None:
            self.input_dim = input_dim
        else:
            self.input_dim = get_input_dim_for_dataset(config=config)
        network_config["input_dim"] = self.input_dim

        self.version = pohf_config.get("version", "matrix")
        self.lamb = pohf_config.get("lambda", 1.0)
        self.nu = pohf_config.get("nu", 0.2)

        from ID_TAP import Network
        from backpack import extend
        self.func = extend(Network(self.input_dim, config=network_config)).to(self.device)

        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)
        self.init_model_weight = copy.deepcopy(self.func.state_dict())

        self.lr = training_config.get("learning_rate", 1e-3)
        self.epoch = training_config.get("epochs", 100)
        self.weight_decay = training_config.get("weight_decay", 1.0)

        optimizer_type = training_config.get("optimizer", "AdamW")
        if optimizer_type == "AdamW":
            self.optimizer_fn = optim.AdamW
        elif optimizer_type == "Adam":
            self.optimizer_fn = optim.Adam
        else:
            self.optimizer_fn = optim.AdamW

        self.optimizer = self.optimizer_fn(self.func.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        scheduler_type = training_config.get("scheduler_type", "CosineAnnealingLR")
        min_lr_ratio = training_config.get("min_lr_ratio", 0.01)

        if scheduler_type == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epoch,
                eta_min=self.lr * min_lr_ratio
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epoch,
                eta_min=self.lr * min_lr_ratio
            )

        print(f"🎯 POHF-Randomalgorithm initialization Complete - input dimension: {input_dim}")

    def select_arm(self, items: List, history: Optional[List] = None) -> tuple:
        if len(items) == 0:
            return 0, 0
        if len(items) == 1:
            return 0, 0

        context = torch.stack([torch.tensor(item, device=self.device, dtype=torch.float32) for item in items])
        greedy_scores = self.calculate_scores_only(context.cpu().numpy())

        del context
        torch.cuda.empty_cache()

        greedy_arm = torch.argmax(greedy_scores).item()

        available_arms = list(range(len(items)))
        random_arm = random.choice(available_arms)

        print(f"🎯 POHF-Randomselection: greedy_arm {greedy_arm} (score: {greedy_scores[greedy_arm].item():.4f}), "
              f"random_arm {random_arm} (random selection)")
        return greedy_arm, random_arm

    def calculate_scores_only(self, items):
        try:
            from IDS_TAP_parameters import POHF_CONFIG
            temperature = POHF_CONFIG.get("softmax_temperature", 1.0)
        except ImportError:
            temperature = 1.0

        if isinstance(items, np.ndarray):
            items = torch.from_numpy(items).to(dtype=torch.float32, device=self.device)
        else:
            items = torch.from_numpy(np.array(items)).to(dtype=torch.float32, device=self.device)

        self.func.eval()
        with torch.no_grad():
            outputs = self.func(items)
            outputs = outputs.squeeze()
            outputs = torch.softmax(outputs / temperature, dim=0)
            outputs = outputs.unsqueeze(-1)

        return outputs

    def restart_model(self, N):
        self.func.load_state_dict(copy.deepcopy(self.init_model_weight))

        from IDS_TAP_parameters import TRAINING_CONFIG
        fixed_weight_decay = self.weight_decay

        self.optimizer = self.optimizer_fn(
            self.func.parameters(),
            lr=self.lr,
            weight_decay=fixed_weight_decay
        )

        scheduler_type = TRAINING_CONFIG.get("scheduler_type", "CosineAnnealingLR")
        min_lr_ratio = TRAINING_CONFIG.get("min_lr_ratio", 0.01)

        if scheduler_type == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epoch,
                eta_min=self.lr * min_lr_ratio
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epoch,
                eta_min=self.lr * min_lr_ratio
            )

    def _reset_optimizer_only(self):
        from IDS_TAP_parameters import TRAINING_CONFIG
        self.optimizer = self.optimizer_fn(self.func.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        min_lr_ratio = TRAINING_CONFIG.get("min_lr_ratio", 0.1)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epoch, eta_min=self.lr * min_lr_ratio, last_epoch=-1
        )

    def save_query_start_weights(self):
        self._query_start_weights = copy.deepcopy(self.func.state_dict())

    def restore_to_query_start(self):
        if hasattr(self, '_query_start_weights') and self._query_start_weights is not None:
            self.func.load_state_dict(copy.deepcopy(self._query_start_weights))
        self._reset_optimizer_only()

    def train_model(self, X1, X2, Y, incremental=False, reset_to_query_start=False, weights=None):
        from torch.utils.data import TensorDataset, DataLoader
        import torch.nn.functional as F

        if reset_to_query_start:
            self.restore_to_query_start()
        elif incremental:
            self._reset_optimizer_only()
        else:
            self.restart_model(Y.shape[0])

        self.func.train()
        self.func.to(self.device)

        from IDS_TAP_parameters import TRAINING_CONFIG
        batch_size = TRAINING_CONFIG.get("batch_size", 32)
        gradient_clip_norm = TRAINING_CONFIG.get("gradient_clip_norm", 1.0)

        early_stopping = TRAINING_CONFIG.get("early_stopping", False)
        patience = TRAINING_CONFIG.get("early_stopping_patience", 5)
        min_delta = TRAINING_CONFIG.get("early_stopping_min_delta", 1e-4)

        X1_tensor = torch.tensor(X1, dtype=torch.float32)
        X2_tensor = torch.tensor(X2, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.float32)

        if weights is not None:
            W_tensor = torch.tensor(weights, dtype=torch.float32)
        else:
            W_tensor = torch.ones(Y.shape[0], dtype=torch.float32)

        dataset = TensorDataset(X1_tensor, X2_tensor, Y_tensor, W_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_loss, patience_counter = float('inf'), 0

        for epoch in range(1, self.epoch + 1):
            epoch_loss, batch_count = 0.0, 0

            for batch_X1, batch_X2, batch_Y, batch_W in dataloader:
                batch_X1, batch_X2, batch_Y, batch_W = batch_X1.to(self.device), batch_X2.to(self.device), batch_Y.to(self.device), batch_W.to(self.device)
                self.func.zero_grad()
                self.optimizer.zero_grad()

                score_1 = self.func(batch_X1)
                score_2 = self.func(batch_X2)

                per_sample_loss = F.binary_cross_entropy_with_logits(score_1 - score_2, batch_Y, reduction='none')
                loss = (per_sample_loss * batch_W).sum() / batch_W.sum()

                loss.backward()
                if gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.func.parameters(), max_norm=gradient_clip_norm)
                self.optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            self.scheduler.step()
            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0

            if early_stopping:
                if avg_loss < best_loss - min_delta:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

    def update(self, arm1_idx: int, arm2_idx: int, preference: int):
        print(f"🎯 POHF-Randomupdate: arm{arm1_idx} vs arm{arm2_idx}, preference: {preference}")
        pass

class DoubleTS:

    def __init__(self, input_dim, config=None):
        if config is None:
            from IDS_TAP_parameters import NETWORK_CONFIG, TRAINING_CONFIG, DEVICE_CONFIG, DATA_CONFIG, DOUBLETS_CONFIG
            network_config = NETWORK_CONFIG.copy()
            training_config = TRAINING_CONFIG
            device_config = DEVICE_CONFIG
            data_config = DATA_CONFIG
            doublets_config = DOUBLETS_CONFIG
        else:
            network_config = config.get("network", {})
            training_config = config.get("training", {})
            device_config = config.get("device", {})
            data_config = config.get("data", {})
            doublets_config = config.get("doublets", {})

        device_type = device_config.get("device", "cuda:1")
        self.device = torch.device(device_type if torch.cuda.is_available() else "cpu")

        if self.device.type == 'cuda' and device_config.get("clear_cache", True):
            torch.cuda.empty_cache()
            print(f"DoubleTS - Using device: {self.device}, cleared CUDA cache")

        if input_dim is not None:
            self.input_dim = input_dim
        else:
            self.input_dim = get_input_dim_for_dataset(config=config)
            dataset_type = "LaMP(contextual)" if is_lamp_dataset() else "Standard"
            print(f"🎰 Using{dataset_type}mode embedding dimension: {self.input_dim}")

        network_config["input_dim"] = self.input_dim

        self.ensemble_lambda = doublets_config.get("ensemble_lambda", 0.1)
        self.weight_decay = doublets_config.get("weight_decay", 0.05)
        self.use_adamw = doublets_config.get("use_adamw", True)

        hidden_size = network_config.get("hidden_size", None)
        depth = network_config.get("depth", None)
        dropout_rate = network_config.get("dropout_rate", None)
        activation = network_config.get("activation", None)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        self.func = ENN(input_dim, hidden_size, depth, dropout_rate, activation).to(self.device)
        self.init_state_dict = copy.deepcopy(self.func.state_dict())

        self.lr = training_config.get("learning_rate", 1e-3)
        self.epoch = training_config.get("epochs", 30)

        self.pair_embedding = torch.empty(2, 0, input_dim, device=self.device)
        self.reward = torch.empty(0, device=self.device, dtype=torch.int64)
        self.len = 0

        print(f"🎰 DoubleTSalgorithm initialization Complete - input dimension: {input_dim}, weight_decay: {self.weight_decay}, use_adamw: {self.use_adamw}")

    def select_arm(self, items: List, history: Optional[List] = None) -> tuple:
        if len(items) == 0:
            return 0, 0
        if len(items) == 1:
            return 0, 0

        context = torch.stack([torch.tensor(item, device=self.device, dtype=torch.float32) for item in items])
        context_size = context.shape[0]

        num_ensembles = len(self.func.layer_list_10)

        self.func.eval()

        with torch.no_grad():
            def compute_scores(epi_idx):
                batch_size = 300
                n_batchs = context_size // batch_size + int((context_size % batch_size) != 0)
                mu = []
                for i in range(n_batchs):
                    if i == n_batchs - 1:
                        context_batch = context[(i*batch_size):]
                    else:
                        context_batch = context[(i*batch_size):((i+1)*batch_size)]
                    mu_ = self.func(context_batch, epi_idx)
                    mu.append(mu_.cpu())
                return torch.vstack(mu).view(-1)

            epi_idx_1 = torch.randint(0, num_ensembles, (1,)).item()
            mu_1 = compute_scores(epi_idx_1)
            arm1 = torch.argmax(mu_1).item()

            epi_idx_2 = torch.randint(0, num_ensembles, (1,)).item()
            mu_2 = compute_scores(epi_idx_2)
            arm2 = torch.argmax(mu_2).item()

            if arm1 == arm2 and context_size >= 2:
                top2_indices = torch.topk(mu_2, k=2).indices
                arm2 = top2_indices[1].item()

        score1 = mu_1[arm1].item()
        score2 = mu_2[arm2].item()
        print(f"🎰 DoubleTSselection: arm1={arm1} (ensemble {epi_idx_1}), arm2={arm2} (ensemble {epi_idx_2})")
        print(f"   📊 model predicted scores: arm{arm1}={score1:.4f}, arm{arm2}={score2:.4f}")

        return arm1, arm2

    def update(self, arm1_idx: int, arm2_idx: int, preference: int):
        print(f"🎰 DoubleTSupdate: arm{arm1_idx} vs arm{arm2_idx}, preference: {preference}")
        pass

    def restart_model(self):
        if self.init_state_dict is not None:
            self.func.load_state_dict(copy.deepcopy(self.init_state_dict))

    def _reset_optimizer_only(self):
        pass

    def save_query_start_weights(self):
        self._query_start_weights = copy.deepcopy(self.func.state_dict())

    def restore_to_query_start(self):
        if hasattr(self, '_query_start_weights') and self._query_start_weights is not None:
            self.func.load_state_dict(copy.deepcopy(self._query_start_weights))

    def train_model(self, X1, X2, Y, incremental=False, reset_to_query_start=False, weights=None):
        if isinstance(X1, np.ndarray):
            X1 = torch.from_numpy(X1).float().to(self.device)
        else:
            X1 = X1.to(self.device)

        if isinstance(X2, np.ndarray):
            X2 = torch.from_numpy(X2).float().to(self.device)
        else:
            X2 = X2.to(self.device)

        if isinstance(Y, np.ndarray):
            Y = torch.from_numpy(Y).float().to(self.device)
        else:
            Y = Y.to(self.device)

        if reset_to_query_start:
            self.restore_to_query_start()
        elif incremental:
            pass
        else:
            self.restart_model()

        from IDS_TAP_parameters import TRAINING_CONFIG
        batch_size = TRAINING_CONFIG.get("batch_size", 8)
        gradient_clip_norm = TRAINING_CONFIG.get("gradient_clip_norm", 100.0)

        early_stopping = TRAINING_CONFIG.get("early_stopping", False)
        patience = TRAINING_CONFIG.get("early_stopping_patience", 5)
        min_delta = TRAINING_CONFIG.get("early_stopping_min_delta", 1e-4)

        ensemble_lambda = self.ensemble_lambda

        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(X1, X2, Y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        if self.use_adamw:
            optimizer = torch.optim.AdamW(self.func.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.Adam(self.func.parameters(), lr=self.lr)
        self.func.train()

        best_loss, patience_counter = float('inf'), 0

        for epoch_idx in range(self.epoch):
            epoch_loss, batch_count = 0.0, 0

            for batch_X1, batch_X2, batch_Y in dataloader:
                num_ensembles = len(self.func.layer_list_10)
                for epi_idx in range(num_ensembles):
                    self.func.zero_grad()
                    optimizer.zero_grad()

                    pred_1 = self.func(batch_X1, epi_idx)
                    pred_2 = self.func(batch_X2, epi_idx)

                    import torch.nn.functional as F
                    ce_loss = F.binary_cross_entropy_with_logits((pred_1 - pred_2).squeeze(-1), batch_Y)

                    if ensemble_lambda > 0:
                        l2_reg_term = 0
                        for param_ens, param_main in zip(self.func.layer_list_10[epi_idx], self.func.layer_list):
                            l2_reg_term += torch.sum((param_ens.weight - param_main.weight) ** 2)
                            l2_reg_term += torch.sum((param_ens.bias - param_main.bias) ** 2)
                        loss = ce_loss + ensemble_lambda * l2_reg_term
                    else:
                        loss = ce_loss

                    loss.backward()

                    if gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.func.parameters(), max_norm=gradient_clip_norm)
                    optimizer.step()

                    epoch_loss += loss.item()
                    batch_count += 1

            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0

            if early_stopping:
                if avg_loss < best_loss - min_delta:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

class PureLinearModel(nn.Module):
    def __init__(self, input_dim):
        super(PureLinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        torch.nn.init.normal_(self.linear.weight, mean=0, std=0.01)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x).squeeze(-1)

class LinearInfoGain:

    def __init__(self, input_dim, config=None):
        if config is None:
            from IDS_TAP_parameters import TRAINING_CONFIG, DEVICE_CONFIG, DATA_CONFIG, POHF_CONFIG
            training_config = TRAINING_CONFIG
            device_config = DEVICE_CONFIG
            data_config = DATA_CONFIG
            pohf_config = POHF_CONFIG
        else:
            training_config = config.get("training", {})
            device_config = config.get("device", {})
            data_config = config.get("data", {})
            pohf_config = config.get("pohf", {})

        device_type = device_config.get("device", "cuda:1")
        self.device = torch.device(device_type if torch.cuda.is_available() else "cpu")

        if self.device.type == 'cuda' and device_config.get("clear_cache", True):
            torch.cuda.empty_cache()

        if input_dim is not None:
            self.input_dim = input_dim
        else:
            self.input_dim = get_input_dim_for_dataset(config=config)

        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(0)
        self.func = PureLinearModel(self.input_dim).to(self.device)
        self.init_state_dict = copy.deepcopy(self.func.state_dict())

        self.lr = training_config.get("learning_rate", 1e-3)
        self.epoch = training_config.get("epochs", 100)
        self.weight_decay = training_config.get("weight_decay", 1.0)

        optimizer_type = training_config.get("optimizer", "AdamW")
        if optimizer_type == "AdamW":
            self.optimizer_fn = optim.AdamW
        elif optimizer_type == "Adam":
            self.optimizer_fn = optim.Adam
        else:
            self.optimizer_fn = optim.AdamW

        self.bayesian_alpha = pohf_config.get("bayesian_alpha", 1.0)

        self.info_manager = None
        self.contextual_info_manager = None

        print(f"🎯 Linear-InfoGainalgorithm initialization Complete - input dimension: {self.input_dim}, device: {self.device}")

    def calculate_scores_only(self, items):
        try:
            from IDS_TAP_parameters import POHF_CONFIG
            temperature = POHF_CONFIG.get("softmax_temperature", 1.0)
        except ImportError:
            temperature = 1.0

        if isinstance(items, np.ndarray):
            items = torch.from_numpy(items).to(dtype=torch.float32, device=self.device)
        else:
            items = torch.from_numpy(np.array(items)).to(dtype=torch.float32, device=self.device)

        self.func.eval()
        with torch.no_grad():
            outputs = self.func(items)
            outputs = outputs.squeeze()
            outputs = torch.softmax(outputs / temperature, dim=0)
            outputs = outputs.unsqueeze(-1)

        return outputs

    def save_query_start_weights(self):
        self._query_start_weights = copy.deepcopy(self.func.state_dict())

    def restore_to_query_start(self):
        if hasattr(self, '_query_start_weights') and self._query_start_weights is not None:
            self.func.load_state_dict(copy.deepcopy(self._query_start_weights))

    def _reset_optimizer(self):
        from IDS_TAP_parameters import TRAINING_CONFIG
        min_lr_ratio = TRAINING_CONFIG.get("min_lr_ratio", 0.01)

        self.optimizer = self.optimizer_fn(
            self.func.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epoch,
            eta_min=self.lr * min_lr_ratio,
            last_epoch=-1
        )

    def train_model(self, X1, X2, Y, incremental=False, reset_to_query_start=False, weights=None):
        if reset_to_query_start:
            self.restore_to_query_start()
        elif not incremental:
            self.func.load_state_dict(copy.deepcopy(self.init_state_dict))

        self._reset_optimizer()
        self.func.train()

        from IDS_TAP_parameters import TRAINING_CONFIG
        batch_size = TRAINING_CONFIG.get("batch_size", 32)
        gradient_clip_norm = TRAINING_CONFIG.get("gradient_clip_norm", 1.0)
        early_stopping = TRAINING_CONFIG.get("early_stopping", False)
        patience = TRAINING_CONFIG.get("early_stopping_patience", 5)
        min_delta = TRAINING_CONFIG.get("early_stopping_min_delta", 1e-4)

        X1_tensor = torch.tensor(X1, dtype=torch.float32).to(self.device)
        X2_tensor = torch.tensor(X2, dtype=torch.float32).to(self.device)
        Y_tensor = torch.tensor(Y, dtype=torch.float32).to(self.device)

        if weights is not None:
            W_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)
        else:
            W_tensor = torch.ones(Y.shape[0], dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(X1_tensor, X2_tensor, Y_tensor, W_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, self.epoch + 1):
            epoch_loss = 0.0
            batch_count = 0

            for batch_X1, batch_X2, batch_Y, batch_W in dataloader:
                self.func.zero_grad()
                self.optimizer.zero_grad()

                score_1 = self.func(batch_X1)
                score_2 = self.func(batch_X2)

                per_sample_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    score_1 - score_2, batch_Y, reduction='none'
                )
                loss = (per_sample_loss * batch_W).sum() / batch_W.sum()

                loss.backward()

                if gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.func.parameters(), max_norm=gradient_clip_norm)

                self.optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            self.scheduler.step()
            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0

            if early_stopping:
                if avg_loss < best_loss - min_delta:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

def create_algorithm(algorithm_name: str, input_dim: int = None, config=None):
    algorithms = {
        "Random": Random,
        "LinearDuelingBandits": LinearDuelingBandits,
        "DoubleTS": DoubleTS,
        "POHFRandom": POHFRandom,
        "POHFRandomPair": POHFRandomPair,
        "LinearInfoGain": LinearInfoGain
    }

    if algorithm_name not in algorithms:
        raise ValueError(f"Unknown algorithm: {algorithm_name}. Available: {list(algorithms.keys())}")

    if input_dim is None and algorithm_name != "Random":
        from IDS_TAP_parameters import DATA_CONFIG
        input_dim = DATA_CONFIG.get("embedding_max_dim", 1024)
        print(f"🔧 for{algorithm_name}UsingPOHFconfigurationembedding dimension: {input_dim}")

    return algorithms[algorithm_name](input_dim, config)

if __name__ == "__main__":
    print("🧪 Starting testLLM regression algorithms...")

    input_dim = 128
    algorithms_to_test = ["Random", "LinearDuelingBandits", "DoubleTS"]

    for alg_name in algorithms_to_test:
        print(f"\n🔍 Testing {alg_name}...")
        try:
            alg = create_algorithm(alg_name, input_dim)
            print(f"✅ {alg_name} created successfully")

            items = [torch.randn(input_dim) for _ in range(5)]

            arm1, arm2 = alg.select_arm(items)
            print(f"✅ {alg_name} selection test passed: arm1={arm1}, arm2={arm2}")

            alg.update(arm1, arm2, 1)
            print(f"✅ {alg_name} update test passed")

        except Exception as e:
            print(f"❌ {alg_name} test failed: {e}")

    print(f"\n🎉 LLM regression algorithmsTest Complete!")
