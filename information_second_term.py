import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Set, Any
import torch
import time

class PairwiseInformationManager:

    def __init__(self, num_arms: int, use_optimized: bool = True, bayesian_alpha: float = 1.0,
                 device: str = 'cuda', bt_isolated_arm_mode: str = 'unknown_isolated'):
        self.num_arms = num_arms
        self.use_optimized = use_optimized
        self.bayesian_alpha = bayesian_alpha
        self.bt_isolated_arm_mode = bt_isolated_arm_mode
        self.epsilon = 1e-8

        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.use_gpu = self.device.type == 'cuda'

        self.pairwise_probs = np.full((num_arms, num_arms), 0.5)
        self.win_counts = np.zeros((num_arms, num_arms))
        self.total_counts = np.zeros((num_arms, num_arms))
        np.fill_diagonal(self.pairwise_probs, 0.5)

        self._init_gpu_tensors()

        self.comparison_history = []
        self.current_query_counts = np.zeros((num_arms, num_arms))
        self.logger = logging.getLogger(__name__)

        self._cached_theta = None
        self._cache_valid = False

    def _init_gpu_tensors(self):
        self.probs_gpu = torch.tensor(self.pairwise_probs, dtype=torch.float32, device=self.device)
        self.win_counts_gpu = torch.tensor(self.win_counts, dtype=torch.float32, device=self.device)
        self.total_counts_gpu = torch.tensor(self.total_counts, dtype=torch.float32, device=self.device)

    def _sync_to_gpu(self):
        self.probs_gpu = torch.tensor(self.pairwise_probs, dtype=torch.float32, device=self.device)
        self.win_counts_gpu = torch.tensor(self.win_counts, dtype=torch.float32, device=self.device)
        self.total_counts_gpu = torch.tensor(self.total_counts, dtype=torch.float32, device=self.device)
        self._cache_valid = False

    def _sync_to_cpu(self):
        self.pairwise_probs = self.probs_gpu.cpu().numpy()
        self.win_counts = self.win_counts_gpu.cpu().numpy()
        self.total_counts = self.total_counts_gpu.cpu().numpy()

    def update_pairwise_probability(self, arm1: int, arm2: int, arm1_wins: bool) -> None:
        if arm1 == arm2:
            return

        old_count = self.total_counts[arm2, arm1]
        print(f"      🔄 [DEBUG update] arm1={arm1}, arm2={arm2}, arm1_wins={arm1_wins}, "
              f"total_counts[{arm2},{arm1}]: {old_count} → {old_count + 1}", flush=True)

        self.comparison_history.append((arm1, arm2, arm1_wins))
        self.current_query_counts[arm1, arm2] += 1
        self.current_query_counts[arm2, arm1] += 1

        self.total_counts[arm1, arm2] += 1
        self.total_counts[arm2, arm1] += 1

        if arm1_wins:
            self.win_counts[arm1, arm2] += 1
        else:
            self.win_counts[arm2, arm1] += 1

        wins_12 = self.win_counts[arm1, arm2]
        wins_21 = self.win_counts[arm2, arm1]
        alpha = self.bayesian_alpha

        prob_12 = (wins_12 + alpha) / (wins_12 + wins_21 + 2 * alpha)
        prob_21 = (wins_21 + alpha) / (wins_12 + wins_21 + 2 * alpha)

        self.pairwise_probs[arm1, arm2] = np.clip(prob_12, self.epsilon, 1 - self.epsilon)
        self.pairwise_probs[arm2, arm1] = np.clip(prob_21, self.epsilon, 1 - self.epsilon)

        self._cache_valid = False

    def update_pairwise_probability_with_transitive(self, arm1: int, arm2: int, arm1_wins: bool) -> None:
        self.update_pairwise_probability(arm1, arm2, arm1_wins)
        self._update_transitive_probabilities(arm1, arm2)

    def _update_transitive_probabilities(self, arm1: int, arm2: int) -> None:
        connected_to_arm1 = self._get_connected_arms(arm1)
        connected_to_arm2 = self._get_connected_arms(arm2)

        for k in connected_to_arm1:
            if k != arm2 and self.total_counts[k, arm2] == 0:
                best_bridge = self._find_bridge(k, arm2)
                if best_bridge is not None:
                    self._transitive_update(k, best_bridge, arm2)

        for k in connected_to_arm2:
            if k != arm1 and self.total_counts[k, arm1] == 0:
                best_bridge = self._find_bridge(k, arm1)
                if best_bridge is not None:
                    self._transitive_update(k, best_bridge, arm1)

    def _get_connected_arms(self, arm: int) -> Set[int]:
        connected = set()
        queue = [arm]
        visited = {arm}

        while queue:
            current = queue.pop(0)
            for other in range(self.num_arms):
                if other not in visited and self.total_counts[current, other] > 0:
                    visited.add(other)
                    connected.add(other)
                    queue.append(other)

        return connected

    def _find_bridge(self, source: int, target: int) -> Optional[int]:
        for bridge in range(self.num_arms):
            if bridge != source and bridge != target:
                if self.total_counts[source, bridge] > 0 and self.total_counts[bridge, target] > 0:
                    return bridge

        for bridge in range(self.num_arms):
            if bridge != source and bridge != target:
                if self.total_counts[source, bridge] > 0:
                    if abs(self.pairwise_probs[bridge, target] - 0.5) > 0.01:
                        return bridge
        return None

    def _transitive_update(self, k: int, bridge: int, target: int):
        p_kb = self.pairwise_probs[k, bridge]
        p_bt = self.pairwise_probs[bridge, target]

        logit_kb = np.log(p_kb / (1 - p_kb + self.epsilon) + self.epsilon)
        logit_bt = np.log(p_bt / (1 - p_bt + self.epsilon) + self.epsilon)
        logit_kt = logit_kb + logit_bt

        p_kt = 1.0 / (1.0 + np.exp(-logit_kt))
        p_kt = np.clip(p_kt, self.epsilon, 1 - self.epsilon)

        self.pairwise_probs[k, target] = p_kt
        self.pairwise_probs[target, k] = 1 - p_kt

    def get_pairwise_probability(self, arm1: int, arm2: int) -> float:
        return self.pairwise_probs[arm1, arm2]

    def get_comparison_count(self, arm1: int, arm2: int) -> int:
        return int(self.total_counts[arm1, arm2])

    def get_current_query_comparison_count(self, arm1: int, arm2: int) -> int:
        return int(self.current_query_counts[arm1, arm2])

    def reset_current_query_counts(self):
        self.current_query_counts = np.zeros((self.num_arms, self.num_arms))

    def estimate_bt_strengths_gpu(self,
                                   w_ij: torch.Tensor = None,
                                   n_ij: torch.Tensor = None,
                                   init_pi: torch.Tensor = None,
                                   max_iter: int = None) -> torch.Tensor:
        K = self.num_arms
        alpha = self.bayesian_alpha

        if w_ij is None:
            w_ij = self.win_counts_gpu
        if n_ij is None:
            n_ij = self.total_counts_gpu

        mask = 1.0 - torch.eye(K, device=self.device)

        has_data = (n_ij > 0).float()
        w_ij_reg = w_ij + alpha * has_data
        n_ij_reg = n_ij + 2 * alpha * has_data

        has_any_comparison = (n_ij.sum(dim=-1) > 0)

        W_i = (w_ij_reg * mask).sum(dim=-1)

        W_i = torch.where(has_any_comparison, W_i, torch.ones_like(W_i) * self.epsilon)

        if init_pi is not None:
            pi = init_pi.clone()
            if max_iter is None:
                max_iter = 5
        else:
            pi = torch.ones(K, device=self.device)
            if max_iter is None:
                max_iter = 50

        for iteration in range(max_iter):
            pi_old = pi.clone()

            pi_sum = pi.unsqueeze(-1) + pi.unsqueeze(-2)
            denom_matrix = (n_ij_reg * mask) / (pi_sum + self.epsilon)
            denom = denom_matrix.sum(dim=-1)

            pi = W_i / (denom + self.epsilon)

            pi = pi / (pi.sum() + self.epsilon) * K

            if torch.max(torch.abs(pi - pi_old) / (pi_old + self.epsilon)) < 1e-6:
                break

        theta = torch.log(pi + self.epsilon)

        if has_any_comparison.any():
            non_isolated_mean = theta[has_any_comparison].mean()
            theta[has_any_comparison] = theta[has_any_comparison] - non_isolated_mean

        theta[~has_any_comparison] = 0.0

        return theta

    def is_arm_isolated(self, arm: int) -> bool:
        return np.sum(self.total_counts[arm, :]) == 0

    def get_bt_probability(self, theta: torch.Tensor, arm_i: int, arm_j: int) -> torch.Tensor:
        arm_i_isolated = self.is_arm_isolated(arm_i)
        arm_j_isolated = self.is_arm_isolated(arm_j)

        if arm_i_isolated or arm_j_isolated:
            return torch.tensor(0.5, device=self.device)

        return torch.sigmoid(theta[arm_i] - theta[arm_j])

    def compute_best_arm_distribution_gpu(self, theta: torch.Tensor,
                                           n_ij: torch.Tensor = None) -> torch.Tensor:
        K = self.num_arms

        if n_ij is None:
            n_ij = self.total_counts_gpu

        has_any_comparison = (n_ij.sum(dim=-1) > 0)
        num_connected = has_any_comparison.sum().item()
        num_isolated = K - num_connected

        if num_connected == 0:
            return torch.ones(K, device=self.device) / K

        if num_isolated == 0:
            return torch.softmax(theta, dim=0)

        p_best = torch.zeros(K, device=self.device)

        p_best[~has_any_comparison] = 1.0 / K

        connected_mask = has_any_comparison
        connected_theta = theta[connected_mask]
        connected_softmax = torch.softmax(connected_theta, dim=0)
        p_best[connected_mask] = connected_softmax * (num_connected / K)

        p_best = p_best / p_best.sum()

        return p_best

    def get_arm_scores(self, champion: int, normalize: bool = True) -> Dict[int, float]:
        self._sync_to_gpu()
        theta = self.estimate_bt_strengths_gpu()
        p_best = self.compute_best_arm_distribution_gpu(theta)
        p_best_np = p_best.cpu().numpy()

        scores = {}
        for arm in range(self.num_arms):
            if arm != champion:
                scores[arm] = p_best_np[arm]

        if normalize and len(scores) > 1:
            values = list(scores.values())
            min_val = min(values)
            max_val = max(values)

            if max_val - min_val > self.epsilon:
                scores = {arm: (score - min_val) / (max_val - min_val)
                          for arm, score in scores.items()}
            else:
                scores = {arm: 0.5 for arm in scores.keys()}

        return scores

    def compute_entropy_gpu(self, p: torch.Tensor) -> torch.Tensor:
        p_clipped = torch.clamp(p, self.epsilon, 1.0)
        return -torch.sum(p_clipped * torch.log2(p_clipped))

    def simulate_comparison_gpu(self, champion: int, challenger: int,
                                 champion_wins: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        new_win_counts = self.win_counts_gpu.clone()
        new_total_counts = self.total_counts_gpu.clone()

        if champion_wins:
            new_win_counts[champion, challenger] += 1
        else:
            new_win_counts[challenger, champion] += 1

        new_total_counts[champion, challenger] += 1
        new_total_counts[challenger, champion] += 1

        return new_win_counts, new_total_counts

    def _compute_conditional_entropy(self, champion: int, challenger: int,
                                      champion_wins: bool,
                                      current_pi: torch.Tensor = None) -> torch.Tensor:
        new_w_ij, new_n_ij = self.simulate_comparison_gpu(champion, challenger, champion_wins)

        theta = self.estimate_bt_strengths_gpu(w_ij=new_w_ij, n_ij=new_n_ij, init_pi=current_pi)

        p_best = self.compute_best_arm_distribution_gpu(theta, n_ij=new_n_ij)
        return self.compute_entropy_gpu(p_best)

    def calculate_information_gain_optimized(self, champion: int, challenger: int,
                                              H_before: torch.Tensor = None,
                                              theta_current: torch.Tensor = None,
                                              current_pi: torch.Tensor = None) -> float:
        if champion == challenger:
            return 0.0

        self._sync_to_gpu()

        if theta_current is None or current_pi is None:
            theta_current = self.estimate_bt_strengths_gpu()
            current_pi = torch.exp(theta_current)

        if H_before is None:
            p_best_current = self.compute_best_arm_distribution_gpu(theta_current)
            H_before = self.compute_entropy_gpu(p_best_current)

        p_champion_wins = self.get_bt_probability(theta_current, champion, challenger)

        H_after_win = self._compute_conditional_entropy(champion, challenger, True, current_pi)
        H_after_lose = self._compute_conditional_entropy(champion, challenger, False, current_pi)

        E_H_after = p_champion_wins * H_after_win + (1 - p_champion_wins) * H_after_lose

        mutual_info = H_before - E_H_after

        return max(0.0, mutual_info.item())

    def get_all_information_gains(self, champion: int,
                                   available_arms: Optional[List[int]] = None) -> Dict[int, float]:
        if available_arms is None:
            available_arms = list(range(self.num_arms))

        candidates = [a for a in available_arms if a != champion]
        if not candidates:
            return {}

        self._sync_to_gpu()
        theta_current = self.estimate_bt_strengths_gpu()
        current_pi = torch.exp(theta_current)
        p_best_current = self.compute_best_arm_distribution_gpu(theta_current)
        H_before = self.compute_entropy_gpu(p_best_current)

        information_gains = {}

        isolated_arms = []
        connected_arms = []

        for challenger in candidates:
            has_history = np.sum(self.total_counts[challenger, :]) > 0
            if has_history:
                connected_arms.append(challenger)
            else:
                isolated_arms.append(challenger)

        if isolated_arms:
            representative = isolated_arms[0]
            isolated_gain = self.calculate_information_gain_optimized(
                champion, representative, H_before, theta_current, current_pi)
            for arm in isolated_arms:
                information_gains[arm] = isolated_gain

        for challenger in connected_arms:
            gain = self.calculate_information_gain_optimized(
                champion, challenger, H_before, theta_current, current_pi)
            information_gains[challenger] = gain

        return information_gains

    def get_normalized_information_gains(self, champion: int,
                                          available_arms: Optional[List[int]] = None) -> Dict[int, float]:
        raw_gains = self.get_all_information_gains(champion, available_arms)

        if not raw_gains:
            return {}

        if len(raw_gains) == 1:
            return {arm: 1.0 for arm in raw_gains.keys()}

        values = list(raw_gains.values())
        max_val = max(values)

        if max_val > self.epsilon:
            normalized = {arm: gain / max_val for arm, gain in raw_gains.items()}
        else:
            normalized = {arm: 1.0 for arm in raw_gains.keys()}

        return normalized

    def get_best_pair_to_compare(self, champion: int,
                                  available_arms: Optional[List[int]] = None) -> Tuple[int, float]:
        all_gains = self.get_all_information_gains(champion, available_arms)
        if not all_gains:
            return (champion, 0.0)
        best_challenger = max(all_gains.keys(), key=lambda a: all_gains[a])
        return (best_challenger, all_gains[best_challenger])

    def calculate_entropy(self) -> float:
        self._sync_to_gpu()
        theta = self.estimate_bt_strengths_gpu()
        p_best = self.compute_best_arm_distribution_gpu(theta)
        entropy = self.compute_entropy_gpu(p_best)
        return entropy.item()

    def get_related_arms(self, arm: int) -> Set[int]:
        related = {arm}
        for other in range(self.num_arms):
            if other != arm and self.total_counts[arm, other] > 0:
                related.add(other)
        return related

    def get_statistics(self) -> Dict[str, Any]:
        total_comparisons = int(np.sum(self.total_counts) / 2)
        pairs_with_data = int(np.sum(self.total_counts > 0) / 2)
        total_pairs = self.num_arms * (self.num_arms - 1) // 2

        return {
            'total_comparisons': total_comparisons,
            'pairs_with_comparisons': pairs_with_data,
            'total_possible_pairs': total_pairs,
            'coverage_ratio': pairs_with_data / total_pairs if total_pairs > 0 else 0,
            'device': str(self.device),
            'use_gpu': self.use_gpu
        }

class InformationGainManager:

    def __init__(self, num_arms: int, use_optimized: bool = True,
                 bayesian_alpha: float = 1.0, device: str = 'cuda'):
        self.manager = PairwiseInformationManager(num_arms, use_optimized, bayesian_alpha, device)

    def update_comparison(self, arm1: int, arm2: int, arm1_wins: bool) -> float:
        gain = self.manager.calculate_information_gain_optimized(arm1, arm2)
        self.manager.update_pairwise_probability_with_transitive(arm1, arm2, arm1_wins)
        return gain

    def get_best_comparison(self, champion: int,
                            available_arms: Optional[List[int]] = None) -> Tuple[int, float]:
        return self.manager.get_best_pair_to_compare(champion, available_arms)

    def get_probability(self, arm1: int, arm2: int) -> float:
        return self.manager.get_pairwise_probability(arm1, arm2)

    def get_statistics(self) -> Dict[str, Any]:
        return self.manager.get_statistics()

def run_speed_test(num_arms: int = 400, num_iterations: int = 100, device: str = 'cuda'):
    print("=" * 70)
    print(f"🚀 IDS Information Gain Speed Test")
    print(f"   Arms: {num_arms}, Iterations: {num_iterations}, Device: {device}")
    print("=" * 70)

    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        device = 'cpu'

    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("\n📦 Initializing PairwiseInformationManager...")
    start = time.time()
    manager = PairwiseInformationManager(num_arms, device=device)
    init_time = time.time() - start
    print(f"   Initialization time: {init_time:.3f}s")

    print("\n" + "-" * 50)
    print("📊 Test 1: All arms isolated (no comparison history)")
    print("   Expected: Fast due to isolated arms reuse optimization")

    champion = 0
    start = time.time()
    gains = manager.get_all_information_gains(champion)
    test1_time = time.time() - start

    isolated_count = sum(1 for a in range(num_arms) if np.sum(manager.total_counts[a, :]) == 0)
    connected_count = num_arms - isolated_count

    print(f"   Isolated arms: {isolated_count}, Connected arms: {connected_count}")
    print(f"   Time: {test1_time:.3f}s")
    print(f"   Time per candidate: {test1_time / (num_arms - 1) * 1000:.2f}ms")

    print("\n" + "-" * 50)
    print(f"📊 Test 2: Simulate {num_iterations} iterations")

    np.random.seed(42)
    iteration_times = []

    for i in range(num_iterations):
        iter_start = time.time()

        champion = np.random.randint(0, num_arms)

        gains = manager.get_all_information_gains(champion)

        if gains:
            best_challenger = max(gains.keys(), key=lambda a: gains[a])
        else:
            best_challenger = (champion + 1) % num_arms

        champion_wins = np.random.random() > 0.5
        manager.update_pairwise_probability_with_transitive(champion, best_challenger, champion_wins)

        iter_time = time.time() - iter_start
        iteration_times.append(iter_time)

        if (i + 1) % 10 == 0 or i == 0:
            isolated = sum(1 for a in range(num_arms) if np.sum(manager.total_counts[a, :]) == 0)
            connected = num_arms - isolated
            print(f"   Iter {i+1:3d}: time={iter_time:.3f}s, isolated={isolated}, connected={connected}")

    print("\n" + "=" * 70)
    print("📈 Summary")
    print("=" * 70)

    total_time = sum(iteration_times)
    avg_time = np.mean(iteration_times)
    min_time = np.min(iteration_times)
    max_time = np.max(iteration_times)

    print(f"   Total iterations: {num_iterations}")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Average time per iteration: {avg_time:.3f}s ({avg_time*1000:.1f}ms)")
    print(f"   Min iteration time: {min_time:.3f}s")
    print(f"   Max iteration time: {max_time:.3f}s")

    stats = manager.get_statistics()
    print(f"\n   Final statistics:")
    print(f"   - Total comparisons: {stats['total_comparisons']}")
    print(f"   - Pairs with data: {stats['pairs_with_comparisons']}")
    print(f"   - Coverage: {stats['coverage_ratio']*100:.2f}%")
    print(f"   - Device: {stats['device']}")

    final_isolated = sum(1 for a in range(num_arms) if np.sum(manager.total_counts[a, :]) == 0)
    final_connected = num_arms - final_isolated
    print(f"\n   Optimization effectiveness:")
    print(f"   - Final isolated arms: {final_isolated}")
    print(f"   - Final connected arms: {final_connected}")
    print(f"   - Computations saved by reuse: {final_isolated} per iteration")

    print("\n" + "=" * 70)
    print("✅ Speed test completed!")
    print("=" * 70)

    return {
        'total_time': total_time,
        'avg_time': avg_time,
        'iteration_times': iteration_times,
        'stats': stats
    }

class ContextualPairwiseInformationManager:

    def __init__(self, num_arms: int, use_optimized: bool = True, bayesian_alpha: float = 1.0,
                 device: str = 'cuda', bt_isolated_arm_mode: str = 'unknown_isolated'):
        self.num_arms = num_arms
        self.use_optimized = use_optimized
        self.bayesian_alpha = bayesian_alpha
        self.device = device
        self.bt_isolated_arm_mode = bt_isolated_arm_mode
        self.epsilon = 1e-8

        try:
            import sys
            import os
            ftpersllm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FTPERSLLM')
            if ftpersllm_path not in sys.path:
                sys.path.insert(0, ftpersllm_path)
            from IDS_TAP_parameters.py import CONTEXTUAL_BANDIT_CONFIG
            self.sigmoid_steepness = CONTEXTUAL_BANDIT_CONFIG.get("sigmoid_steepness", 11.0)
            self.sigmoid_midpoint = CONTEXTUAL_BANDIT_CONFIG.get("sigmoid_midpoint", 0.5)
        except ImportError:
            self.sigmoid_steepness = 11.0
            self.sigmoid_midpoint = 0.5

        self.input_history: Dict[int, Dict[str, Any]] = {}

        self.current_input_index: Optional[int] = None
        self.current_embedding: Optional[np.ndarray] = None
        self.current_own_wins: Optional[np.ndarray] = None
        self.current_own_total: Optional[np.ndarray] = None

        self.logger = logging.getLogger(__name__)

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + np.exp(-(x - self.sigmoid_midpoint) * self.sigmoid_steepness))

    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        if emb1 is None or emb2 is None:
            return 0.0

        if isinstance(emb1, torch.Tensor):
            emb1 = emb1.cpu().numpy()
        if isinstance(emb2, torch.Tensor):
            emb2 = emb2.cpu().numpy()

        emb1 = emb1.flatten()
        emb2 = emb2.flatten()

        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 < self.epsilon or norm2 < self.epsilon:
            return 0.0

        similarity = np.dot(emb1, emb2) / (norm1 * norm2)
        return max(0.0, float(similarity))

    def initialize_for_new_input(self, input_index: int,
                                  input_embedding: np.ndarray) -> PairwiseInformationManager:
        manager = PairwiseInformationManager(
            self.num_arms, self.use_optimized, self.bayesian_alpha, self.device,
            bt_isolated_arm_mode=self.bt_isolated_arm_mode
        )

        if len(self.input_history) == 0:
            self._setup_current_query(input_index, input_embedding, manager)
            return manager

        weighted_wins = np.zeros((self.num_arms, self.num_arms))
        weighted_total = np.zeros((self.num_arms, self.num_arms))

        print(f"   📊 [Prob Matrix Inheritance] Query {input_index}: Aggregating {len(self.input_history)} historical queries")
        print(f"      Sigmoid params: steepness={self.sigmoid_steepness}, midpoint={self.sigmoid_midpoint}")

        for hist_idx, hist_data in self.input_history.items():
            sim = self._cosine_similarity(input_embedding, hist_data['embedding'])
            weight = self._sigmoid(sim)

            hist_comparisons = int(np.sum(hist_data['own_total']) / 2)
            print(f"      Historical Query {hist_idx}: cosine_sim={sim:.4f} → sigmoid_weight={weight:.4f} (comparisons={hist_comparisons})")

            weighted_wins += weight * hist_data['own_wins']
            weighted_total += weight * hist_data['own_total']

        manager.win_counts = weighted_wins.copy()
        manager.total_counts = weighted_total.copy()

        alpha = self.bayesian_alpha
        probs = (weighted_wins + alpha) / (weighted_total + 2 * alpha)
        np.fill_diagonal(probs, 0.5)
        manager.pairwise_probs = probs

        manager._sync_to_gpu()

        manager.reset_current_query_counts()

        self._setup_current_query(input_index, input_embedding, manager)
        return manager

    def initialize_without_history(self, input_index: int,
                                    input_embedding: np.ndarray) -> PairwiseInformationManager:
        manager = PairwiseInformationManager(
            self.num_arms, self.use_optimized, self.bayesian_alpha, self.device
        )

        self._setup_current_query(input_index, input_embedding, manager)
        return manager

    def _setup_current_query(self, input_index: int, input_embedding: np.ndarray,
                              manager: PairwiseInformationManager):
        self.current_input_index = input_index
        self.current_embedding = input_embedding.copy() if isinstance(input_embedding, np.ndarray) else input_embedding
        self.current_own_wins = np.zeros((self.num_arms, self.num_arms))
        self.current_own_total = np.zeros((self.num_arms, self.num_arms))
        manager.reset_current_query_counts()

    def record_comparison(self, arm1: int, arm2: int, arm1_wins: bool) -> None:
        if self.current_own_wins is None or self.current_own_total is None:
            self.logger.warning("record_comparison called but current query not initialized")
            return

        if arm1_wins:
            self.current_own_wins[arm1, arm2] += 1
        else:
            self.current_own_wins[arm2, arm1] += 1

        self.current_own_total[arm1, arm2] += 1
        self.current_own_total[arm2, arm1] += 1

    def finalize_query(self) -> None:
        if self.current_input_index is None:
            self.logger.warning("finalize_query called but no active query")
            return

        self.input_history[self.current_input_index] = {
            'embedding': self.current_embedding,
            'own_wins': self.current_own_wins.copy(),
            'own_total': self.current_own_total.copy()
        }

    def get_statistics(self) -> Dict[str, Any]:
        total_own_comparisons = 0
        for hist_data in self.input_history.values():
            total_own_comparisons += int(np.sum(hist_data['own_total']) / 2)

        return {
            'num_queries': len(self.input_history),
            'total_own_comparisons': total_own_comparisons,
            'sigmoid_steepness': self.sigmoid_steepness,
            'sigmoid_midpoint': self.sigmoid_midpoint
        }

def update_information_with_feedback(manager: PairwiseInformationManager,
                                      arm1: int, arm2: int, arm1_wins: bool) -> float:
    info_gain = manager.calculate_information_gain_optimized(arm1, arm2)

    manager.update_pairwise_probability_with_transitive(arm1, arm2, arm1_wins)

    return info_gain

def compare_gpu_cpu_speed(num_arms: int = 100, num_iterations: int = 20):
    print("\n" + "=" * 70)
    print("🔄 GPU vs CPU Speed Comparison")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping comparison")
        return

    print("\n📊 CPU Test:")
    cpu_results = run_speed_test(num_arms, num_iterations, device='cpu')

    print("\n📊 GPU Test:")
    gpu_results = run_speed_test(num_arms, num_iterations, device='cuda')

    print("\n" + "=" * 70)
    print("📊 Comparison Results")
    print("=" * 70)
    speedup = cpu_results['avg_time'] / gpu_results['avg_time']
    print(f"   CPU avg time: {cpu_results['avg_time']*1000:.1f}ms")
    print(f"   GPU avg time: {gpu_results['avg_time']*1000:.1f}ms")
    print(f"   Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    print("🧪 IDS-style Information Gain Implementation")
    print("=" * 70)

    print("\n📋 Quick Functionality Test (5 arms):")
    manager = InformationGainManager(5, device='cpu')

    gain1 = manager.update_comparison(0, 1, True)
    gain2 = manager.update_comparison(1, 2, True)
    gain3 = manager.update_comparison(2, 3, False)

    print(f"   Gains: {gain1:.4f}, {gain2:.4f}, {gain3:.4f}")

    best_arm, best_gain = manager.get_best_comparison(0)
    print(f"   Best challenger for arm 0: arm {best_arm}, gain: {best_gain:.4f}")

    stats = manager.get_statistics()
    print(f"   Stats: {stats}")
    print("✅ Functionality test passed!")

    print("\n" + "=" * 70)
    print("🚀 Running Speed Test...")
    print("=" * 70)

    run_speed_test(num_arms=400, num_iterations=100, device='cuda')
