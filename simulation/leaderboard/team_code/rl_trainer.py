#rl_trainer.py
import csv
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from team_code.rl_policy import GammaPolicy
from team_code.rl_buffer import RolloutBuffer


class CRPOTrainer(object):
    def __init__(
        self,
        obs_dim,
        action_dim=3,
        device=None,
        n_steps=2000,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        update_epochs=10,
        num_minibatches=32,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        cost_limit=0.1,
        lagrange_mult=1.0,
        lagrange_mult_max=100.0,
        use_pid_lagrangian=True,
        pid_gains=(1.0, 0.1, 0.1),
        pid_ema_alpha=0.95,
        pid_d_delay=1,
        hidden_sizes=(64, 64),
        log_dir=None,
        save_every=10,
        log_scale=1000.0,
        resume_path=None,
        eval_only=False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.eval_only = bool(eval_only)

        if self.eval_only and (not resume_path or not os.path.exists(resume_path)):
            raise ValueError(
                "CRPOTrainer eval_only=True requires a valid resume_path checkpoint, got: {}".format(
                    resume_path
                )
            )

        self.policy = GammaPolicy(obs_dim, action_dim, hidden_sizes=hidden_sizes).to(
            self.device
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate, eps=1e-5)

        self.buffer = RolloutBuffer(n_steps)
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.update_epochs = update_epochs
        self.num_minibatches = num_minibatches
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        # Lagrangian
        self.use_pid_lagrangian = use_pid_lagrangian
        self.kp, self.ki, self.kd = pid_gains
        self.pid_ema_alpha = pid_ema_alpha
        self.pid_d_delay = pid_d_delay
        self.lagrange_mult = lagrange_mult
        self.lagrange_mult_max = lagrange_mult_max
        self.cost_limit = cost_limit

        self.p_int = 0.0
        self.ema_p = 0.0
        self.ema_d = 0.0
        self.dt_queue = [0.0 for _ in range(pid_d_delay)]

        # Previous step cache
        self.prev_obs = None
        self.prev_action = None
        self.prev_logprob = None
        self.prev_value = None
        self.last_obs = None

        # Logging / checkpoints
        self.update_count = 0
        self.total_steps = 0
        self.log_dir = log_dir
        self.save_every = int(save_every)
        self.log_scale = float(log_scale)
        self._log_path = None
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            self._log_path = os.path.join(self.log_dir, "train_log.csv")
            if not os.path.exists(self._log_path):
                with open(self._log_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "timestamp",
                            "update",
                            "total_steps",
                            "batch_steps",
                            "avg_reward",
                            "avg_cost",
                            "avg_return",
                            "avg_cost_return",
                            "avg_reward_scaled",
                            "avg_cost_scaled",
                            "avg_return_scaled",
                            "avg_cost_return_scaled",
                            "lagrange_mult",
                            "mean_action_0",
                            "mean_action_1",
                            "mean_action_2",
                        ]
                    )
        if resume_path:
            self.load_checkpoint(resume_path)

    def _to_tensor(self, x):
        return torch.tensor(x, dtype=torch.float32, device=self.device)

    def select_action(self, obs, deterministic=False):
        obs_t = self._to_tensor(obs)
        with torch.no_grad():
            action, logprob, _, value = self.policy.get_action_and_value(
                obs_t, deterministic=deterministic
            )
        return (
            action.squeeze(0).cpu().numpy(),
            logprob.squeeze(0).cpu().item(),
            value.squeeze(0).cpu().item(),
        )

    def step(self, obs, reward, cost, done=False, deterministic=False):
        """
        Use current obs to (1) record transition for previous action, and
        (2) produce next action.
        """
        if self.eval_only:
            action, _, _ = self.select_action(obs, deterministic=deterministic)
            return action

        self.last_obs = obs
        if self.prev_obs is not None:
            self.buffer.add(
                self.prev_obs,
                self.prev_action,
                self.prev_logprob,
                self.prev_value,
                reward,
                cost,
                done,
            )
            self.total_steps += 1

        action, logprob, value = self.select_action(obs, deterministic=deterministic)
        self.prev_obs = obs
        self.prev_action = action
        self.prev_logprob = logprob
        self.prev_value = value

        # Update when buffer is full
        if self.buffer.is_full():
            # Ensure gradients are enabled even if caller runs under torch.no_grad().
            with torch.enable_grad():
                self._update(next_obs=obs, next_done=done)
            self.buffer.clear()

        return action

    def finish_episode(
        self, completed=False, collided=False, cost_terminal_collision=1.0
    ):
        """Flush last transition when route ends."""
        if self.eval_only:
            self.prev_obs = None
            self.prev_action = None
            self.prev_logprob = None
            self.prev_value = None
            return

        if self.prev_obs is None:
            return

        reward_terminal = 1.0 if completed else 0.0
        cost_terminal = float(cost_terminal_collision) if collided else 0.0
        self.buffer.add(
            self.prev_obs,
            self.prev_action,
            self.prev_logprob,
            self.prev_value,
            reward_terminal,
            cost_terminal,
            True,
        )
        self.total_steps += 1

        self.prev_obs = None
        self.prev_action = None
        self.prev_logprob = None
        self.prev_value = None

        if self.buffer.is_full():
            with torch.enable_grad():
                self._update(next_obs=None, next_done=True)
            self.buffer.clear()

    def flush_pending(self):
        """
        Run one update with any remaining samples in the buffer.
        Returns True if an update was performed.
        """
        if self.eval_only:
            return False

        if self.buffer.size() == 0:
            return False
        with torch.enable_grad():
            self._update(next_obs=None, next_done=True)
        self.buffer.clear()
        return True

    def save_checkpoint(self, path):
        state = {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self.update_count,
            "total_steps": self.total_steps,
            "lagrange_mult": self.lagrange_mult,
            "p_int": self.p_int,
            "ema_p": self.ema_p,
            "ema_d": self.ema_d,
            "dt_queue": self.dt_queue,
        }
        torch.save(state, path)

    def load_checkpoint(self, path):
        state = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(state.get("policy", {}))
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        self.update_count = int(state.get("update_count", 0))
        self.total_steps = int(state.get("total_steps", 0))
        self.lagrange_mult = float(state.get("lagrange_mult", self.lagrange_mult))
        self.p_int = float(state.get("p_int", self.p_int))
        self.ema_p = float(state.get("ema_p", self.ema_p))
        self.ema_d = float(state.get("ema_d", self.ema_d))
        self.dt_queue = list(state.get("dt_queue", self.dt_queue))

    def _update(self, next_obs=None, next_done=False):
        # Prepare tensors
        obs = self._to_tensor(np.array(self.buffer.obs))
        actions = self._to_tensor(np.array(self.buffer.actions))
        logprobs = self._to_tensor(np.array(self.buffer.logprobs))
        values = self._to_tensor(np.array(self.buffer.values))
        rewards = self._to_tensor(np.array(self.buffer.rewards))
        costs = self._to_tensor(np.array(self.buffer.costs))
        dones = self._to_tensor(np.array(self.buffer.dones).astype(np.float32))

        # Bootstrap value
        if next_done or next_obs is None:
            next_value = torch.zeros(1, device=self.device)
        else:
            with torch.no_grad():
                next_value = self.policy.get_value(self._to_tensor(next_obs)).view(1)

        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        cost_returns = torch.zeros_like(costs)

        lastgaelam = 0.0
        next_cost = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 0.0 if next_done else 1.0
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]

            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam
            returns[t] = advantages[t] + values[t]

            # cost returns
            cost_returns[t] = costs[t] + self.gamma * nextnonterminal * next_cost
            next_cost = cost_returns[t]

        # Normalize advantages
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - advantages.mean()) / adv_std

        # Update Lagrange multiplier
        if self.use_pid_lagrangian:
            avg_cost = float(costs.mean().item()) if costs.numel() > 0 else 0.0

            err = avg_cost - self.cost_limit
            self.p_int = max(0.0, self.p_int + err)
            self.ema_p = self.pid_ema_alpha * self.ema_p + (1 - self.pid_ema_alpha) * err
            self.ema_d = self.pid_ema_alpha * self.ema_d + (1 - self.pid_ema_alpha) * avg_cost
            p_der = max(0.0, self.ema_d - self.dt_queue[0])
            self.dt_queue.pop(0)
            self.dt_queue.append(self.ema_d)

            self.lagrange_mult = self.kp * self.ema_p + self.ki * self.p_int + self.kd * p_der
            self.lagrange_mult = max(0.0, min(self.lagrange_mult, self.lagrange_mult_max))
        else:
            self.lagrange_mult = max(0.0, min(self.lagrange_mult, self.lagrange_mult_max))

        # PPO updates
        batch_size = obs.shape[0]
        minibatch_size = max(1, batch_size // self.num_minibatches)
        b_inds = np.arange(batch_size)

        for _ in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                mb_obs = obs[mb_inds]
                mb_actions = actions[mb_inds]
                mb_logprobs = logprobs[mb_inds]
                mb_adv = advantages[mb_inds]
                mb_cost = cost_returns[mb_inds]
                mb_returns = returns[mb_inds]

                newlogprob, entropy, newvalue = self.policy.evaluate_actions(
                    mb_obs, mb_actions
                )
                ratio = (newlogprob - mb_logprobs).exp()

                # Policy loss
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Cost loss
                cost_loss1 = mb_cost * ratio
                cost_loss2 = mb_cost * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                cost_loss = torch.max(cost_loss1, cost_loss2).mean()

                # Value loss
                v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

                entropy_loss = entropy.mean()

                loss = pg_loss + self.lagrange_mult * cost_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        # Logging + checkpointing
        self.update_count += 1
        avg_reward = float(rewards.mean().item()) if rewards.numel() > 0 else 0.0
        avg_cost = float(costs.mean().item()) if costs.numel() > 0 else 0.0
        avg_return = float(returns.mean().item()) if returns.numel() > 0 else 0.0
        avg_cost_return = (
            float(cost_returns.mean().item()) if cost_returns.numel() > 0 else 0.0
        )
        avg_reward_scaled = avg_reward * self.log_scale
        avg_cost_scaled = avg_cost * self.log_scale
        avg_return_scaled = avg_return * self.log_scale
        avg_cost_return_scaled = avg_cost_return * self.log_scale
        mean_action = np.mean(np.array(self.buffer.actions), axis=0) if self.buffer.actions else np.zeros(3)

        if self._log_path:
            with open(self._log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        datetime.utcnow().isoformat(),
                        self.update_count,
                        self.total_steps,
                        batch_size,
                        avg_reward,
                        avg_cost,
                        avg_return,
                        avg_cost_return,
                        avg_reward_scaled,
                        avg_cost_scaled,
                        avg_return_scaled,
                        avg_cost_return_scaled,
                        float(self.lagrange_mult),
                        float(mean_action[0]) if len(mean_action) > 0 else 0.0,
                        float(mean_action[1]) if len(mean_action) > 1 else 0.0,
                        float(mean_action[2]) if len(mean_action) > 2 else 0.0,
                    ]
                )

        if self.log_dir and self.save_every > 0 and self.update_count % self.save_every == 0:
            ckpt_dir = os.path.join(self.log_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_update_{self.update_count}.pt")
            self.save_checkpoint(ckpt_path)
            latest_path = os.path.join(ckpt_dir, "ckpt_latest.pt")
            self.save_checkpoint(latest_path)
