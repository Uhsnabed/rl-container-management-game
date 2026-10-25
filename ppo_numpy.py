"""PPO actor-critic implemented with PyTorch autograd/backpropagation.

The historical filename is retained so existing imports keep working. The
policy and value function are now small neural networks rather than linear maps.
This is an educational PPO implementation, not a production RL library.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)  # avoid large thread-pool overhead on tiny minibatches
from torch import nn
from torch.distributions import Categorical


def _mlp(input_size: int, output_size: int, hidden: int = 64) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_size, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, output_size),
    )


class ActorCritic(nn.Module):
    def __init__(self, obs_size: int, actions: int, hidden: int = 64):
        super().__init__()
        self.actor = _mlp(obs_size, actions, hidden)
        self.critic = _mlp(obs_size, 1, hidden)

    def forward(self, obs: torch.Tensor):
        return self.actor(obs), self.critic(obs).squeeze(-1)


class LinearPPO:
    """Compatibility name; now a neural actor-critic trained with backprop."""
    def __init__(self, obs_size=6, actions=4, seed=0, policy_lr=3e-4,
                 value_lr=1e-3, gamma=0.99, gae_lambda=0.95, clip=0.2,
                 value_clip=0.2, entropy_coef=0.025, value_coef=0.5,
                 hidden_size=64, max_grad_norm=0.5, device="cpu"):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = torch.device(device)
        self.model = ActorCritic(obs_size, actions, hidden_size).to(self.device)
        # Separate optimizers make actor/critic learning rates independently tunable.
        self.actor_optimizer = torch.optim.Adam(self.model.actor.parameters(), lr=policy_lr)
        self.critic_optimizer = torch.optim.Adam(self.model.critic.parameters(), lr=value_lr)
        self.rng = np.random.default_rng(seed)
        self.gamma, self.gae_lambda, self.clip = gamma, gae_lambda, clip
        self.value_clip, self.entropy_coef, self.value_coef = value_clip, entropy_coef, value_coef
        self.max_grad_norm = max_grad_norm

    def _tensor(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def _masked_logits(self, obs, action_masks=None):
        logits = self.model.actor(obs)
        if action_masks is not None:
            masks = self._tensor(action_masks).bool()
            logits = logits.masked_fill(~masks, -1e9)
        return logits

    def probs(self, obs, valid_actions=None):
        x = self._tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        mask = None
        if valid_actions is not None:
            mask = np.arange(self.model.actor[-1].out_features)[None, :] < int(valid_actions)
        with torch.no_grad():
            p = torch.softmax(self._masked_logits(x, mask), dim=-1)[0].cpu().numpy()
        return p

    def act(self, obs, deterministic=False, valid_actions=None):
        x = self._tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        mask = None
        if valid_actions is not None:
            mask = np.arange(self.model.actor[-1].out_features)[None, :] < int(valid_actions)
        with torch.no_grad():
            logits, value = self.model(x)
            if mask is not None:
                logits = logits.masked_fill(~self._tensor(mask).bool(), -1e9)
            dist = Categorical(logits=logits)
            action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
            logp = dist.log_prob(action)
        return int(action.item()), float(logp.item()), float(value.item())

    def _returns_and_advantages(self, rewards, values, dones, last_value=0.0):
        rewards = np.asarray(rewards, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)
        advantages = np.zeros_like(rewards)
        gae, next_value = 0.0, float(last_value)
        for t in range(len(rewards) - 1, -1, -1):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * nonterminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
            advantages[t] = gae
            next_value = values[t]
        return advantages + values, advantages

    def update(self, transitions, epochs=8, minibatch_size=64):
        if not transitions:
            return {}
        obs = self._tensor(np.asarray([t['obs'] for t in transitions], dtype=np.float32))
        actions = torch.as_tensor([t['action'] for t in transitions], dtype=torch.long, device=self.device)
        old_logp = self._tensor([t['logp'] for t in transitions])
        returns = self._tensor([t['return'] for t in transitions])
        advantages = self._tensor([t['advantage'] for t in transitions])
        old_values = self._tensor([t.get('value', 0.0) for t in transitions])
        # Store the valid action set from collection so the old/new policies match.
        masks = np.asarray([t.get('action_mask', np.ones(self.model.actor[-1].out_features, dtype=bool))
                            for t in transitions], dtype=bool)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        n = len(transitions)
        losses = []
        for _ in range(epochs):
            permutation = torch.randperm(n, device=self.device)
            for start in range(0, n, minibatch_size):
                idx = permutation[start:start + minibatch_size]
                batch_obs, batch_actions = obs[idx], actions[idx]
                batch_masks = masks[idx.cpu().numpy()]
                logits = self._masked_logits(batch_obs, batch_masks)
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(batch_actions)
                ratio = torch.exp(torch.clamp(new_logp - old_logp[idx], -10, 10))
                adv = advantages[idx]
                surrogate1 = ratio * adv
                surrogate2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv
                policy_loss = -torch.minimum(surrogate1, surrogate2).mean()
                entropy = dist.entropy().mean()

                _, values = self.model(batch_obs)
                # PPO-style clipped value objective; clip is relative to rollout values.
                clipped_values = old_values[idx] + torch.clamp(
                    values - old_values[idx], -self.value_clip, self.value_clip)
                value_loss = 0.5 * torch.maximum(
                    (values - returns[idx]).pow(2),
                    (clipped_values - returns[idx]).pow(2)).mean()
                actor_loss = policy_loss - self.entropy_coef * entropy
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.model.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad(set_to_none=True)
                (self.value_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                losses.append((float(policy_loss.detach()), float(value_loss.detach()), float(entropy.detach())))
        arr = np.asarray(losses)
        return {'policy_loss': float(arr[:, 0].mean()), 'value_loss': float(arr[:, 1].mean()),
                'entropy': float(arr[:, 2].mean())}

    def save(self, path):
        path = Path(path)
        torch.save({'model': self.model.state_dict(),
                    'actor_optimizer': self.actor_optimizer.state_dict(),
                    'critic_optimizer': self.critic_optimizer.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model'])
        if 'actor_optimizer' in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
