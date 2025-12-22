import torch
import torch.nn.functional as F

from spectralrl.algo.visual.base import BaseVisualAlgorithm
from spectralrl.algo.visual.mulvrep_drqv2.network import (
    Actor,
    Decoder,
    Encoder,
    FeatDecoder,
    FeatEncoder,
    GaussianFeature,
    PredictEncoder,
    RandomShiftsAug,
    RFFCritic,
    setup_schedule,
)
from spectralrl.utils.utils import convert_to_tensor, make_target, sync_target


class MuLVRep_DrQv2(BaseVisualAlgorithm):
    def __init__(
        self,
        obs_space,
        action_space,
        cfg,
        device
    ) -> None:
        super().__init__()
        self.cfg = cfg

        self.obs_dim = obs_space.shape
        self.action_dim = action_space.shape[0]
        self.action_range = [
            action_space.minimum,
            action_space.maximum
        ]
        self.tau = cfg.tau
        self.update_every = cfg.update_every
        self.critic_loss_type = cfg.critic_loss_type
        self.critic_loss_fn = {
            "mse": F.mse_loss,
            "huber": F.huber_loss
        }.get(self.critic_loss_type)
        self.stddev_schedule = setup_schedule(cfg.stddev_schedule)
        self.stddev_clip = cfg.stddev_clip
        self.device = device

        # losses
        self.repr_coef = cfg.repr_coef
        self.kl_coef = cfg.kl_coef

        # networks
        self.encoder = Encoder(self.obs_dim).to(self.device)
        self.encoder_target = make_target(self.encoder)
        self.decoder = Decoder().to(self.device)
        self.predict_encoder = PredictEncoder(self.obs_dim).to(self.device)

        # the vae structure for spectral representation
        self.feat_encoder = FeatEncoder(
            repr_dim=self.encoder.repr_dim,
            action_dim=self.action_dim,
            feature_dim=cfg.feature_dim,
        ).to(self.device)
        self.feat_decoder = FeatDecoder(
            repr_dim=self.encoder.repr_dim,
            feature_dim=cfg.feature_dim,
            hidden_dim=cfg.hidden_dim,
        ).to(self.device)
        self.feat_f = GaussianFeature(
            repr_dim=self.encoder.repr_dim,
            action_dim=self.action_dim,
            feature_dim=cfg.feature_dim
        ).to(self.device)
        self.feat_f_target = make_target(self.feat_f)

        self.actor = Actor(
            repr_dim=self.encoder.repr_dim,
            action_dim=self.action_dim,
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.hidden_dim
        ).to(self.device)
        self.rff_critic = RFFCritic(
            feature_dim=cfg.feature_dim,
            c_noise=0.1,
            q_activ="relu",
            hidden_dim=cfg.hidden_dim
        ).to(self.device)
        self.rff_critic_target = make_target(self.rff_critic)
        self.aug = RandomShiftsAug(pad=4)

        self.optim = {
            "encoder": torch.optim.Adam(self.encoder.parameters(), lr=cfg.lr),
            "decoder": torch.optim.Adam(self.decoder.parameters(), lr=cfg.lr),
            "predict_encoder": torch.optim.Adam(self.predict_encoder.parameters(), lr=cfg.lr),
            "actor": torch.optim.Adam(self.actor.parameters(), lr=cfg.lr),
            "rff_critic": torch.optim.Adam(self.rff_critic.parameters(), lr=cfg.lr),
            "feat_encoder": torch.optim.Adam(self.feat_encoder.parameters(), lr=cfg.lr),
            "feat_decoder": torch.optim.Adam(self.feat_decoder.parameters(), lr=cfg.lr),
            "feat_f": torch.optim.Adam(self.feat_f.parameters(), lr=cfg.lr)
        }
        self._step = 1
        self.train()

    def train(self, training=True):
        self.training = training
        for module in [
            self.encoder,
            self.decoder,
            self.predict_encoder,
            self.actor,
            self.rff_critic,
            self.feat_encoder,
            self.feat_decoder,
            self.feat_f
        ]:
            module.train(training)

    @torch.no_grad()
    def evalute(self, replay_iter):
        return {}

    @torch.no_grad()
    def select_action(self, obs, step, deterministic=False):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        obs = obs.view(1, -1)
        stddev = self.stddev_schedule(step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=None) if not deterministic else dist.mean
        return action.squeeze().cpu().numpy()

    def pretrain_step(self, replay_iter, step):
        batch = next(replay_iter)
        img_stack, action, reward, discount, next_img_stack, next_img_step = [
            convert_to_tensor(t, self.device) for t in batch
        ]
        img_stack = self.aug(img_stack.float()).detach()
        next_img_stack = self.aug(next_img_stack.float()).detach()
        next_img_step = next_img_step[:, -3:].float().detach()
        metrics = {}

        repr_metrics, recon_loss, kl_loss, *_ = self.repr_step(img_stack, next_img_stack, next_img_step, action, reward)
        metrics.update(repr_metrics)
        for optim in {"encoder", "decoder", "predict_encoder", "feat_encoder", "feat_decoder", "feat_f"}:
            self.optim[optim].zero_grad()
        (self.repr_coef * (recon_loss + self.kl_coef * kl_loss)).backward()
        for optim in {"encoder", "decoder", "predict_encoder", "feat_encoder", "feat_decoder", "feat_f"}:
            self.optim[optim].step()
        sync_target(self.encoder, self.encoder_target, 1.0)
        sync_target(self.feat_f, self.feat_f_target, 1.0)

        return metrics

    def train_step(self, replay_iter, step):
        self._step += 1
        if self._step % self.update_every != 0:
            return {}
        batch = next(replay_iter)
        img_stack, action, reward, discount, next_img_stack, next_img_step = [
            convert_to_tensor(t, self.device) for t in batch
        ]
        img_stack = self.aug(img_stack.float()).detach()
        next_img_stack = self.aug(next_img_stack.float()).detach()
        next_img_step = next_img_step[:, -3:].float().detach()
        metrics = {}

        # update repr and critic
        repr_metrics, recon_loss, kl_loss, state = self.repr_step(img_stack, next_img_stack, next_img_step, action, reward)
        metrics.update(repr_metrics)
        critic_metrics, critic_loss = self.critic_step(img_stack, next_img_stack, action, reward, discount, step, state)
        metrics.update(critic_metrics)
        for optim in {"encoder", "decoder", "predict_encoder", "feat_encoder", "feat_decoder", "feat_f", "rff_critic"}:
            self.optim[optim].zero_grad()
        (self.repr_coef * (recon_loss + self.kl_coef * kl_loss) + critic_loss).backward()
        for optim in {"encoder", "decoder", "predict_encoder", "feat_encoder", "feat_decoder", "feat_f", "rff_critic"}:
            self.optim[optim].step()

        # update actor
        actor_metrics, actor_loss = self.actor_step(state.detach(), step)
        metrics.update(actor_metrics)
        self.optim["actor"].zero_grad()
        actor_loss.backward()
        self.optim["actor"].step()

        # target networks
        sync_target(self.encoder, self.encoder_target, self.tau)
        sync_target(self.feat_f, self.feat_f_target, self.tau)
        sync_target(self.rff_critic, self.rff_critic_target, self.tau)

        return metrics

    def repr_step(self, img_stack, next_img_stack, next_img_step, action, reward):
        state = self.encoder(img_stack)
        next_state_step = self.predict_encoder(next_img_step)

        # reconstruction loss
        z = self.feat_encoder.sample(state, action, next_state_step)
        x, r = self.feat_decoder(z)
        pred_next_img_step = self.decoder(x)
        target_next_img_step = next_img_step / 255.0 - 0.5
        s_loss = F.l1_loss(pred_next_img_step, target_next_img_step)
        r_loss = F.mse_loss(r, reward)
        recon_loss = r_loss + 10. * s_loss

        # kl loss
        mean1, log_std1 = self.feat_encoder(state, action, next_state_step)
        mean2, log_std2 = self.feat_f(state, action)
        var1 = (2 * log_std1).exp()
        var2 = (2 * log_std2).exp()
        kl_loss = log_std2 - log_std1 + 0.5 * (var1 + (mean1 - mean2) ** 2) / var2 - 0.5
        kl_loss = kl_loss.mean()

        return {
            "loss/recon_loss": recon_loss.item(),
            "loss/kl_loss": kl_loss.item()
        }, recon_loss, kl_loss, state

    def critic_step(self, img_stack, next_img_stack, action, reward, discount, step, state):
        with torch.no_grad():
            next_state = self.encoder_target(next_img_stack)
            stddev = self.stddev_schedule(step)
            dist = self.actor(next_state, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            next_mean, next_log_std = self.feat_f_target(next_state, next_action)
            target_Q1, target_Q2 = self.rff_critic_target(next_mean, next_log_std)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)
        mean, log_std = self.feat_f(state, action)
        Q1, Q2 = self.rff_critic(mean, log_std)
        critic_loss = self.critic_loss_fn(Q1, target_Q) + self.critic_loss_fn(Q2, target_Q)

        return {
            "info/q_pred": Q1.mean().item(),
            "info/reward": reward.mean().item(),
            "loss/critic_loss": critic_loss.item()
        }, critic_loss

    def actor_step(self, state, step):
        stddev = self.stddev_schedule(step)
        dist = self.actor(state, stddev)
        action = dist.sample(clip=self.stddev_clip)
        mean, log_std = self.feat_f(state, action)
        Q1, Q2 = self.rff_critic(mean, log_std)
        actor_loss = - torch.min(Q1, Q2).mean()

        return {
            "loss/actor_loss": actor_loss.item(),
        }, actor_loss
