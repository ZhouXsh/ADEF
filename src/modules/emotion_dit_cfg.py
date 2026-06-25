import torch

from .emotion_dit import DiffusionSchedule, DenoisingNetwork, DitTalkingHead as _DitTalkingHead


class DitTalkingHead(_DitTalkingHead):
    def _cfg_lip_mask(self, x):  # 优化CFG，2026年6月25日
        lip_mask = torch.zeros_like(x, dtype=torch.bool)  # 优化CFG，2026年6月25日
        if x.shape[-1] >= 63:  # 优化CFG，2026年6月25日
            for lip_idx in [6, 12, 14, 17, 19, 20]:  # 优化CFG，2026年6月25日
                lip_mask[..., lip_idx * 3:lip_idx * 3 + 3] = True  # 优化CFG，2026年6月25日
        return lip_mask  # 优化CFG，2026年6月25日

    def _cfg_modulate_audio(self, audio_feat, emo_feat):  # 优化CFG，2026年6月25日
        emo_shift, emo_scale = self.adaLN_modulation(emo_feat).chunk(2, dim=2)  # 优化CFG，2026年6月25日
        return self.audio_norm(audio_feat) * (1 + emo_scale) + emo_shift  # 优化CFG，2026年6月25日

    def _cfg_make_scale(self, cfg_cond, cfg_scale):  # 优化CFG，2026年6月25日
        raw_cfg_scale = cfg_scale  # 优化CFG，2026年6月25日
        if not isinstance(cfg_scale, list):  # 优化CFG，2026年6月25日
            cfg_scale = [cfg_scale] * len(cfg_cond)  # 优化CFG，2026年6月25日
        elif len(cfg_scale) < len(cfg_cond):  # 优化CFG，2026年6月25日
            cfg_scale = cfg_scale + [cfg_scale[-1]] * (len(cfg_cond) - len(cfg_scale))  # 优化CFG，2026年6月25日
        if len(cfg_cond) > 0:  # 优化CFG，2026年6月25日
            cfg_cond, cfg_scale = zip(*sorted(zip(cfg_cond, cfg_scale), key=lambda x: ['audio', 'emotion'].index(x[0])))  # 优化CFG，2026年6月25日
            cfg_cond, cfg_scale = list(cfg_cond), list(cfg_scale)  # 优化CFG，2026年6月25日
        else:  # 优化CFG，2026年6月25日
            cfg_cond, cfg_scale = [], []  # 优化CFG，2026年6月25日
        emo_lip_scale = raw_cfg_scale[len(cfg_scale)] if isinstance(raw_cfg_scale, list) and len(raw_cfg_scale) > len(cfg_scale) else 0.2  # 优化CFG，2026年6月25日
        return cfg_cond, cfg_scale, emo_lip_scale  # 优化CFG，2026年6月25日

    def _cfg_target_theta(self, results, cfg_cond, cfg_scale, emo_lip_scale):  # 优化CFG，2026年6月25日
        result_uncond = results[0][:, -self.n_motions:]  # 优化CFG，2026年6月25日
        if 'audio' in cfg_cond and 'emotion' in cfg_cond and len(results) == 3:  # 优化CFG，2026年6月25日
            result_audio = results[cfg_cond.index('audio') + 1][:, -self.n_motions:]  # 优化CFG，2026年6月25日
            result_full = results[cfg_cond.index('emotion') + 1][:, -self.n_motions:]  # 优化CFG，2026年6月25日
            audio_scale = cfg_scale[cfg_cond.index('audio')]  # 优化CFG，2026年6月25日
            emotion_scale = cfg_scale[cfg_cond.index('emotion')]  # 优化CFG，2026年6月25日
            delta_audio = result_audio - result_uncond  # 优化CFG，2026年6月25日
            delta_emo = result_full - result_audio  # 优化CFG，2026年6月25日
            lip_mask = self._cfg_lip_mask(result_uncond)  # 优化CFG，2026年6月25日
            target_theta = result_uncond + audio_scale * delta_audio  # 优化CFG，2026年6月25日
            target_theta = target_theta + emotion_scale * delta_emo.masked_fill(lip_mask, 0)  # 优化CFG，2026年6月25日
            target_theta = target_theta + emo_lip_scale * delta_emo.masked_fill(~lip_mask, 0)  # 优化CFG，2026年6月25日
            return target_theta  # 优化CFG，2026年6月25日
        target_theta = result_uncond  # 优化CFG，2026年6月25日
        for i in range(0, len(results) - 1):  # 优化CFG，2026年6月25日
            target_theta += cfg_scale[i] * (results[i + 1][:, -self.n_motions:] - results[0][:, -self.n_motions:])  # 优化CFG，2026年6月25日
        return target_theta  # 优化CFG，2026年6月25日

    def _cfg_build_inputs(self, audio_feat, prev_audio_feat, cfg_cond, emo_index):  # 优化CFG，2026年6月25日
        batch_size = audio_feat.shape[0]  # 优化CFG，2026年6月25日
        audio_feat_in, prev_audio_feat_in = [], []  # 优化CFG，2026年6月25日
        if 'emotion' in cfg_cond:  # 优化CFG，2026年6月25日
            emo_null = self.null_emotion_feat.expand(batch_size, -1, -1)  # 优化CFG，2026年6月25日
            emo_real = self.emo_embed(emo_index).unsqueeze(1)  # 优化CFG，2026年6月25日
        else:  # 优化CFG，2026年6月25日
            emo_null, emo_real = None, None  # 优化CFG，2026年6月25日
        if 'audio' in cfg_cond:  # 优化CFG，2026年6月25日
            audio_null = self.null_audio_feat.expand(batch_size, self.n_motions, -1)  # 优化CFG，2026年6月25日
            prev_audio_null = self.null_audio_feat.expand(batch_size, self.n_prev_motions, -1)  # 优化CFG，2026年6月25日
        else:  # 优化CFG，2026年6月25日
            audio_null, prev_audio_null = audio_feat, prev_audio_feat  # 优化CFG，2026年6月25日
        if 'emotion' in cfg_cond:  # 优化CFG，2026年6月25日
            audio_feat_in.append(self._cfg_modulate_audio(audio_null, emo_null))  # 优化CFG，2026年6月25日
            prev_audio_feat_in.append(self._cfg_modulate_audio(prev_audio_null, emo_null))  # 优化CFG，2026年6月25日
        else:  # 优化CFG，2026年6月25日
            audio_feat_in.append(audio_null)  # 优化CFG，2026年6月25日
            prev_audio_feat_in.append(prev_audio_null)  # 优化CFG，2026年6月25日
        for cond in cfg_cond:  # 优化CFG，2026年6月25日
            if cond == 'audio':  # 优化CFG，2026年6月25日
                audio_item = self._cfg_modulate_audio(audio_feat, emo_null) if 'emotion' in cfg_cond else audio_feat  # 优化CFG，2026年6月25日
                prev_audio_item = self._cfg_modulate_audio(prev_audio_feat, emo_null) if 'emotion' in cfg_cond else prev_audio_feat  # 优化CFG，2026年6月25日
            else:  # 优化CFG，2026年6月25日
                audio_item = self._cfg_modulate_audio(audio_feat, emo_real)  # 优化CFG，2026年6月25日
                prev_audio_item = self._cfg_modulate_audio(prev_audio_feat, emo_real)  # 优化CFG，2026年6月25日
            audio_feat_in.append(audio_item)  # 优化CFG，2026年6月25日
            prev_audio_feat_in.append(prev_audio_item)  # 优化CFG，2026年6月25日
        return torch.cat(audio_feat_in, dim=0), torch.cat(prev_audio_feat_in, dim=0), len(audio_feat_in)  # 优化CFG，2026年6月25日

    def forward(self, motion_feat, audio_or_feat, prev_motion_feat=None, prev_audio_feat=None, time_step=None, indicator=None, emo_index=None, return_cfg_state=False):  # 优化CFG，2026年6月25日
        batch_size = motion_feat.shape[0]  # 优化CFG，2026年6月25日
        if audio_or_feat.ndim == 2:  # 优化CFG，2026年6月25日
            assert audio_or_feat.shape[1] == round(16000 * self.n_motions / self.fps), f'Incorrect audio length {audio_or_feat.shape[1]}'  # 优化CFG，2026年6月25日
            audio_feat_saved = self.extract_audio_feature(audio_or_feat)  # 优化CFG，2026年6月25日
        elif audio_or_feat.ndim == 3:  # 优化CFG，2026年6月25日
            assert audio_or_feat.shape[1] == self.n_motions, f'Incorrect audio feature length {audio_or_feat.shape[1]}'  # 优化CFG，2026年6月25日
            audio_feat_saved = audio_or_feat  # 优化CFG，2026年6月25日
        else:  # 优化CFG，2026年6月25日
            raise ValueError(f'Incorrect audio input shape {audio_or_feat.shape}')  # 优化CFG，2026年6月25日
        audio_feat = audio_feat_saved.clone()  # 优化CFG，2026年6月25日
        if prev_motion_feat is None:  # 优化CFG，2026年6月25日
            prev_motion_feat = torch.index_select(self.start_motion_feat, 0, emo_index)  # 优化CFG，2026年6月25日
        if prev_audio_feat is None:  # 优化CFG，2026年6月25日
            prev_audio_feat = torch.index_select(self.start_audio_feat, 0, emo_index)  # 优化CFG，2026年6月25日
        prev_audio_feat_saved = prev_audio_feat.clone()  # 优化CFG，2026年6月25日
        cfg_state = torch.full((batch_size,), 2, dtype=torch.long, device=self.device)  # 优化CFG，2026年6月25日
        cfg_is_full = torch.ones(batch_size, dtype=torch.bool, device=self.device)  # 优化CFG，2026年6月25日
        p_AE = 0.1  # 优化CFG，2026年6月25日
        p_E = 0.55  # 优化CFG，2026年6月25日
        if len(self.guiding_conditions) > 0:  # 优化CFG，2026年6月25日
            assert len(self.guiding_conditions) <= 2, 'Only support 1 or 2 CFG conditions!'  # 优化CFG，2026年6月25日
            if 'audio' in self.guiding_conditions and 'emotion' in self.guiding_conditions:  # 优化CFG，2026年6月25日
                mask_flag = torch.rand(batch_size, device=self.device)  # 优化CFG，2026年6月25日
                mask_uncond = mask_flag < p_AE  # 优化CFG，2026年6月25日
                mask_audio_only = (mask_flag >= p_AE) & (mask_flag < p_E)  # 优化CFG，2026年6月25日
                cfg_is_full = ~(mask_uncond | mask_audio_only)  # 优化CFG，2026年6月25日
                cfg_state = torch.where(mask_uncond, torch.zeros_like(cfg_state), torch.where(mask_audio_only, torch.ones_like(cfg_state), cfg_state))  # 优化CFG，2026年6月25日
                audio_null = self.null_audio_feat.expand(batch_size, self.n_motions, -1)  # 优化CFG，2026年6月25日
                prev_audio_null = self.null_audio_feat.expand(batch_size, self.n_prev_motions, -1)  # 优化CFG，2026年6月25日
                emo_null = self.null_emotion_feat.expand(batch_size, -1, -1)  # 优化CFG，2026年6月25日
                emo_real = self.emo_embed(emo_index).unsqueeze(1)  # 优化CFG，2026年6月25日
                audio_cond = torch.where(mask_uncond.view(-1, 1, 1), audio_null, audio_feat_saved)  # 优化CFG，2026年6月25日
                prev_audio_cond = torch.where(mask_uncond.view(-1, 1, 1), prev_audio_null, prev_audio_feat_saved)  # 优化CFG，2026年6月25日
                emo_cond = torch.where((mask_uncond | mask_audio_only).view(-1, 1, 1), emo_null, emo_real)  # 优化CFG，2026年6月25日
                audio_feat = self._cfg_modulate_audio(audio_cond, emo_cond)  # 优化CFG，2026年6月25日
                prev_audio_feat = self._cfg_modulate_audio(prev_audio_cond, emo_cond)  # 优化CFG，2026年6月25日
            elif 'audio' in self.guiding_conditions:  # 优化CFG，2026年6月25日
                mask_audio = torch.rand(batch_size, device=self.device) < p_AE  # 优化CFG，2026年6月25日
                cfg_state = torch.where(mask_audio, torch.zeros_like(cfg_state), torch.ones_like(cfg_state))  # 优化CFG，2026年6月25日
                cfg_is_full = ~mask_audio  # 优化CFG，2026年6月25日
                audio_null = self.null_audio_feat.expand(batch_size, self.n_motions, -1)  # 优化CFG，2026年6月25日
                prev_audio_null = self.null_audio_feat.expand(batch_size, self.n_prev_motions, -1)  # 优化CFG，2026年6月25日
                audio_feat = torch.where(mask_audio.view(-1, 1, 1), audio_null, audio_feat_saved)  # 优化CFG，2026年6月25日
                prev_audio_feat = torch.where(mask_audio.view(-1, 1, 1), prev_audio_null, prev_audio_feat_saved)  # 优化CFG，2026年6月25日
            elif 'emotion' in self.guiding_conditions:  # 优化CFG，2026年6月25日
                mask_emotion = torch.rand(batch_size, device=self.device) < p_AE  # 优化CFG，2026年6月25日
                cfg_state = torch.where(mask_emotion, torch.zeros_like(cfg_state), cfg_state)  # 优化CFG，2026年6月25日
                cfg_is_full = ~mask_emotion  # 优化CFG，2026年6月25日
                emo_null = self.null_emotion_feat.expand(batch_size, -1, -1)  # 优化CFG，2026年6月25日
                emo_real = self.emo_embed(emo_index).unsqueeze(1)  # 优化CFG，2026年6月25日
                emo_cond = torch.where(mask_emotion.view(-1, 1, 1), emo_null, emo_real)  # 优化CFG，2026年6月25日
                audio_feat = self._cfg_modulate_audio(audio_feat_saved, emo_cond)  # 优化CFG，2026年6月25日
                prev_audio_feat = self._cfg_modulate_audio(prev_audio_feat_saved, emo_cond)  # 优化CFG，2026年6月25日
        if time_step is None:  # 优化CFG，2026年6月25日
            time_step = self.diffusion_sched.uniform_sample_t(batch_size)  # 优化CFG，2026年6月25日
        alpha_bar = self.diffusion_sched.alpha_bars[time_step]  # 优化CFG，2026年6月25日
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)  # 优化CFG，2026年6月25日
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)  # 优化CFG，2026年6月25日
        eps = torch.randn_like(motion_feat)  # 优化CFG，2026年6月25日
        motion_feat_noisy = c0 * motion_feat + c1 * eps  # 优化CFG，2026年6月25日
        motion_feat_target = self.denoising_net(motion_feat_noisy, audio_feat, prev_motion_feat, prev_audio_feat, time_step, indicator)  # 优化CFG，2026年6月25日
        if return_cfg_state:  # 优化CFG，2026年6月25日
            return eps, motion_feat_target, motion_feat.detach(), audio_feat_saved.detach(), cfg_is_full.detach(), cfg_state.detach()  # 优化CFG，2026年6月25日
        return eps, motion_feat_target, motion_feat.detach(), audio_feat_saved.detach()  # 优化CFG，2026年6月25日

    @torch.no_grad()
    def sample(self, audio_or_feat, prev_motion_feat=None, prev_audio_feat=None, motion_at_T=None,
               indicator=None, cfg_mode=None, cfg_cond=None, cfg_scale=1.15, flexibility=0,
               dynamic_threshold=None, ret_traj=False, emo_index=None):
        batch_size = audio_or_feat.shape[0]
        cfg_mode = self.cfg_mode if cfg_mode is None else cfg_mode
        cfg_cond = self.guiding_conditions if cfg_cond is None else cfg_cond
        cfg_cond = [c for c in cfg_cond if c in ['audio', 'emotion']]
        cfg_cond, cfg_scale, emo_lip_scale = self._cfg_make_scale(cfg_cond, cfg_scale)  # 优化CFG，2026年6月25日
        print(f'cfg_cond: {cfg_cond}, cfg_scale: {cfg_scale}, emo_lip_scale: {emo_lip_scale}')  # 优化CFG，2026年6月25日

        if audio_or_feat.ndim == 2:
            assert audio_or_feat.shape[1] == 16000 * self.n_motions / self.fps, f'Incorrect audio length {audio_or_feat.shape[1]}'
            audio_feat = self.extract_audio_feature(audio_or_feat)
        elif audio_or_feat.ndim == 3:
            assert audio_or_feat.shape[1] == self.n_motions, f'Incorrect audio feature length {audio_or_feat.shape[1]}'
            audio_feat = audio_or_feat
        else:
            raise ValueError(f'Incorrect audio input shape {audio_or_feat.shape}')

        if prev_motion_feat is None:
            prev_motion_feat = torch.index_select(self.start_motion_feat, 0, emo_index)
        if prev_audio_feat is None:
            prev_audio_feat = torch.index_select(self.start_audio_feat, 0, emo_index)
        if motion_at_T is None:
            motion_at_T = torch.randn((batch_size, self.n_motions, self.motion_feat_dim)).to(self.device)

        audio_feat_in, prev_audio_feat_in, n_entries = self._cfg_build_inputs(audio_feat.clone(), prev_audio_feat.clone(), cfg_cond, emo_index)  # 优化CFG，2026年6月25日
        prev_motion_feat_in = torch.cat([prev_motion_feat] * n_entries, dim=0)
        indicator_in = torch.cat([indicator] * n_entries, dim=0) if indicator is not None else None

        traj = {self.diffusion_sched.num_steps: motion_at_T}
        for t in range(self.diffusion_sched.num_steps, 0, -1):
            z = torch.randn_like(motion_at_T) if t > 1 else torch.zeros_like(motion_at_T)
            alpha = self.diffusion_sched.alphas[t]
            alpha_bar = self.diffusion_sched.alpha_bars[t]
            alpha_bar_prev = self.diffusion_sched.alpha_bars[t - 1]
            sigma = self.diffusion_sched.get_sigmas(t, flexibility)
            motion_at_t = traj[t]
            motion_in = torch.cat([motion_at_t] * n_entries, dim=0)
            step_in = torch.tensor([t] * batch_size, device=self.device)
            step_in = torch.cat([step_in] * n_entries, dim=0)
            results = self.denoising_net(motion_in, audio_feat_in, prev_motion_feat_in, prev_audio_feat_in, step_in, indicator_in)
            if dynamic_threshold:
                dt_ratio, dt_min, dt_max = dynamic_threshold
                abs_results = results[:, -self.n_motions:].reshape(batch_size * n_entries, -1).abs()
                s = torch.quantile(abs_results, dt_ratio, dim=1)
                s = torch.clamp(s, min=dt_min, max=dt_max)
                s = s[..., None, None]
                results = torch.clamp(results, min=-s, max=s)
            results = results.chunk(n_entries)
            if cfg_mode in ['independent', 'incremental']:  # 优化CFG，2026年6月25日
                target_theta = self._cfg_target_theta(results, cfg_cond, cfg_scale, emo_lip_scale)  # 优化CFG，2026年6月25日
            else:
                raise NotImplementedError(f'Unknown cfg_mode {cfg_mode}')
            if self.target == 'noise':
                c0 = 1 / torch.sqrt(alpha)
                c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
                motion_next = c0 * (motion_at_t - c1 * target_theta) + sigma * z
            elif self.target == 'sample':
                c0 = (1 - alpha_bar_prev) * torch.sqrt(alpha) / (1 - alpha_bar)
                c1 = (1 - alpha) * torch.sqrt(alpha_bar_prev) / (1 - alpha_bar)
                motion_next = c0 * motion_at_t + c1 * target_theta + sigma * z
            else:
                raise ValueError('Unknown target type: {}'.format(self.target))
            traj[t - 1] = motion_next.detach()
            traj[t] = traj[t].cpu()
            if not ret_traj:
                del traj[t]
        if ret_traj:
            return traj, motion_at_T, audio_feat  # 优化CFG，2026年6月25日
        return traj[0], motion_at_T, audio_feat  # 优化CFG，2026年6月25日
