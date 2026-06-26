from .emotion_dit_cfg import DiffusionSchedule, DenoisingNetwork, DitTalkingHead as _DitTalkingHead


class DitTalkingHead(_DitTalkingHead):
    def _cfg_make_scale(self, cfg_cond, cfg_scale):  # 优化CFG，2026年6月25日
        if not isinstance(cfg_scale, list):  # 优化CFG，2026年6月25日
            cfg_scale = [cfg_scale] * len(cfg_cond)  # 优化CFG，2026年6月25日
        elif len(cfg_scale) < len(cfg_cond):  # 优化CFG，2026年6月25日
            cfg_scale = cfg_scale + [cfg_scale[-1]] * (len(cfg_cond) - len(cfg_scale))  # 优化CFG，2026年6月25日
        else:  # 优化CFG，2026年6月25日
            cfg_scale = cfg_scale[:len(cfg_cond)]  # 优化CFG，2026年6月25日
        if len(cfg_cond) > 0:  # 优化CFG，2026年6月25日
            cfg_cond, cfg_scale = zip(*sorted(zip(cfg_cond, cfg_scale), key=lambda x: ['audio', 'emotion'].index(x[0])))  # 优化CFG，2026年6月25日
            cfg_cond, cfg_scale = list(cfg_cond), list(cfg_scale)  # 优化CFG，2026年6月25日
        else:  # 优化CFG，2026年6月25日
            cfg_cond, cfg_scale = [], []  # 优化CFG，2026年6月25日
        return cfg_cond, cfg_scale, None  # 优化CFG，2026年6月25日

    def _cfg_target_theta(self, results, cfg_cond, cfg_scale, emo_lip_scale=None):  # 优化CFG，2026年6月25日
        result_uncond = results[0][:, -self.n_motions:]  # 优化CFG，2026年6月25日
        if 'audio' in cfg_cond and 'emotion' in cfg_cond and len(results) == 3:  # 优化CFG，2026年6月25日
            result_audio = results[cfg_cond.index('audio') + 1][:, -self.n_motions:]  # 优化CFG，2026年6月25日
            result_full = results[cfg_cond.index('emotion') + 1][:, -self.n_motions:]  # 优化CFG，2026年6月25日
            audio_scale = cfg_scale[cfg_cond.index('audio')]  # 优化CFG，2026年6月25日
            emotion_scale = cfg_scale[cfg_cond.index('emotion')]  # 优化CFG，2026年6月25日
            target_theta = result_uncond + audio_scale * (result_audio - result_uncond)  # 优化CFG，2026年6月25日
            target_theta = target_theta + emotion_scale * (result_full - result_audio)  # 优化CFG，2026年6月25日
            return target_theta  # 优化CFG，2026年6月25日
        target_theta = result_uncond  # 优化CFG，2026年6月25日
        for i in range(0, len(results) - 1):  # 优化CFG，2026年6月25日
            target_theta += cfg_scale[i] * (results[i + 1][:, -self.n_motions:] - results[0][:, -self.n_motions:])  # 优化CFG，2026年6月25日
        return target_theta  # 优化CFG，2026年6月25日
