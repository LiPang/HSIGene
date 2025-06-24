import einops
import torch
import torch.nn.functional as F

from einops import rearrange, repeat
from torchvision.utils import make_grid
from peft import LoraConfig, get_peft_model

from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import disabled_train


class CRSControlNet(LatentDiffusion):
    def __init__(self, mode, global_strength=1.0, text_strength=1.0, local_control_config=None, global_content_control_config=None,global_text_control_config=None,metadata_config=None, use_lora=False, lora_r=16, lora_alpha=16, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mode in ['local', 'global', 'CRS', 'UNet']
        self.mode = mode
        self.global_strength = global_strength
        self.text_strength = text_strength
        # if self.mode in ['local', 'CRS']:
        #     self.local_adapter = instantiate_from_config(local_control_config)
        #     self.local_control_scales = [1.0] * 5
        # if self.mode in ['global', 'CRS']:
        #     self.metadata_emb=instantiate_from_config(metadata_config).cuda()
        #     self.global_content_adapter = instantiate_from_config(global_content_control_config)
        #     self.global_text_adapter = instantiate_from_config(global_text_control_config)
        self.local_adapter = instantiate_from_config(local_control_config)
        self.local_control_scales = [1.0] * 13
        self.metadata_emb=instantiate_from_config(metadata_config).cuda()
        self.global_content_adapter = instantiate_from_config(global_content_control_config)
        self.global_text_adapter = instantiate_from_config(global_text_control_config)

        # LoRA 在加载预训练权重后再注入，避免 load_state_dict 的键值不匹配。
        self._want_lora = use_lora
        self._lora_r = lora_r
        self._lora_alpha = lora_alpha
        self.use_lora = False  # 尚未真正注入 LoRA


    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, bs=bs, *args, **kwargs)
        if len(batch['local_conditions']) != 0:
            local_conditions = batch['local_conditions']
            if bs is not None:
                local_conditions = local_conditions[:bs]
            local_conditions = local_conditions.to(self.device)
            local_conditions = einops.rearrange(local_conditions, 'b h w c -> b c h w')
            local_conditions = local_conditions.to(memory_format=torch.contiguous_format).float()
        else:
            local_conditions = torch.zeros(1,1,1,1).to(self.device).to(memory_format=torch.contiguous_format).float()
        if len(batch['global_conditions']) != 0:
            global_conditions = batch['global_conditions']
            if bs is not None:
                global_conditions = global_conditions[:bs]
            global_conditions = global_conditions.to(self.device).to(memory_format=torch.contiguous_format).float()
        else:
            global_conditions = torch.zeros(1,1).to(self.device).to(memory_format=torch.contiguous_format).float()

        if len(batch['metadata']) != 0:
            metadata = batch['metadata']
            if bs is not None:
                metadata = metadata[:bs]
            metadata=metadata.to(self.device).to(memory_format=torch.contiguous_format).float()
        return x, dict(c_crossattn=[c], metadata=[metadata],local_control=[local_conditions], global_control=[global_conditions])

    def apply_model(self, x_noisy, t, cond, metadata=None,global_strength=None, text_strength=None, metadata_strength=1, *args, **kwargs):
        assert isinstance(cond, dict)
        if metadata==None:
            metadata=cond['metadata']
        
        current_global_strength = global_strength if global_strength is not None else self.global_strength
        current_text_strength = text_strength if text_strength is not None else self.text_strength

        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond['c_crossattn'], 1)

        assert cond['global_control'][0] != None
        metadata = self.metadata_emb(metadata)
        # content_t,meta_t=cond['global_control'][0].chunk(2,dim=1)
        content_t = cond['global_control'][0]
        global_control = self.global_content_adapter(content_t)
        cond_txt = self.global_text_adapter(cond_txt)

        # Normalize and scale features independently for stability
        cond_txt = F.normalize(cond_txt, p=2, dim=-1) * current_text_strength
        global_control = F.normalize(global_control, p=2, dim=-1) * current_global_strength
        
        cond_txt = torch.cat([cond_txt, global_control], dim=1)

        assert cond['local_control'][0] != None
        local_control = torch.cat(cond['local_control'], 1)
        local_control = self.local_adapter(x=x_noisy, timesteps=t, context=cond_txt, local_conditions=local_control)
        local_control = [c * scale for c, scale in zip(local_control, self.local_control_scales)]

        eps = diffusion_model(x=x_noisy, timesteps=t, metadata=metadata, context=cond_txt,
                              local_control=local_control, meta=True)

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=1, n_row=2, sample=True, ddim_steps=50, ddim_eta=0.0, plot_denoise_rows=False,
                   plot_diffusion_rows=False, unconditional_guidance_scale=1.0, **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)

        c_meta=c["metadata"][0][:N]
        c_cat = c["local_control"][0][:N]
        c_global = c["global_control"][0][:N]
        c = c["c_crossattn"][0][:N]

        
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        # Log local image condition (e.g., canny edge map)
        log["local_image_condition"] = c_cat * 2.0 - 1.0
        # Log text condition
        log["text_condition"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)
        # Log source for global image condition (CLIP feature is extracted from this)
        log["global_condition_source"] = batch['jpg'].permute(0, 3, 1, 2)[:N] * \
                                         (batch['global_conditions'].max(-1)[0] > 0)[:N].view(-1, 1, 1, 1)

        if plot_diffusion_rows:
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            samples, z_denoise_row = self.sample_log(cond={"local_control": [c_cat], "c_crossattn": [c], "global_control": [c_global]},
                                                     batch_size=N, ddim=use_ddim,metadata=[c_meta],
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat
            uc_global = torch.zeros_like(c_global)
            uc_full = {"local_control": [uc_cat], "c_crossattn": [uc_cross], "global_control": [uc_global]}#,"metadata":[c_meta]
            samples_cfg, _ = self.sample_log(cond={"local_control": [c_cat], "c_crossattn": [c], "global_control": [c_global]},
                                             batch_size=N, ddim=use_ddim,metadata=[c_meta],
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size,metadata, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        if self.mode == 'global':
            h, w = 512, 512
        else:
            _, _, h, w = cond["local_control"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps,batch_size, shape, metadata, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params, params_count = [], 0
        if self.mode in ['local']:
            params += list(self.local_adapter.parameters())
            print(f'Training local_adapter {sum(p.numel() for p in params) / 1e6 - params_count}M')
            params_count = sum(p.numel() for p in params) / 1e6
        if self.mode in ['global']:
            params += list(self.global_text_adapter.parameters())
            params += list(self.cond_stage_model.parameters())
            params += list(self.global_content_adapter.parameters())
            print(f'Training global_adapter {sum(p.numel() for p in params) / 1e6 - params_count}M')
            params_count = sum(p.numel() for p in params) / 1e6
        if self.mode in ['CRS']:
            params += list(self.local_adapter.parameters())
            params += list(self.global_text_adapter.parameters())
            params += list(self.global_content_adapter.parameters())
        if self.use_lora:
            print("Training LoRA parameters.")
            lora_params = []
            for n, p in self.model.diffusion_model.named_parameters():
                if "lora_" in n:
                    lora_params.append(p)
            params += lora_params
            print(f'Training LoRA params {sum(p.numel() for p in lora_params) / 1e6}M')
        elif not self.sd_locked:
            # params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
            print(f'Training diffusion_model {sum(p.numel() for p in params) / 1e6 - params_count}M')
            params_count = sum(p.numel() for p in params) / 1e6
        opt = torch.optim.AdamW(params, lr=lr)
        # self.reset_requires_grad([opt])
        return opt


    def reset_requires_grad(self, optimizers):
        if not optimizers:
            return

        trainable_set = set()
        for opt in optimizers:
            for group in opt.param_groups:
                for p in group['params']:
                    trainable_set.add(p)

        for module in self.modules():
            module_params = list(module.parameters())
            if not module_params:
                continue

            if not any(p in trainable_set for p in module_params):
                module.eval()
                module.train = disabled_train
                for p in module_params:
                    p.requires_grad = False

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            if self.mode in ['local', 'CRS']:
                self.local_adapter = self.local_adapter.cuda()
            if self.mode in ['global', 'CRS']:
                self.global_text_adapter = self.global_text_adapter.cuda()
                self.global_content_adapter = self.global_content_adapter.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            if self.mode in ['local', 'CRS']:
                self.local_adapter = self.local_adapter.cpu()
            if self.mode in ['global', 'CRS']:
                self.global_text_adapter = self.global_text_adapter.cpu()
                self.global_content_adapter = self.global_content_adapter.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()

    def enable_lora(self, r: int | None = None, alpha: int | None = None,
                    target_modules=None, dropout: float = 0.1, bias: str = "none"):
        """Inject LoRA layers after calling load_state_dict."""
        if self.use_lora:
            print("LoRA already enabled, skipping duplicate injection.")
            return

        r = r or self._lora_r
        alpha = alpha or self._lora_alpha

        print(f"[LoRA] Injecting LoRA into UNet (diffusion_model), r={r}, alpha={alpha}")
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias=bias,
        )
        
        # Wrap the entire UNet model (diffusion_model), peft will automatically find target_modules
        peft_model = get_peft_model(self.model.diffusion_model, lora_config)
        peft_model.print_trainable_parameters()
        self.model.diffusion_model = peft_model
        
        self.use_lora = True
