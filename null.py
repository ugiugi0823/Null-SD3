import ptp_utils, seq_aligner
import torch, inspect, abc, shutil


import numpy as np
import torch.nn.functional as nnf

from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline, AutoencoderKL, StableDiffusionPipeline, DDIMScheduler
from typing import Optional, Union, Tuple, List, Callable, Dict
from diffusers.utils import load_image
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm
from torch.optim.adam import Adam
from PIL import Image
from compel import Compel, ReturnedEmbeddingsType





LOW_RESOURCE = False
NUM_DDIM_STEPS = 50
GUIDANCE_SCALE = 5.0
MAX_NUM_WORDS = 77
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps




def retrieve_latents(
        encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
    ):
        if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
            return encoder_output.latent_dist.sample(generator)
        elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
            return encoder_output.latent_dist.mode()
        elif hasattr(encoder_output, "latents"):
            return encoder_output.latents
        else:
            raise AttributeError("Could not access latents of provided encoder_output")

def load_img(image_path, do_1024=False):
    
    
    
    
    image = load_image(image_path)
    do_1024=False
    if do_1024:
        if image.size[0] != 1024:
            image = image.resize((1024, 1024)) 
        
            
    else:
        if image.size[0] != 512:
            image = image.resize((512, 512)) 
            
        

    print("🌊 image resize = ",image.size[0])



    return image


class NullInversion:

    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray], step_index: int):
        # prev_sample = self.scheduler.step(model_output, timestep, sample, return_dict=False)[0]
        shift= 1.0        
        num_train_timesteps = self.scheduler.config.num_train_timesteps
        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        sigmas = timesteps / num_train_timesteps
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = sigmas * num_train_timesteps

        # self._step_index = None
        # self._begin_index = None

        self.sigmas = sigmas.to("cpu")  # to avoid too much CPU/GPU communication
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()
        
        
        s_churn = 0.0
        s_tmin = 0.0
        s_tmax= float("inf")
        s_noise = 1.0
        
        num_inference_steps = NUM_DDIM_STEPS
        
        timesteps = np.linspace(
            self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
        )

        sigmas = timesteps / self.scheduler.config.num_train_timesteps
        sigmas = self.scheduler.config.shift * sigmas / (1 + (self.scheduler.config.shift - 1) * sigmas)
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)

        timesteps = sigmas * self.scheduler.config.num_train_timesteps
        self.timesteps = timesteps.to(device=device)
        self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)
        
        
        
        # print("⭐️self.step_index", step_index)
        
        
        
        sigma = self.sigmas[step_index]

        gamma = min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0
        
        generator = torch.Generator("cuda").manual_seed(33)
        noise = randn_tensor(
            model_output.shape, dtype=model_output.dtype, device=model_output.device, generator=generator
        )

        eps = noise * s_noise
        sigma_hat = sigma * (gamma + 1)

        if gamma > 0:
            sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        # NOTE: "original_sample" should not be an expected prediction_type but is left in for
        # backwards compatibility

        # if self.config.prediction_type == "vector_field":

        denoised = sample - model_output * sigma
        # 2. Convert to an ODE derivative
        derivative = (sample - denoised) / sigma_hat

        dt = self.sigmas[step_index + 1] - sigma_hat

        prev_sample = sample + derivative * dt
        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)
        
        # self.scheduler._step_index += 1
        
        return prev_sample
    
    
    def _sigma_to_t(self, sigma):
        return sigma * self.scheduler.config.num_train_timesteps
    
    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self.scheduler._step_index
    
    def _init_step_index(self, timestep):
        if self.scheduler.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.scheduler.timesteps.device)
            self._step_index = self.scheduler.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray], step_index: int):
        
        # time_term = self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps 
        # timestep, next_timestep = min(timestep, 999), min(timestep + time_term, 999)
        shift= 1.0
        num_train_timesteps = self.scheduler.config.num_train_timesteps
        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        sigmas = timesteps / num_train_timesteps
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = sigmas * num_train_timesteps

        # self._step_index = None
        # self._begin_index = None

        self.sigmas = sigmas.to("cpu")  # to avoid too much CPU/GPU communication
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()
        
        
        s_churn = 0.0
        s_tmin = 0.0
        s_tmax= float("inf")
        s_noise = 1.0
        
        num_inference_steps = NUM_DDIM_STEPS
        
        timesteps = np.linspace(
            self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
        )

        sigmas = timesteps / self.scheduler.config.num_train_timesteps
        sigmas = self.scheduler.config.shift * sigmas / (1 + (self.scheduler.config.shift - 1) * sigmas)
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)

        timesteps = sigmas * self.scheduler.config.num_train_timesteps
        self.timesteps = timesteps.to(device=device)
        self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)
        
        
        
        print("⭐️self.step_index", step_index)
        num_inference_steps_1 = num_inference_steps -1 
        num_inference_steps_1_step_index = num_inference_steps_1 - step_index
        print("⭐️num_inference_steps_1_step_index", num_inference_steps_1_step_index)
        
        sigma = self.sigmas[num_inference_steps_1_step_index]

        gamma = min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0
        
        generator = torch.Generator("cuda").manual_seed(33)
        noise = randn_tensor(
            model_output.shape, dtype=model_output.dtype, device=model_output.device, generator=generator
        )

        eps = noise * s_noise
        sigma_hat = sigma * (gamma + 1)

        if gamma > 0:
            sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        # NOTE: "original_sample" should not be an expected prediction_type but is left in for
        # backwards compatibility

        # if self.config.prediction_type == "vector_field":

        denoised = sample - model_output * sigma
        # 2. Convert to an ODE derivative
        derivative = (sample - denoised) / sigma_hat

        dt = self.sigmas[step_index + 1] - sigma_hat

        prev_sample = sample + derivative * dt
        # Cast sample back to model compatible dtype
        next_sample = prev_sample.to(model_output.dtype)
        
        # self.scheduler._step_index += 1
 
        return next_sample

    def get_noise_pred_single(self, latents, t, context, context_p, add_time_ids):        
        added_cond_kwargs = {"text_embeds": context_p, "time_ids": add_time_ids}
        
        # latents = torch.cat([latents] * 2)
        
        
        t = t.expand(latents.shape[0])
        t=t.to(torch.float32)
        latents=latents.to(torch.float32)
        context=context.to(torch.float32)
        context_p=context_p.to(torch.float32)
        joint_attention_kwargs = None
        
        
        
        # print("🤪"*40)
        # print("latent_model_input",latents.size())
        # print("timestep", t.size())


        # print("prompt_embeds",context.size())
        # print("pooled_prompt_embeds", context_p.size())
        
        # print("🤪"*40)
        
        
        
        
        
        
        # [o]norm_hidden_states torch.Size([2, 4096, 1536])
        # [o]norm_encoder_hidden_states torch.Size([2, 154, 1536])
        # norm_hidden_states torch.Size([1, 1024, 1536])
        # norm_encoder_hidden_states torch.Size([1, 154, 1536])
        # ---
        # [o]hidden_states torch.Size([2, 4096, 1536])
        # [o]encoder_hidden_states torch.Size([2, 154, 1536])
        # hidden_states torch.Size([1, 1024, 1536])
        # encoder_hidden_states torch.Size([1, 154, 1536])
        
        
        
        noise_pred = self.model.transformer(
                    hidden_states=latents,
                    timestep=t,
                    encoder_hidden_states=context,
                    pooled_projections=context_p,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=False,
                )[0]
        
        # print("😿 noise_pred = ",noise_pred)
        latents_dtype = latents.dtype
        if latents.dtype != latents_dtype:
            if torch.backends.mps.is_available():
                 # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                latents = latents.to(latents_dtype)

        
        # noise_pred = self.model.transformer(
        #                             hidden_states=latents, 
                                     
        #                             encoder_hidden_states=context,
        #                             timestep=t,
                                     
        #                              )[0]
        return noise_pred
    

    
    


    def get_noise_pred(self, latents, t, is_forward=True, context=None, context_p=None, step=None):
        latents_input = torch.cat([latents] * 2)
        
        context = context if context is not None else self.context
        context_p = context_p if context_p is not None else self.context_p
        
        
        
        
        guidance_scale = 1 if is_forward else GUIDANCE_SCALE
    
        
        # latents_input = self.scheduler.scale_model_input(latents_input, t)
        
        
        
        t = t.expand(latents.shape[0])
        t=t.to(torch.float32)
        latents=latents.to(torch.float32)
        context=context.to(torch.float32)
        context_p=context_p.to(torch.float32)
        joint_attention_kwargs = None
        
        
        
        noise_pred = self.model.transformer(
            hidden_states=latents,
            timestep=t,
            encoder_hidden_states=context,
            pooled_projections=context_p,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=False,
        )[0]
        

       
        
        
        
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents, step)
        else:
            latents = self.prev_step(noise_pred, t, latents, step)
        return latents
    
    
# 여기는 손을 봤음 
    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 1.5305 * latents.detach()
        # latents = latents.to(torch.float32)
        
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]        
            image = (image * 255).round().astype(np.uint8)
            
            
            
        
        return image
    
    
    






    @torch.no_grad()
    def image2latent(self, image):

        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                
                # image = torch.from_numpy(image).float() / 127.5 - 1   
                # image = image.permute(2, 0, 1).unsqueeze(0).to("cuda")
                self.model.to(device)
                image = self.model.image_processor.preprocess(image).to("cuda")
                model_dtype = self.model.transformer.dtype
                image = image.to(model_dtype)
                
                generator = torch.Generator("cuda").manual_seed(33)
                

                init_latents = retrieve_latents(self.model.vae.encode(image), generator=generator)
                latents = (init_latents - self.model.vae.config.shift_factor) * self.model.vae.config.scaling_factor

                # latents = self.model.vae.encode(image.to(self.model.transformer.dtype)).latent_dist.sample(generator)
                # latents = self.model.vae.encode(image.to(self.model.unet.dtype))['latent_dist'].mean
                
                
                if torch.isnan(latents).any():
                    print("wldnjsdfjklsfjkld")
                    # raise ValueError("NaN detected in image2latent!")
                    
                
                # latents = latents * 1.5305

                print("🌊 latents.size = ",latents.size())    
        return latents




    @torch.no_grad()
    def init_prompt(self, prompt: str):
        
        # prompts = [prompt] * 3
        # negative_prompts = [" "] * 3
        
        # prompt_1, prompt_2, prompt_3 = prompts
        # negative_prompt_1, negative_prompt_2, negative_prompt_3 = negative_prompts

        
        prompt_1 = prompt
        prompt_2 = prompt
        prompt_3 = prompt
        negative_prompt_1 = " "
        negative_prompt_2 = " "
        negative_prompt_3 = " "
        
        self.model.to(device)
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.model.encode_prompt(
        prompt=prompt_1,
        prompt_2=prompt_2,
        prompt_3=prompt_3,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=negative_prompt_1,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        )
        
        
        
        
        
        # compel = Compel(
        # tokenizer=[self.model.tokenizer, self.model.tokenizer_2] ,
        # text_encoder=[self.model.text_encoder, self.model.text_encoder_2],
        # returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        # requires_pooled=[False, True]
        # )
        
        # prompt_embeds, pooled_prompt_embeds = compel(prompt)
        # negative_prompt_embeds, negative_pooled_prompt_embeds = compel("") 
   
        self.model.vae_scale_factor = 2 ** (len(self.model.vae.config.block_out_channels) - 1)
        self.model.default_sample_size = self.model.transformer.config.sample_size
        
        height = self.model.default_sample_size * self.model.vae_scale_factor
        width =  self.model.default_sample_size * self.model.vae_scale_factor
        
        
        
        

        original_size =  (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)    
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        
        # passed_add_embed_dim = (
        #     self.model.transformer.config.addition_time_embed_dim * len(add_time_ids) + self.model.text_encoder_2.config.projection_dim
        # )
        # expected_add_embed_dim = self.model.transformer.add_embedding.linear_1.in_features

        

        # if expected_add_embed_dim != passed_add_embed_dim:
        #     raise ValueError(
        #         f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        #     )

        add_time_ids = torch.tensor([add_time_ids], dtype=self.model.transformer.dtype).to(self.model.device)
        batch_size = prompt_embeds.shape[0]
        num_images_per_prompt = 1
        add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)

        

        
        self.context = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        self.context_p = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        self.add_time_ids = torch.cat([add_time_ids, add_time_ids])        
        self.prompt = prompt


    @torch.no_grad()
    def ddim_loop(self, latent):

        uncond_embeddings_p, cond_embeddings_p = self.context_p.chunk(2)
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        add_time_ids1, add_time_ids2 = self.add_time_ids.chunk(2)
        
        
        # cond_embeddings_p = self.context_p
        # cond_embeddings = self.context
        # add_time_ids2 = self.add_time_ids
        
        
        
        
        
        all_latent = [latent]
        latent = latent.clone().detach()
        
        timesteps = None
        num_inference_steps = NUM_DDIM_STEPS
        timesteps, num_inference_steps = retrieve_timesteps(self.model.scheduler, num_inference_steps, device, timesteps)
        
        

        for i in range(NUM_DDIM_STEPS):
            
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1].unsqueeze(0)
            
            
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings, cond_embeddings_p, add_time_ids2)
            # noise_pred_1, noise_pred_2 = noise_pred.chunk(2)
            
            latent = self.next_step(noise_pred, t, latent, i)
            all_latent.append(latent)
            
            
        return all_latent


    @property
    def scheduler(self):
        return self.model.scheduler
    

    

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent(image)
        if latent is None:
            raise ValueError("latents가 None입니다. image2latent 함수가 올바르게 동작하지 않았습니다.")
               
        image_rec = self.latent2image(latent)
        if image_rec is None:
            raise ValueError("image_rec가 None입니다. latent2image 함수가 올바르게 동작하지 않았습니다.")

        ddim_latents = self.ddim_loop(latent)
        if ddim_latents is None:
            raise ValueError("ddim_latent가  None입니다. ddim_loop 함수가 올바르게 동작하지 않았습니다.")

        return image_rec, ddim_latents
    
    

    def null_optimization(self, latents, num_inner_steps, epsilon):
    
        uncond_embeddings_p, cond_embeddings_p = self.context_p.chunk(2)
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        add_time_ids1, add_time_ids2 = self.add_time_ids.chunk(2)
        
            
        uncond_embeddings_list = []
        uncond_embeddings_p_list = []
        add_time_ids1_list = []
        latent_cur = latents[-1]
        # print(latent_cur.size())

        # Set total for tqdm
        # total_iterations = num_inner_steps * NUM_DDIM_STEPS 
        total_iterations = NUM_DDIM_STEPS 
        bar = tqdm(total=total_iterations)
        
        
        for i in range(NUM_DDIM_STEPS):
            
            
            uncond_embeddings = uncond_embeddings.clone().detach().requires_grad_(True)
            uncond_embeddings_p = uncond_embeddings_p.clone().detach().requires_grad_(True)
            add_time_ids1 = add_time_ids1.clone().detach().requires_grad_(True)
            
            optimizer = Adam([uncond_embeddings, uncond_embeddings_p], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            # print("🩵🩵 timestep =", t)
            # print("🩵🩵 timestep =", t.size())
            
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings, cond_embeddings_p, add_time_ids2)

            
            
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings, uncond_embeddings_p, add_time_ids1)
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur, j)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                loss_item = loss.item()
                optimizer.zero_grad()
                loss.backward()
                if torch.isnan(uncond_embeddings.grad).any():
                    
                    print("Nan!!")
                    # bar.update()
                    break
                optimizer.step()
                loss_item = loss.item()
                bar.set_description(f"Step {i}, Iteration {j}/{num_inner_steps}, Loss: {loss_item:.4f}")
                
                if loss_item < epsilon + i * 2e-5:
                    break
                
        
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            uncond_embeddings_p_list.append(uncond_embeddings_p[:1].detach())
            
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                context_p = torch.cat([uncond_embeddings_p, cond_embeddings_p])
                add_time_ids = torch.cat([add_time_ids1, add_time_ids2]) 
                latent_cur = self.get_noise_pred(latent_cur, t, False, context, context_p=context_p, step=i)
            bar.update()
            
        
        return uncond_embeddings_list, uncond_embeddings_p_list


    def invert(self, image_path: str, prompt: str, num_inner_steps=50, early_stop_epsilon=1e-5, verbose=False, do_1024=False):
        self.init_prompt(prompt)
        ptp_utils.register_attention_control(self.model, None)
        image_gt = load_img(image_path, do_1024)
        if verbose:
            print("----- DDIM inversion...")
        
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        

        if verbose:
            print("----- Null-text optimization...")
        uncond_embeddings, uncond_embeddings_p = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
        # uncond_embeddings = None
        # uncond_embeddings_p = None
        
        
        return (image_gt, image_rec), ddim_latents[-1], uncond_embeddings, uncond_embeddings_p


    def __init__(self, model):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False)
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.prompt = None
        self.add_time_ids = None
        self.context = None
        self.context_p = None

