import cv2, lpips
import numpy as np
import torch, datetime, pytz

from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Union, Tuple, List, Callable, Dict

from tqdm.notebook import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from diffusers.utils.torch_utils import randn_tensor

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img


def view_images(images, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_image = np.ones(images[0].size, dtype=np.uint8) * 255
    
    images = [np.array(image).astype(np.uint8) if isinstance(image, Image.Image) else image.astype(np.uint8) for image in images]
    # images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    images += [empty_image] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    seoul_tz = pytz.timezone("Asia/Seoul")
    current_time = datetime.datetime.now(seoul_tz).strftime("%Y-%m-%dT%H-%M-%S")
    pil_img.save(f"./result/result_{current_time}.png")
    print(f"Image saved as ./result/result_{current_time}.png ") 
    
    
    
 
        
        
def save_individual_images(images,directory="./result"):
    if not isinstance(images, list):
        images = [images]
        
    if len(images) < 3:
        raise ValueError("At least three images are required to compare index 0 and 2.")
    
    # Saving images
    pil_img = Image.fromarray(images[2].astype(np.uint8))
    pil_img2 = Image.fromarray(images[1].astype(np.uint8))
    
    seoul_tz = pytz.timezone("Asia/Seoul")
    current_time = datetime.datetime.now(seoul_tz).strftime("%Y-%m-%dT%H-%M-%S")
    
    
    
    percept = lpips.LPIPS(net='vgg').cuda()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) 
    
    
    # Comparing images at index 0 and 2
    images = np.array(images)
    image0 = images[0].astype(np.uint8)
    image1 = images[1].astype(np.uint8)
    image2 = images[2].astype(np.uint8)
    
    imageA_t = transform(image0).unsqueeze(0).cuda()
    imageB_t = transform(image2).unsqueeze(0).cuda()
    
    imageC_t = transform(image1).unsqueeze(0).cuda()
    
    
    
    
    psnr_value = psnr(image0, image2)
    ssim_value, _ = ssim(image0, image2, full=True, channel_axis=2,win_size=7)
    lpips_value = percept(imageA_t, imageB_t).item()
    
    
    psnr_value_1 = psnr(image0, image1)
    ssim_value_1, _ = ssim(image0, image1, full=True, channel_axis=2,win_size=7)
    lpips_value_1 = percept(imageA_t, imageC_t).item()
    
    
    print(f"⛴️ PSNR original vs new: {psnr_value_1:.2f}")
    print(f"⛴️ SSIM original vs new: {ssim_value_1:.3f}")  
    print(f"⛴️ LPIPS original vs new: {lpips_value_1:.3f}") 
    
    
    
    print(f"🔥 PSNR original vs new: {psnr_value:.2f}")
    print(f"🔥 SSIM original vs new: {ssim_value:.3f}")  
    print(f"🔥 LPIPS original vs new: {lpips_value:.3f}")
    
    file_path = f"{directory}/result_{current_time}_{psnr_value}_new.png"
    file_path_2 = f"{directory}/result_{current_time}_ori.png"
    pil_img2.save(file_path_2)
    pil_img.save(file_path)
    print(f"Image saved as {file_path}")   
    

def diff_individual(images):
    
    if not isinstance(images, list):
        print("images not list")
        images = [images]
        
    
    
    percept = lpips.LPIPS(net='vgg').cuda()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) 
    
    
    # Comparing images at index 0 and 2
    image0 = images[0].astype(np.uint8)
    image2 = images[1].astype(np.uint8)
    
    imageA_t = transform(image0).unsqueeze(0).cuda()
    imageB_t = transform(image2).unsqueeze(0).cuda()
    
    
    
    
    psnr_value = psnr(image0, image2)
    ssim_value, _ = ssim(image0, image2, full=True, channel_axis=2,win_size=7)
    lpips_value = percept(imageA_t, imageB_t).item()
    
    
    
    
    
    print(f"🔥 PSNR original vs new: {psnr_value:.2f}")
    print(f"🔥 SSIM original vs new: {ssim_value:.3f}")  
    print(f"🔥 LPIPS original vs new: {lpips_value:.3f}") 
    


def prev_step_local(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray], step_index: int):
    prev_sample = self.scheduler.step(model_output, timestep, sample, return_dict=False)[0]
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # shift= 1.0        
    # num_train_timesteps = self.scheduler.config.num_train_timesteps
    # timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
    # timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

    # sigmas = timesteps / num_train_timesteps
    # sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

    # self.timesteps = sigmas * num_train_timesteps

    # # self._step_index = None
    # # self._begin_index = None

    # self.sigmas = sigmas.to("cpu")  # to avoid too much CPU/GPU communication
    # self.sigma_min = self.sigmas[-1].item()
    # self.sigma_max = self.sigmas[0].item()
    
    
    # s_churn = 0.0
    # s_tmin = 0.0
    # s_tmax= float("inf")
    # s_noise = 1.0
    
    # num_inference_steps = 50
    
    # timesteps = np.linspace(
    #     _sigma_to_t(self, self.sigma_max), _sigma_to_t(self, self.sigma_min), num_inference_steps
    # )

    # sigmas = timesteps / self.scheduler.config.num_train_timesteps
    # sigmas = self.scheduler.config.shift * sigmas / (1 + (self.scheduler.config.shift - 1) * sigmas)
    # sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)

    # timesteps = sigmas * self.scheduler.config.num_train_timesteps
    # self.timesteps = timesteps.to(device=device)
    # self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

    # # Upcast to avoid precision issues when computing prev_sample
    # sample = sample.to(torch.float32)
    
    
    
    # # print("⭐️self.step_index", step_index)
    
    
    
    # sigma = self.sigmas[step_index]

    # gamma = min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0
    
    # generator = torch.Generator("cuda").manual_seed(33)
    # noise = randn_tensor(
    #     model_output.shape, dtype=model_output.dtype, device=model_output.device, generator=generator
    # )

    # eps = noise * s_noise
    # sigma_hat = sigma * (gamma + 1)

    # if gamma > 0:
    #     sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

    # # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
    # # NOTE: "original_sample" should not be an expected prediction_type but is left in for
    # # backwards compatibility

    # # if self.config.prediction_type == "vector_field":

    # denoised = sample - model_output * sigma
    # # 2. Convert to an ODE derivative
    # derivative = (sample - denoised) / sigma_hat

    # dt = self.sigmas[step_index + 1] - sigma_hat

    # prev_sample = sample + derivative * dt
    # # Cast sample back to model compatible dtype
    # prev_sample = prev_sample.to(model_output.dtype)
    
    # # self.scheduler._step_index += 1
    
    return prev_sample


def _sigma_to_t(self, sigma):
        return sigma * self.scheduler.config.num_train_timesteps



def make_dataset(images,directory="./new_dataset", image_path=None):
    if not isinstance(images, list):
        images = [images]
        
    if len(images) < 3:
        raise ValueError("At least three images are required to compare index 0 and 2.")
    
    # Saving images
    pil_img = Image.fromarray(images[2].astype(np.uint8))
    
    
    file_path = f"{directory}/new_{image_path}"
    pil_img.save(file_path)
    print(f"Image saved as {file_path}") 
    
    
    percept = lpips.LPIPS(net='vgg').cuda()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) 
    
    
    # Comparing images at index 0 and 2
    image0 = images[0].astype(np.uint8)
    image2 = images[2].astype(np.uint8)
    
    imageA_t = transform(image0).unsqueeze(0).cuda()
    imageB_t = transform(image2).unsqueeze(0).cuda()
    
    
    
    
    
    psnr_value = psnr(image0, image2)
    ssim_value, _ = ssim(image0, image2, full=True, channel_axis=2,win_size=7)
    lpips_value = percept(imageA_t, imageB_t).item()
    
    print(f"🔥 PSNR original vs new: {psnr_value:.2f}")
    print(f"🔥 SSIM original vs new: {ssim_value:.3f}")  
    print(f"🔥 LPIPS original vs new: {lpips_value:.3f}") 
    
    
    # print(f"🔥 LPIPS took a long time so I excluded it. Check it out later in results.txt! ") 


def diffusion_step(model, controller, latents, context, context_p, step, t, guidance_scale, low_resource=False):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        
        
        
        int_t = t
        t = t.expand(latents_input.shape[0])
        t=t.to(torch.float32).to(model.device)
        latents_input=latents_input.to(torch.float32)
        context=context.to(torch.float32)
        context_p=context_p.to(torch.float32)
        joint_attention_kwargs = None
        
        
        noise_pred = model.transformer(
                    hidden_states=latents_input,
                    timestep=t,
                    encoder_hidden_states=context,
                    pooled_projections=context_p,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=False,
                )[0]
        
        
        
        # noise_pred = model.unet(latents_input, 
        #                              t, 
        #                              encoder_hidden_states=context,
        #                              added_cond_kwargs=added_cond_kwargs,
        #                              )["sample"]
        
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    t = t.to(noise_pred.device)
    latents = latents.to(noise_pred.device)
    
    
    latents = prev_step_local(model, noise_pred, int_t, latents, step)
    # latents = model.scheduler.step(noise_pred, t, latents)[0]
    latents = controller.step_callback(latents)
    return latents


def latent2image(vae, latents):
    
    latents = (1 / vae.config.scaling_factor * latents.detach()) + vae.config.shift_factor
    # latents = (1 / 1.5305 * latents.detach()) + 0.0609
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).round().astype(np.uint8)
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        print("init_latent-- latent is None:")
        latent = torch.randn(
            (1, model.transformer.config.in_channels, height // 8, width // 8),
            generator=generator)
        
    if latent.size()[-1] == 128:
        latents = latent.expand(batch_size,  model.transformer.config.in_channels, height // 8, width // 8).to(model.device)
    elif latent.size()[-1] == 32:
        latents = latent.expand(batch_size,  model.transformer.config.in_channels, height // 32, width // 32).to(model.device)
        
    else:
        latents = latent.expand(batch_size,  model.transformer.config.in_channels, height // 16, width // 16).to(model.device)
    
    # latents = latents * model.scheduler.init_noise_sigma
    
    
    return latent, latents





def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None,temb=None):
            
            is_cross = encoder_hidden_states is not None
            residual = hidden_states
            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)
            input_ndim = hidden_states.ndim
            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
            query = self.to_q(hidden_states)
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)
            query = self.head_to_batch_dim(query)
            key = self.head_to_batch_dim(key)
            value = self.head_to_batch_dim(value)
            attention_probs = self.get_attention_scores(query, key, attention_mask)
            if hasattr(self, "store_attn_map"):
                self.attn_map = attention_probs
                
                
            attn_base_chunks = attention_probs.chunk(2, dim=-1)[0]
    
        
            controller(attn_base_chunks, is_cross, place_in_unet)
            hidden_states = torch.bmm(attention_probs, value)
            
            hidden_states = self.batch_to_head_dim(hidden_states)
            # linear proj
            hidden_states = self.to_out[0](hidden_states)
            # dropout
            hidden_states = self.to_out[1](hidden_states)
            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
            if self.residual_connection:
                hidden_states = hidden_states + residual
            hidden_states = hidden_states / self.rescale_output_factor
            # to_out(hidden_states)
            return hidden_states, encoder_hidden_states
        return forward


    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        print("🌊 Dummy Controller Declaration because there is no Controller")
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        
        if net_.__class__.__name__ == 'Attention':
            
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.transformer.named_children()
    for net in sub_nets:
        if "transformer" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count

    
def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def update_alpha_time_word(alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int,
                           word_inds: Optional[torch.Tensor]=None):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(prompts, num_steps,
                                   cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
                                   tokenizer, max_num_words=77):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    
    num_steps = 49
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"],
                                                  i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
             inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
             for i, ind in enumerate(inds):
                 if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
    return alpha_time_words




