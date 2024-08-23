import gc, pickle, torch, argparse, os, sys
import ptp_utils, seq_aligner
import wandb

from diffusers import DiffusionPipeline, DDIMScheduler, StableDiffusion3Pipeline, EDMEulerScheduler
from null import NullInversion
from local import AttentionStore, show_cross_attention, run_and_display, make_controller

NUM_DDIM_STEPS = 28
NUM_INNER_STEPS = 50
GUIDANCE_SCALE = 7.5

def main(args):
    wandb.init(project="null-sd3")  # 프로젝트 이름 설정
    config = wandb.config
    config.learning_rate = args.learning_rate
    config.optimizer = args.optimizer

    prompt = args.prompt
    neg_prompt = args.neg_prompt
    image_path = args.image_path

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    DISN = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", scheduler=scheduler, torch_dtype=torch.float32).to(device)
    # DISN.scheduler = EDMEulerScheduler.from_config(DISN.scheduler.config)
    DISN.scheduler.set_timesteps(NUM_DDIM_STEPS)
  
    null_inversion = NullInversion(DISN, NUM_DDIM_STEPS, GUIDANCE_SCALE)
    (image_gt, image_enc), x_t, uncond_embeddings, uncond_embeddings_p = null_inversion.invert(image_path, prompt, num_inner_steps=NUM_INNER_STEPS, early_stop_epsilon=1e-5, verbose=True, do_1024=args.bigger, config=config)

    torch.cuda.empty_cache()
    gc.collect()

    prompt = "defect with lots of crack"
    prompts = [args.prompt, prompt]
    controller = AttentionStore()
    neg_prompts = [neg_prompt, neg_prompt]

    image_inv, x_t = run_and_display(DISN, neg_prompts, prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings, uncond_embeddings_p=uncond_embeddings_p, verbose=False, steps=NUM_DDIM_STEPS)
    ptp_utils.view_images([image_gt, image_enc, image_inv[0]])
    ptp_utils.save_individual_images([image_gt, image_enc, image_inv[0]])
    show_cross_attention(DISN, prompts, controller, 32, ["down"])

    

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--image_path", type=str, default="./img/[0001]TopBF0.png", help="Image Path")
    p.add_argument("--prompt", type=str, default="photo of a crack defect image", help="Positive Prompt")
    p.add_argument("--neg_prompt", type=str, default="", help="Negative Prompt")
    p.add_argument("--bigger", default=False, help="If you want to create an image 1024")
    p.add_argument("--learning_rate", type=float, required=True, help="Learning rate for the optimizer")
    p.add_argument("--optimizer", type=str, required=True, help="Optimizer to use (sgd, adam, rmsprop 등)")

    args = p.parse_args()
    main(args)
