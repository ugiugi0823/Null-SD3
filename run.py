import gc, pickle, torch, argparse, os, sys
import ptp_utils, seq_aligner


from diffusers import DiffusionPipeline, DDIMScheduler, StableDiffusion3Pipeline
from null import NullInversion
from local import AttentionStore, show_cross_attention, run_and_display, make_controller

def get_pickle_path(filename, bigger):
        return f'./pickle/{filename}_1024.pkl' if bigger else f'./pickle/{filename}.pkl'


def save_to_pickle(data, filename, bigger):
    with open(get_pickle_path(filename, bigger), 'wb') as f:
        pickle.dump(data, f)

def load_from_pickle(filename, bigger):
    with open(get_pickle_path(filename, bigger), 'rb') as f:
        return pickle.load(f)


def main(args):
    # learning_rate = args.learning_rate
    # optimizer = args.optimizer

    
    

    

    prompt = args.prompt
    neg_prompt = args.neg_prompt
    image_path = args.image_path

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    DISN = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False), torch_dtype=torch.float32).to(device)

    null_inversion = NullInversion(DISN)
    (image_gt, image_enc), x_t, uncond_embeddings, uncond_embeddings_p = null_inversion.invert(image_path, prompt, num_inner_steps=28, early_stop_epsilon=1e-5, verbose=True, do_1024=args.bigger, config=args)
    
    # x_t = load_from_pickle('x_t_p', args.bigger)
    # uncond_embeddings = load_from_pickle('uncond_embeddings_p', args.bigger)
    # uncond_embeddings_p = load_from_pickle('uncond_embeddings_p_p', args.bigger)

    torch.cuda.empty_cache()
    gc.collect()

    prompts = [prompt, prompt]
    controller = AttentionStore()
    neg_prompts = [neg_prompt, neg_prompt]

    image_inv, x_t = run_and_display(DISN, neg_prompts, prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings, uncond_embeddings_p=uncond_embeddings_p, verbose=False)
    ptp_utils.view_images([image_gt, image_enc, image_inv[0]])
    ptp_utils.save_individual_images([image_gt, image_enc, image_inv[0]])
    # show_cross_attention(DISN, prompts, controller, 32, ["down"])
    
    
    
    
    save_to_pickle(x_t, 'x_t_p', args.bigger)
    save_to_pickle(uncond_embeddings, 'uncond_embeddings_p', args.bigger)
    save_to_pickle(uncond_embeddings_p, 'uncond_embeddings_p_p', args.bigger)

    

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--image_path", type=str, default="./img/[0001]TopBF0.png", help="Image Path")
    p.add_argument("--prompt", type=str, default="photo of a crack defect image", help="Positive Prompt")
    p.add_argument("--neg_prompt", type=str, default="", help="Negative Prompt")
    p.add_argument("--bigger", action='store_true', help="If you want to create an image 1024")
    p.add_argument("--learning_rate", type=float, required=True, help="Learning rate for the optimizer")
    p.add_argument("--optimizer", type=str, required=True, help="Optimizer to use (sgd, adam, rmsprop 등)")

    args = p.parse_args()
    main(args)
