import logging

from DeepCache.flops import count_ops_and_params

from vars import DEFAULT_MODEL, DEFAULT_SEED, labels_file

# import torch_pruning as tp

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

import torch
from torchvision.utils import save_image

from DeepCache.sd.pipeline_stable_diffusion import StableDiffusionPipeline as DeepCacheStableDiffusionPipeline, \
    StableDiffusionPipeline


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gen_baseline(prompts, model, seed, directory, start, end):
    prompts = prompts[start:end]
    # Baseline generation
    # baseline_pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16).to("cuda:0")
    #
    # for i in range(0, len(prompts)):
    #     prompt = prompts[i]
    #
    #     set_random_seed(seed)
    #     ori_output = baseline_pipe(prompt, output_type='pt').images
    #
    #     save_image(ori_output[0], f"{directory}/baseline_{i}.png")
    #
    # del baseline_pipe
    # torch.cuda.empty_cache()

    # Deepcache generation
    pipe = DeepCacheStableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16).to("cuda:0")

    for i in range(start, end):
        prompt = prompts[i]

        set_random_seed(seed)
        deepcache_output = pipe(
            prompt,
            cache_interval=10, cache_layer_id=0, cache_block_id=0,
            uniform=False, pow=1.4, center=15,
            output_type='pt', return_dict=True
        ).images
        save_image(deepcache_output[0], f"{directory}/deepcache_{i}.png")


if __name__ == "__main__":
    lines = []

    with open(labels_file) as file:
        lines = [line.rstrip() for line in file]

    gen_baseline(lines, DEFAULT_MODEL, DEFAULT_SEED, "data/deepcache-c10", 0, 5000)
