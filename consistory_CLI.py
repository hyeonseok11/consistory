# Copyright (C) 2024 NVIDIA Corporation.  All rights reserved.
#
# This work is licensed under the LICENSE file
# located at the root directory.

import os
import argparse
import yaml
from consistory_run import load_pipeline, run_batch_generation, run_anchor_generation, run_extra_generation

def print_args(args):
    print("\n=== Configuration ===")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("===================\n")

def run_batch(gpu, seed=40, mask_dropout=0.5, same_latent=False,
              style="A photo of ", subject="a cute dog", concept_token=['dog'],
              settings=["sitting in the beach", "standing in the snow"],
              out_dir=None, clip_threshold=0.28):
    
    story_pipeline = load_pipeline(gpu)
    prompts = [f'{style}{subject} {setting}' for setting in settings]

    # Set default output directory if not specified
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), 'dataset')
    elif not os.path.isabs(out_dir):
        out_dir = os.path.join(os.path.dirname(__file__), 'dataset', out_dir)
    os.makedirs(out_dir, exist_ok=True)

    images, image_all, clip_scores = run_batch_generation(
        story_pipeline, prompts, concept_token, seed, 
        mask_dropout=mask_dropout, same_latent=same_latent,
        clip_threshold=clip_threshold
    )

    if images:
        for i, (image, score) in enumerate(zip(images, clip_scores)):
            image_path = f'{out_dir}/image_{i}.png'
            image.save(image_path)
            print(f"Saved image {i} with CLIP score: {score:.4f} at {image_path}")

    return images, image_all

def run_cached_anchors(gpu, seed=40, mask_dropout=0.5, same_latent=False,
                      style="A photo of ", subject="a cute dog", concept_token=['dog'],
                      settings=["sitting in the beach", "standing in the snow"],
                      cache_cpu_offloading=False, out_dir=None, clip_threshold=0.28):
    
    story_pipeline = load_pipeline(gpu)
    prompts = [f'{style}{subject} {setting}' for setting in settings]
    anchor_prompts = prompts[:2]
    extra_prompts = prompts[2:]

    # Set default output directory if not specified
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), 'dataset')
    elif not os.path.isabs(out_dir):
        out_dir = os.path.join(os.path.dirname(__file__), 'dataset', out_dir)
    os.makedirs(out_dir, exist_ok=True)

    anchor_out_images, anchor_image_all, anchor_scores, anchor_cache_first_stage, anchor_cache_second_stage = run_anchor_generation(
        story_pipeline, anchor_prompts, concept_token, 
        seed=seed, mask_dropout=mask_dropout, same_latent=same_latent,
        cache_cpu_offloading=cache_cpu_offloading,
        clip_threshold=clip_threshold
    )
    
    if anchor_out_images:
        for i, (image, score) in enumerate(zip(anchor_out_images, anchor_scores)):
            image_path = f'{out_dir}/anchor_image_{i}.png'
            image.save(image_path)
            print(f'Saved anchor image {i} with CLIP score: {score:.4f} at {image_path}')
    
    if len(extra_prompts) > 0:
        for i, extra_prompt in enumerate(extra_prompts):
            extra_out_images, extra_image_all, extra_scores = run_extra_generation(
                story_pipeline, [extra_prompt], concept_token, 
                anchor_cache_first_stage, anchor_cache_second_stage, 
                seed=seed, mask_dropout=mask_dropout, same_latent=same_latent,
                cache_cpu_offloading=cache_cpu_offloading,
                clip_threshold=clip_threshold
            )
            
            if extra_out_images:
                for j, (image, score) in enumerate(zip(extra_out_images, extra_scores)):
                    image_path = f'{out_dir}/extra_image_{i}_{j}.png'
                    image.save(image_path)
                    print(f'Saved extra image {i}_{j} with CLIP score: {score:.4f} at {image_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to YAML config file in config directory')
    parser.add_argument('--run_type', default="batch", type=str, required=False) # batch, cached
    parser.add_argument('--gpu', default=0, type=int, required=False)
    parser.add_argument('--seed', default=40, type=int, required=False)
    parser.add_argument('--mask_dropout', default=0.5, type=float, required=False)
    parser.add_argument('--same_latent', default=False, type=bool, required=False)
    parser.add_argument('--style', default="A photo of ", type=str, required=False)
    parser.add_argument('--subject', default="a cute dog", type=str, required=False)
    parser.add_argument('--concept_token', default=["dog"], 
                        type=str, nargs='*', required=False)
    parser.add_argument('--settings', default=["sitting in the beach", "standing in the snow"], 
                        type=str, nargs='*', required=False)
    parser.add_argument('--cache_cpu_offloading', default=False, type=bool, required=False)
    parser.add_argument('--out_dir', default=None, type=str, required=False)
    parser.add_argument('--clip_threshold', default=0.28, type=float, required=False,
                      help='Minimum CLIP score threshold for saving images')

    args = parser.parse_args()

    # Load config from YAML if provided
    if args.config:
        config_path = os.path.join(os.path.dirname(__file__), 'config', args.config)
        print(f"\nLoading configuration from {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            # Update args with config values, CLI arguments take precedence
            args_dict = vars(args)
            for key, value in config.items():
                if key in args_dict and args_dict[key] == parser.get_default(key):  # Only update if arg is at default value
                    setattr(args, key, value)
        
    # Print final configuration
    print_args(args)

    if args.run_type == "batch":
        run_batch(args.gpu, args.seed, args.mask_dropout, args.same_latent, args.style, 
                  args.subject, args.concept_token, args.settings, args.out_dir,
                  args.clip_threshold)
    elif args.run_type == "cached":
        run_cached_anchors(args.gpu, args.seed, args.mask_dropout, args.same_latent, args.style, 
                  args.subject, args.concept_token, args.settings, args.cache_cpu_offloading, args.out_dir,
                  args.clip_threshold)
    else:
        print("Invalid run type")