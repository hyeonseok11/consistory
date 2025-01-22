# Copyright (C) 2024 NVIDIA Corporation.  All rights reserved.
#
# This work is licensed under the LICENSE file
# located at the root directory.
# python consistory_CLI.py --config config.yaml


import os
import argparse
import yaml
from consistory_run import load_pipeline, run_batch_generation, run_anchor_generation, run_extra_generation

def print_args(args, config=None):
    print("\n=== Configuration ===")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    if config and 'settings_groups' in config:
        print("\n=== Settings Groups ===")
        print(f"Output Directory: {config.get('out_dir', 'Not specified')}")
        for group_name, group_data in config['settings_groups'].items():
            print(f"\n{group_name}:")
            print("  settings:")
            for setting in group_data['settings']:
                print(f"    - {setting}")
    print("===================\n")

def get_validation_suffix(validation_result):
    failed_checks = []
    if not validation_result['passed_clip']:
        failed_checks.append('clip')
    if not validation_result['passed_gpt4v']:
        failed_checks.append('gpt4v')
    if not validation_result['passed_attractiveness']:
        failed_checks.append('attr')
    return f"_failed_{'-'.join(failed_checks)}" if failed_checks else ""

def save_validated_image(image, validation_result, output_dir, group_idx, seed, image_idx, clip_score):
    validation_suffix = get_validation_suffix(validation_result)
    image_path = f'{output_dir}/{group_idx}_seed{seed}_image_{image_idx}{validation_suffix}.png'
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    image.save(image_path)
    print(f"Saved image {image_idx} with CLIP score: {clip_score:.4f} at {image_path}")

def run_batch(gpu, seed=40, mask_dropout=0.5, same_latent=False,
              style="A photo of ", subject="a cute dog", concept_token=['dog'],
              settings=["sitting in the beach", "standing in the snow"],
              out_dir=None, clip_threshold=0.28, settings_groups=None):
    
    story_pipeline = load_pipeline(gpu)

    if settings_groups:
        if out_dir is None:
            out_dir = os.path.join(os.path.dirname(__file__), 'dataset')
        elif not os.path.isabs(out_dir):
            out_dir = os.path.join(os.path.dirname(__file__), out_dir)
        os.makedirs(out_dir, exist_ok=True)

        for group_name, group_data in settings_groups.items():
            print(f"\nProcessing group: {group_name}")
            group_settings = group_data['settings']
            
            prompts = [f'{style}{subject} {setting}' for setting in group_settings]
            
            images, image_all, clip_scores, validation_results = run_batch_generation(
                story_pipeline, prompts, concept_token, seed, 
                mask_dropout=mask_dropout, same_latent=same_latent,
                clip_threshold=clip_threshold,
            )

            if images:
                for i, (image, score, validation) in enumerate(zip(images, clip_scores, validation_results)):
                    save_validated_image(
                        image=image,
                        validation_result=validation,
                        output_dir=out_dir,
                        group_idx=group_name,
                        seed=validation['seed'],
                        image_idx=i,
                        clip_score=score
                    )
    else:
        # Set default output directory if not specified
        if out_dir is None:
            out_dir = os.path.join(os.path.dirname(__file__), 'dataset')
        elif not os.path.isabs(out_dir):
            out_dir = os.path.join(os.path.dirname(__file__), out_dir)
        os.makedirs(out_dir, exist_ok=True)

        prompts = [f'{style}{subject} {setting}' for setting in settings]
        
        images, image_all, clip_scores, validation_results = run_batch_generation(
            story_pipeline, prompts, concept_token, seed, 
            mask_dropout=mask_dropout, same_latent=same_latent,
            clip_threshold=clip_threshold
        )

        if images:
            for i, (image, score, validation) in enumerate(zip(images, clip_scores, validation_results)):
                save_validated_image(
                    image=image,
                    validation_result=validation,
                    output_dir=out_dir,
                    group_idx=None,
                    seed=validation['seed'],
                    image_idx=i,
                    clip_score=score
                )

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
        out_dir = os.path.join(os.path.dirname(__file__), out_dir)
    os.makedirs(out_dir, exist_ok=True)

    anchor_out_images, anchor_image_all, anchor_scores, anchor_validation_results, anchor_cache_first_stage, anchor_cache_second_stage = run_anchor_generation(
        story_pipeline, anchor_prompts, concept_token, 
        seed=seed, mask_dropout=mask_dropout, same_latent=same_latent,
        cache_cpu_offloading=cache_cpu_offloading,
        clip_threshold=clip_threshold
    )
    
    if anchor_out_images:
        for i, (image, score, validation) in enumerate(zip(anchor_out_images, anchor_scores, anchor_validation_results)):
            validation_suffix = get_validation_suffix(validation)
            image_path = f'{out_dir}/anchor_seed{validation["seed"]}_image_{i}{validation_suffix}.png'
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            image.save(image_path)
            print(f'Saved anchor image {i} with CLIP score: {score:.4f} at {image_path}')
    
    if len(extra_prompts) > 0:
        for i, extra_prompt in enumerate(extra_prompts):
            extra_out_images, extra_image_all, extra_scores, extra_validation_results = run_extra_generation(
                story_pipeline, [extra_prompt], concept_token, 
                anchor_cache_first_stage, anchor_cache_second_stage, 
                seed=seed, mask_dropout=mask_dropout, same_latent=same_latent,
                cache_cpu_offloading=cache_cpu_offloading,
                clip_threshold=clip_threshold
            )
            
            if extra_out_images:
                for j, (image, score, validation) in enumerate(zip(extra_out_images, extra_scores, extra_validation_results)):
                    validation_suffix = get_validation_suffix(validation)
                    image_path = f'{out_dir}/extra_seed{validation["seed"]}_image_{i}_{j}{validation_suffix}.png'
                    os.makedirs(os.path.dirname(image_path), exist_ok=True)
                    image.save(image_path)
                    print(f'Saved extra image {i}_{j} with CLIP score: {score:.4f} at {image_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', type=str, help='Category of condition to use')
    parser.add_argument('--config', type=str, help='Path to YAML config file in config directory')
    parser.add_argument('--concept_override', type=str, help='Override the concept token in config (e.g., "black man" instead of "white man")')
    parser.add_argument('--run_type', default="batch", type=str, required=False) # batch, cached
    parser.add_argument('--gpu', default=0, type=int, required=False)
    parser.add_argument('--seed', default=40, type=int, required=False)
    parser.add_argument('--mask_dropout', default=0.5, type=float, required=False)
    parser.add_argument('--same_latent', default=False, type=bool, required=False)
    parser.add_argument('--style', type=str, required=False)
    parser.add_argument('--subject', type=str, required=False)
    parser.add_argument('--concept_token', type=str, nargs='*', required=False)
    parser.add_argument('--settings', type=str, nargs='*', required=False)
    parser.add_argument('--cache_cpu_offloading', default=False, type=bool, required=False)
    parser.add_argument('--out_dir', default=None, type=str, required=False)
    parser.add_argument('--clip_threshold', default=0.28, type=float, required=False,
                      help='Minimum CLIP score threshold for saving images')

    args = parser.parse_args()

    # Load config from YAML if provided
    if args.config:
        if not args.condition:
            raise ValueError("--condition argument is required when using a config file")
        config_path = os.path.join(os.path.dirname(__file__), 'config', f'condition_{args.condition}', args.config)
        print(f"\nLoading configuration from {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
            # Apply concept override if specified
            if args.concept_override:
                if 'concept_token' in config and isinstance(config['concept_token'], list) and len(config['concept_token']) > 0:
                    original_concept = config['concept_token'][0]
                    config['concept_token'][0] = args.concept_override
                    if 'subject' in config:
                        config['subject'] = config['subject'].replace(original_concept, args.concept_override)
                    print(f"\nOverriding concept: '{original_concept}' -> '{args.concept_override}'")
                    
                    # Set output directory based on concept_override and original out_dir
                    concept_dir = args.concept_override.replace(" ", "_")
                    if 'out_dir' in config:
                        config['out_dir'] = os.path.join(concept_dir, config['out_dir'])
                    else:
                        config['out_dir'] = concept_dir
                    config['out_dir'] = os.path.join('dataset', config['out_dir'])
                    print(f"Output directory set to: {config['out_dir']}")
            
            # Update args with config values, CLI arguments take precedence
            args_dict = vars(args)
            settings_groups = config.pop('settings_groups', None)  # Extract settings_groups before updating args
            print(settings_groups)
            for key, value in config.items():
                if key in args_dict and args_dict[key] == parser.get_default(key):  # Only update if arg is at default value
                    setattr(args, key, value)
        
    # Print final configuration
    print_args(args, config)

    if args.run_type == "batch":
        run_batch(args.gpu, args.seed, args.mask_dropout, args.same_latent, args.style, 
                  args.subject, args.concept_token, args.settings, args.out_dir,
                  args.clip_threshold, settings_groups=settings_groups)
    elif args.run_type == "cached":
        run_cached_anchors(args.gpu, args.seed, args.mask_dropout, args.same_latent, args.style, 
                  args.subject, args.concept_token, args.settings, args.cache_cpu_offloading, args.out_dir,
                  args.clip_threshold)
    else:
        print("Invalid run type")