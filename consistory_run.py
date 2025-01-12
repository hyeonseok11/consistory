# Copyright (C) 2024 NVIDIA Corporation.  All rights reserved.
#
# This work is licensed under the LICENSE file
# located at the root directory.

import torch
from diffusers import DDIMScheduler
from consistory_unet_sdxl import ConsistorySDXLUNet2DConditionModel
from consistory_pipeline import ConsistoryExtendAttnSDXLPipeline
from consistory_utils import FeatureInjector, AnchorCache
from utils.general_utils import *
import gc
import os
from clip_score import get_clip_score
from gpt4o_check import check_image_with_gpt4o
import io
from PIL import Image

from utils.ptp_utils import view_images
from dotenv import load_dotenv
load_dotenv()

negative_prompts = """
out of focus, blurry, lowres, poorly drawn, jpeg artifacts, watermark, 
unnatural body proportions, awkward pose, extra limbs, fused limbs, distorted body, unrealistic shadows, harsh shadows, 
wrong perspective, motion blur, camera shake, tilted horizon, oversaturated colors, double head, double face, 
symmetrical issues, disfigured face, incomplete anatomy, large text, repeated text, overexposed, underexposed, 
cartoonish, missing limbs, extreme muscle definition, extra clothing, random floating objects, glitchy, pixelated, 
chromatic aberration, twised hands, mutated hands, mutated feet, mutated legs, mutated arms, mutated body,
"""

LATENT_RESOLUTIONS = [32, 64]

def load_pipeline(gpu_id=0):
    float_type = torch.float16
    sd_id = "stabilityai/stable-diffusion-xl-base-1.0"
    
    device = torch.device(f'cuda:{gpu_id}') if torch.cuda.is_available() else torch.device('cpu')
    unet = ConsistorySDXLUNet2DConditionModel.from_pretrained(sd_id, subfolder="unet", torch_dtype=float_type)
    scheduler = DDIMScheduler.from_pretrained(sd_id, subfolder="scheduler")

    story_pipeline = ConsistoryExtendAttnSDXLPipeline.from_pretrained(
        sd_id, unet=unet, torch_dtype=float_type, variant="fp16", use_safetensors=True, scheduler=scheduler,
    ).to(device)
    story_pipeline.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
    
    return story_pipeline

def create_anchor_mapping(bsz, anchor_indices=[0]):
    anchor_mapping = torch.eye(bsz, dtype=torch.bool)
    for anchor_idx in anchor_indices:
        anchor_mapping[:, anchor_idx] = True

    return anchor_mapping

def create_token_indices(prompts, batch_size, concept_token, tokenizer):
    if isinstance(concept_token, str):
        concept_token = [concept_token]

    concept_token_id = [tokenizer.encode(x, add_special_tokens=False)[0] for x in concept_token]
    tokens = tokenizer.batch_encode_plus(prompts, padding=True, return_tensors='pt')['input_ids']

    token_indices = torch.full((len(concept_token), batch_size), -1, dtype=torch.int64)
    for i, token_id in enumerate(concept_token_id):
        batch_loc, token_loc = torch.where(tokens == token_id)
        token_indices[i, batch_loc] = token_loc

    return token_indices

def create_latents(story_pipeline, seed, batch_size, same_latent, device, float_type):
    # if seed is int
    if isinstance(seed, int):
        g = torch.Generator('cuda').manual_seed(seed)
        shape = (batch_size, story_pipeline.unet.config.in_channels, 128, 128)
        latents = randn_tensor(shape, generator=g, device=device, dtype=float_type)
    elif isinstance(seed, list):
        shape = (batch_size, story_pipeline.unet.config.in_channels, 128, 128)
        latents = torch.empty(shape, device=device, dtype=float_type)
        for i, seed_i in enumerate(seed):
            g = torch.Generator('cuda').manual_seed(seed_i)
            curr_latent = randn_tensor(shape, generator=g, device=device, dtype=float_type)
            latents[i] = curr_latent[i]

    if same_latent:
        latents = latents[:1].repeat(batch_size, 1, 1, 1)

    return latents, g

# Batch inference
def run_batch_generation(story_pipeline, prompts, concept_token,
                        seed=40, n_steps=50, mask_dropout=0.5,
                        same_latent=False, share_queries=True,
                        perform_sdsa=True, perform_injection=True,
                        downscale_rate=4, n_achors=2,
                        clip_threshold=0.28,
                        max_attempts=10):  
    device = story_pipeline.device
    tokenizer = story_pipeline.tokenizer
    float_type = story_pipeline.dtype
    unet = story_pipeline.unet

    batch_size = len(prompts)
    filtered_images = []
    filtered_indices = []
    remaining_indices = list(range(batch_size))
    clip_scores = []  # Store CLIP scores for each successful image

    attempt = 0
    current_mask_dropout = mask_dropout
    while remaining_indices and attempt < max_attempts:
        current_prompts = [prompts[i] for i in remaining_indices]
        current_batch_size = len(current_prompts)
        
        token_indices = create_token_indices(current_prompts, current_batch_size, concept_token, tokenizer)
        anchor_mappings = create_anchor_mapping(current_batch_size, anchor_indices=list(range(min(n_achors, current_batch_size))))

        default_attention_store_kwargs = {
            'token_indices': token_indices,
            'mask_dropout': current_mask_dropout,
            'extended_mapping': anchor_mappings
        }

        default_extended_attn_kwargs = {'extend_kv_unet_parts': ['up']}
        query_store_kwargs = {'t_range': [0,n_steps//10], 'strength_start': 0.9, 'strength_end': 0.81836735}

        latents, g = create_latents(story_pipeline, seed, current_batch_size, same_latent, device, float_type)

        if perform_sdsa:
            extended_attn_kwargs = {**default_extended_attn_kwargs, 't_range': [(1, n_steps)]}
        else:
            extended_attn_kwargs = {**default_extended_attn_kwargs, 't_range': []}

        out = story_pipeline(prompt=current_prompts, generator=g, latents=latents, 
                            attention_store_kwargs=default_attention_store_kwargs,
                            extended_attn_kwargs=extended_attn_kwargs,
                            share_queries=share_queries,
                            query_store_kwargs=query_store_kwargs,
                            num_inference_steps=n_steps,
                            negative_prompt=negative_prompts)
        
        last_masks = story_pipeline.attention_store.last_mask
        dift_features = unet.latent_store.dift_features['261_0'][current_batch_size:]
        dift_features = torch.stack([gaussian_smooth(x, kernel_size=3, sigma=1) for x in dift_features], dim=0)
        nn_map, nn_distances = cyclic_nn_map(dift_features, last_masks, LATENT_RESOLUTIONS, device)

        torch.cuda.empty_cache()
        gc.collect()

        if perform_injection:
            feature_injector = FeatureInjector(nn_map, nn_distances, last_masks, inject_range_alpha=[(n_steps//10, n_steps//3,0.8)], 
                                            swap_strategy='min', inject_unet_parts=['up', 'down'], dist_thr='dynamic')

            out = story_pipeline(prompt=current_prompts, generator=g, latents=latents, 
                                attention_store_kwargs=default_attention_store_kwargs,
                                extended_attn_kwargs=extended_attn_kwargs,
                                share_queries=share_queries,
                                query_store_kwargs=query_store_kwargs,
                                feature_injector=feature_injector,
                                num_inference_steps=n_steps,
                            negative_prompt=negative_prompts)

            # Check CLIP scores for current batch
            new_remaining_indices = []
            for batch_idx, image in enumerate(out.images):
                original_idx = remaining_indices[batch_idx]
                # Convert PIL image to bytes for CLIP score calculation
                with io.BytesIO() as bio:
                    image.save(bio, format='PNG')
                    image_bytes = bio.getvalue()
                with io.BytesIO(image_bytes) as bio:
                    temp_image = Image.open(bio)
                    clip_score = get_clip_score(temp_image, prompts[original_idx])
                
                print(f"Image {original_idx} CLIP score: {clip_score}")
                
                if clip_score >= clip_threshold and check_image_with_gpt4o(image, prompts[original_idx]):
                    filtered_images.append(image)
                    filtered_indices.append(original_idx)
                    clip_scores.append(clip_score)
                else:
                    new_remaining_indices.append(original_idx)

            remaining_indices = new_remaining_indices
            if current_mask_dropout <= 0.01:  # If we've reached the minimum mask_dropout
                print("Reached minimum mask_dropout. Adding remaining images...")
                for batch_idx, original_idx in enumerate(remaining_indices):
                    if batch_idx < len(out.images):  # Only add if we have an image for this index
                        filtered_images.append(out.images[batch_idx])
                        filtered_indices.append(original_idx)
                        clip_scores.append(current_clip_scores[batch_idx])
                break
            current_mask_dropout = max(0.01, current_mask_dropout - 0.05)
            print(f"Attempt {attempt + 1}: {len(remaining_indices)} images below threshold. Retrying with mask_dropout: {current_mask_dropout:.2f}")
            attempt += 1

            torch.cuda.empty_cache()
            gc.collect()
        else:
            filtered_images = out.images
            filtered_indices = list(range(len(out.images)))
            clip_scores = [0.0] * len(out.images)  # Default scores for non-injection mode
            remaining_indices = []

    if filtered_images:
        # Sort images by original indices
        sorted_pairs = sorted(zip(filtered_indices, filtered_images, clip_scores), key=lambda x: x[0])
        filtered_images = [img for _, img, _ in sorted_pairs]
        clip_scores = [score for _, _, score in sorted_pairs]
        img_all = view_images([np.array(x) for x in filtered_images], display_image=False, downscale_rate=downscale_rate)
    else:
        img_all = None
        print("No images met the CLIP score threshold after maximum attempts.")

    if remaining_indices:
        print(f"Warning: Could not generate satisfactory images for indices {remaining_indices} after {max_attempts} attempts")
    
    return filtered_images, img_all, clip_scores

# Anchors
def run_anchor_generation(story_pipeline, prompts, concept_token,
                        seed=40, n_steps=50, mask_dropout=0.5,
                        same_latent=False, share_queries=True,
                        perform_sdsa=True, perform_injection=True,
                        downscale_rate=4, cache_cpu_offloading=False,
                        clip_threshold=0.28,
                        max_attempts=10):
    device = story_pipeline.device
    tokenizer = story_pipeline.tokenizer
    float_type = story_pipeline.dtype
    unet = story_pipeline.unet

    batch_size = len(prompts)
    filtered_images = []
    filtered_indices = []
    remaining_indices = list(range(batch_size))
    clip_scores = []

    attempt = 0
    current_mask_dropout = mask_dropout
    while remaining_indices and attempt < max_attempts:
        current_prompts = [prompts[i] for i in remaining_indices]
        current_batch_size = len(current_prompts)

        token_indices = create_token_indices(current_prompts, current_batch_size, concept_token, tokenizer)

        default_attention_store_kwargs = {
            'token_indices': token_indices,
            'mask_dropout': current_mask_dropout
        }

        default_extended_attn_kwargs = {'extend_kv_unet_parts': ['up']}
        query_store_kwargs={'t_range': [0,n_steps//10], 'strength_start': 0.9, 'strength_end': 0.81836735}

        latents, g = create_latents(story_pipeline, seed, current_batch_size, same_latent, device, float_type)

        anchor_cache_first_stage = AnchorCache()
        anchor_cache_second_stage = AnchorCache()

        # ------------------ #
        # Extended attention First Run #

        if perform_sdsa:
            extended_attn_kwargs = {**default_extended_attn_kwargs, 't_range': [(1, n_steps)]}
        else:
            extended_attn_kwargs = {**default_extended_attn_kwargs, 't_range': []}

        print(extended_attn_kwargs['t_range'])
        out = story_pipeline(prompt=current_prompts, generator=g, latents=latents, 
                            attention_store_kwargs=default_attention_store_kwargs,
                            extended_attn_kwargs=extended_attn_kwargs,
                            share_queries=share_queries,
                            query_store_kwargs=query_store_kwargs,
                            anchors_cache=anchor_cache_first_stage,
                            num_inference_steps=n_steps,
                            negative_prompt=negative_prompts)
        last_masks = story_pipeline.attention_store.last_mask

        # Calculate CLIP scores for current batch
        current_clip_scores = []
        for idx, (image, prompt) in enumerate(zip(out.images, current_prompts)):
            clip_score = get_clip_score(image, prompt)
            current_clip_scores.append(clip_score)
            if clip_score >= clip_threshold and check_image_with_gpt4o(image, prompt):
                filtered_images.append(image)
                filtered_indices.append(remaining_indices[idx])
                clip_scores.append(clip_score)

        # Update remaining indices based on CLIP scores
        remaining_indices = [remaining_indices[i] for i, score in enumerate(current_clip_scores) if score < clip_threshold or not check_image_with_gpt4o(out.images[i], current_prompts[i])]

        if len(remaining_indices) > 0:
            if current_mask_dropout <= 0.01:  # If we've reached the minimum mask_dropout
                print("Reached minimum mask_dropout. Adding remaining images...")
                for batch_idx, original_idx in enumerate(remaining_indices):
                    if batch_idx < len(out.images):  # Only add if we have an image for this index
                        filtered_images.append(out.images[batch_idx])
                        filtered_indices.append(original_idx)
                        clip_scores.append(current_clip_scores[batch_idx])
                break
            current_mask_dropout = max(0.01, current_mask_dropout - 0.05)
            print(f"Attempt {attempt + 1}: {len(remaining_indices)} images below threshold. Retrying with mask_dropout: {current_mask_dropout:.2f}")
        
        dift_features = unet.latent_store.dift_features['261_0'][current_batch_size:]
        dift_features = torch.stack([gaussian_smooth(x, kernel_size=3, sigma=1) for x in dift_features], dim=0)

        anchor_cache_first_stage.dift_cache = dift_features
        anchor_cache_first_stage.anchors_last_mask = last_masks

        if cache_cpu_offloading:
            anchor_cache_first_stage.to_device(torch.device('cpu'))

        nn_map, nn_distances = cyclic_nn_map(dift_features, last_masks, LATENT_RESOLUTIONS, device)

        torch.cuda.empty_cache()
        gc.collect()

        # ------------------ #
        # Extended attention with nn_map #
        
        if perform_injection:
            feature_injector = FeatureInjector(nn_map, nn_distances, last_masks, inject_range_alpha=[(n_steps//10, n_steps//3,0.8)], 
                                            swap_strategy='min', inject_unet_parts=['up', 'down'], dist_thr='dynamic')

            out = story_pipeline(prompt=current_prompts, generator=g, latents=latents, 
                                attention_store_kwargs=default_attention_store_kwargs,
                                extended_attn_kwargs=extended_attn_kwargs,
                                share_queries=share_queries,
                                query_store_kwargs=query_store_kwargs,
                                feature_injector=feature_injector,
                                anchors_cache=anchor_cache_second_stage,
                                num_inference_steps=n_steps,
                                negative_prompt=negative_prompts)
            img_all = view_images([np.array(x) for x in out.images], display_image=False, downscale_rate=downscale_rate)
            # display_attn_maps(story_pipeline.attention_store.last_mask, out.images)

            if cache_cpu_offloading:
                anchor_cache_second_stage.to_device(torch.device('cpu'))

            torch.cuda.empty_cache()
            gc.collect()

        attempt += 1

    # Sort the filtered images based on original indices
    sorted_results = sorted(zip(filtered_indices, filtered_images, clip_scores))
    sorted_images = [img for _, img, _ in sorted_results]
    sorted_scores = [score for _, _, score in sorted_results]
    
    if len(sorted_images) == 0:
        print("Warning: No images met the CLIP score threshold after all attempts")
        # Use the last generated images as fallback
        sorted_images = out.images
        sorted_scores = current_clip_scores
        
    img_all = view_images([np.array(x) for x in sorted_images], display_image=False, downscale_rate=downscale_rate)
    
    return sorted_images, img_all, sorted_scores, anchor_cache_first_stage, anchor_cache_second_stage

def run_extra_generation(story_pipeline, prompts, concept_token, 
                         anchor_cache_first_stage, anchor_cache_second_stage,
                         seed=40, n_steps=50, mask_dropout=0.5,
                         same_latent=False, share_queries=True,
                         perform_sdsa=True, perform_injection=True,
                         downscale_rate=4, cache_cpu_offloading=False,
                         clip_threshold=0.28,
                         max_attempts=10):
    device = story_pipeline.device
    tokenizer = story_pipeline.tokenizer
    float_type = story_pipeline.dtype
    unet = story_pipeline.unet

    batch_size = len(prompts)
    filtered_images = []
    filtered_indices = []
    clip_scores = []
    remaining_indices = list(range(len(prompts)))
    current_mask_dropout = mask_dropout

    for attempt in range(max_attempts):
        if len(remaining_indices) == 0:
            break

        current_prompts = [prompts[i] for i in remaining_indices]
        current_batch_size = len(current_prompts)
        
        token_indices = create_token_indices(current_prompts, current_batch_size, concept_token, tokenizer)

        default_attention_store_kwargs = {
            'token_indices': token_indices,
            'mask_dropout': current_mask_dropout
        }

        default_extended_attn_kwargs = {'extend_kv_unet_parts': ['up']}
        query_store_kwargs={'t_range': [0,n_steps//10], 'strength_start': 0.9, 'strength_end': 0.81836735}

        extra_batch_size = batch_size + 2
        if isinstance(seed, list):
            seed = [seed[0], seed[0], *seed]

        latents, g = create_latents(story_pipeline, seed, extra_batch_size, same_latent, device, float_type)
        latents = latents[2:]

        anchor_cache_first_stage.set_mode_inject()
        anchor_cache_second_stage.set_mode_inject()

        # ------------------ #
        # Extended attention First Run #

        if cache_cpu_offloading:
            anchor_cache_first_stage.to_device(device)

        if perform_sdsa:
            extended_attn_kwargs = {**default_extended_attn_kwargs, 't_range': [(1, n_steps)]}
        else:
            extended_attn_kwargs = {**default_extended_attn_kwargs, 't_range': []}

        print(extended_attn_kwargs['t_range'])
        out = story_pipeline(prompt=current_prompts, generator=g, latents=latents, 
                            attention_store_kwargs=default_attention_store_kwargs,
                            extended_attn_kwargs=extended_attn_kwargs,
                            share_queries=share_queries,
                            query_store_kwargs=query_store_kwargs,
                            anchors_cache=anchor_cache_first_stage,
                            num_inference_steps=n_steps,
                            negative_prompt=negative_prompts)
        last_masks = story_pipeline.attention_store.last_mask

        dift_features = unet.latent_store.dift_features['261_0'][batch_size:]
        dift_features = torch.stack([gaussian_smooth(x, kernel_size=3, sigma=1) for x in dift_features], dim=0)

        anchor_dift_features = anchor_cache_first_stage.dift_cache
        anchor_last_masks = anchor_cache_first_stage.anchors_last_mask

        nn_map, nn_distances = anchor_nn_map(dift_features, anchor_dift_features, last_masks, anchor_last_masks, LATENT_RESOLUTIONS, device)

        if cache_cpu_offloading:
            anchor_cache_first_stage.to_device(torch.device('cpu'))

        torch.cuda.empty_cache()
        gc.collect()

        # Calculate CLIP scores for current batch
        current_clip_scores = []
        for idx, (image, prompt) in enumerate(zip(out.images, current_prompts)):
            clip_score = get_clip_score(image, prompt)
            current_clip_scores.append(clip_score)
            if clip_score >= clip_threshold and check_image_with_gpt4o(image, prompt):
                filtered_images.append(image)
                filtered_indices.append(remaining_indices[idx])
                clip_scores.append(clip_score)

        # Update remaining indices based on CLIP scores
        remaining_indices = [remaining_indices[i] for i, score in enumerate(current_clip_scores) if score < clip_threshold or not check_image_with_gpt4o(out.images[i], current_prompts[i])]

        if len(remaining_indices) > 0:
            if current_mask_dropout <= 0.01:  # If we've reached the minimum mask_dropout
                print("Reached minimum mask_dropout. Adding remaining images...")
                for batch_idx, original_idx in enumerate(remaining_indices):
                    if batch_idx < len(out.images):  # Only add if we have an image for this index
                        filtered_images.append(out.images[batch_idx])
                        filtered_indices.append(original_idx)
                        clip_scores.append(current_clip_scores[batch_idx])
                break
            current_mask_dropout = max(0.01, current_mask_dropout - 0.05)
            print(f"Attempt {attempt + 1}: {len(remaining_indices)} images below threshold. Retrying with mask_dropout: {current_mask_dropout:.2f}")

        if perform_injection:

            if cache_cpu_offloading:
                anchor_cache_second_stage.to_device(device)

            feature_injector = FeatureInjector(nn_map, nn_distances, last_masks, inject_range_alpha=[(n_steps//10, n_steps//3,0.8)], 
                                            swap_strategy='min', inject_unet_parts=['up', 'down'], dist_thr='dynamic')

            out = story_pipeline(prompt=current_prompts, generator=g, latents=latents, 
                                attention_store_kwargs=default_attention_store_kwargs,
                                extended_attn_kwargs=extended_attn_kwargs,
                                share_queries=share_queries,
                                query_store_kwargs=query_store_kwargs,
                                feature_injector=feature_injector,
                                anchors_cache=anchor_cache_second_stage,
                                num_inference_steps=n_steps,
                                negative_prompt=negative_prompts)
            img_all = view_images([np.array(x) for x in out.images], display_image=False, downscale_rate=downscale_rate)
            # display_attn_maps(story_pipeline.attention_store.last_mask, out.images)

            if cache_cpu_offloading:
                anchor_cache_second_stage.to_device(torch.device('cpu'))

            torch.cuda.empty_cache()
            gc.collect()
        else:
            img_all = view_images([np.array(x) for x in out.images], display_image=False, downscale_rate=downscale_rate)
    
    return filtered_images, img_all, clip_scores