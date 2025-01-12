import os
from openai import OpenAI
import base64
from PIL import Image
import io


def encode_image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def check_image_with_gpt4o(image, prompt):
    """
    Check if the generated image has any visual artifacts or inconsistencies using GPT-4V
    
    Args:
        image (PIL.Image): The image to check
        prompt (str): The original prompt used to generate the image
        
    Returns:
        bool: True if the image passes the check, False otherwise
    """
    
    check_prompt = f"""
    This image was generated based on the prompt: '{prompt}'. 
    Please analyze if there are any visual artifacts, anatomical inconsistencies, or unnatural elements in the image.
    Focus on checking: 
    - out of focus, blurry, lowres, poorly drawn, jpeg artifacts, watermark, unnatural body proportions, awkward pose, extra limbs, fused limbs, distorted body, 
    unrealistic shadows, harsh shadows, wrong perspective, motion blur, camera shake, tilted horizon, oversaturated colors, double head, double face, 
    symmetrical issues, disfigured face, incomplete anatomy, large text, repeated text, overexposed, underexposed, cartoonish, missing limbs, extreme muscle definition, 
    extra clothing, random floating objects, glitchy, pixelated, chromatic aberration, twised hands, mutated hands, mutated feet, mutated legs, mutated arms, mutated body
    
    Respond with 'PASS' if the image looks natural and consistent, or 'FAIL' with a brief explanation if you find any issues.
    """
    
    client = OpenAI()
    
    # Encode image to base64
    base64_image = encode_image_to_base64(image)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": check_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        
        result = response.choices[0].message.content.strip().upper()
        print(result)
        return result.startswith("PASS")
    
    except Exception as e:
        print(f"Error in GPT-4V check: {str(e)}")
        # In case of API error, we'll pass the image to avoid blocking the pipeline
        return True
