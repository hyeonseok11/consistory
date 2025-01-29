import os
from openai import OpenAI
import base64
from PIL import Image
import io
import json

def encode_image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def check_criteria(result):
    criteria = {
        'facial_fat_deposit': ['low', 'moderate'],
        'skin_rhytids': ['no visible lines', 'fine lines, but no wrinkles'],
        'eye_canthal_tilt': ['straight', 'upturned'],
        'eye_width_height_ratio': ['small', 'average'],
        'nose_width': ['small', 'average'],
        'mouth_length': ['small', 'average'],
        'overal rating': ['superior']
    }
    
    for key, valid_values in criteria.items():
        if result[key].lower() not in valid_values:
            print(f"Criteria not met: {key} - {result[key]}")
            return False
    print("All criteria met")
    return True

def attractiveness_check(image):
    prompt = '''Evaluate the image.

### Facial Fat Deposit:
 very low / low / moderate / high / very high

### Skin Rhytids (wrinkles/lines):
no visible lines / fine lines, but no wrinkles / early wrinkles / deep wrinkles

### Eye Canthal Tilt:
downturned / straight / upturned

### Eye Width to Height Ratio:
smallest / small / average / large / largest

### Nose Width:
smallest / small / average / large / largest

### Mouth Length:
smallest / small / average / large / largest

### Overal Rating:
below standard / moderate / superior

**Output Format (JSON):**  
You must return your result strictly in the following JSON structure:
{
"facial_fat_deposit": "",
"skin_rhytids": "",
"eye_canthal_tilt": "",
"eye_width_height_ratio": "",
"nose_width": "",
"mouth_length": "",
"overal rating": ""
}
'''
    
    client = OpenAI()
    
    # Encode image to base64
    base64_image = encode_image_to_base64(image)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {
                    "role":"system",
                    "content": "I love to analyze face."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                },
                {
                    "role": "system",
                    "content": "Got it."
                },
                {
                    "role":"user",
                    "content": prompt
                }
            ],
            max_tokens=300
        )
        
        content = response.choices[0].message.content.strip().lower()
        
        try:
            # JSON 문자열에서 실제 JSON 부분만 추출
            json_str = content
            if '```json' in content:
                json_str = content.split('```json')[1].split('```')[0]
            
            result = json.loads(json_str.strip())
            return check_criteria(result)
            
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 에러: {str(e)}")
            print(f"받은 응답: {content}")
            return False
    
    except Exception as e:
        print(f"Error in GPT-4V check: {str(e)}")
        return False  # 에러 발생 시 불합격 처리
