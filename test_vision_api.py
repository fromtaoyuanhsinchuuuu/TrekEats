from openai import OpenAI
import os
from dotenv import load_dotenv
import base64
from pathlib import Path

def encode_image_to_base64(image_path):
    """將圖片轉換為 base64 編碼"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_text_from_image():
    """使用 OpenAI Vision 模型從圖片中提取文字"""
    
    # 載入環境變量
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    # 設置圖片路徑
    image_path = "menu.webp"
    
    # 確認圖片存在
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    # 將圖片轉換為 base64
    base64_image = encode_image_to_base64(image_path)
    
    client = OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # 使用支持圖片的模型
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please extract all the text from this menu image. Format it nicely and include all menu items and prices."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/webp;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        # 打印提取的文字
        print("\nExtracted text from the menu:")
        print("-" * 50)
        print(response.choices[0].message.content)
        print("-" * 50)
        
        # 保存提取的文字到文件
        output_file = "menu_text_gpt4vision.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(response.choices[0].message.content)
        print(f"\nText has been saved to: {output_file}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    extract_text_from_image()
