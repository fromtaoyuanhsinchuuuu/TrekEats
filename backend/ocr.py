import cv2
import pytesseract
import numpy as np
from PIL import Image
import os
from datetime import datetime

class MenuOCR:
    def __init__(self):
        # Tesseract 配置
        # -l eng 指定英文
        # --oem 3 使用默認引擎模式
        # --psm 6 假設統一的文本塊
        self.tesseract_config = r'--oem 3 --psm 6 -l eng+kor'
        
        # 打印 Tesseract 版本信息
        try:
            version = pytesseract.get_tesseract_version()
            print(f"Tesseract version: {version}")
        except Exception as e:
            print(f"Error getting Tesseract version: {e}")
        
    def preprocess_image(self, image):
        """
        預處理圖像以提高 OCR 準確性
        """
        # 轉換為灰度圖
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 降噪
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # 自適應閾值處理
        thresh = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # 塊大小
            2    # C常數
        )
        
        # 膨脹操作，使文字更清晰
        kernel = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        
        return dilated

    def save_extracted_text(self, text, filename='menu_text.txt'):
        """
        保存提取的文字到文件
        
        Args:
            text (str): 提取的文字
            filename (str): 輸出文件名
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"\nExtracted text has been saved to: {filename}")
            return True
        except Exception as e:
            print(f"Error saving text: {str(e)}")
            return False

    def extract_text(self, image_path):
        """從圖像中提取文字並保存"""
        try:
            # 讀取圖像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not read image")

            # 預處理圖像
            processed_image = self.preprocess_image(image)
            
            # 保存處理後的圖像
            cv2.imwrite('processed_image.png', processed_image)
            
            # 執行 OCR
            text = pytesseract.image_to_string(
                processed_image,
                config=self.tesseract_config
            )
            
            # 保存提取的文字
            self.save_extracted_text(text.strip())
            
            # 打印提取的文字
            print("\nExtracted Text:")
            print("-" * 50)
            print(text.strip())
            print("-" * 50)
            
            return {
                'status': 'success',
                'text': text.strip(),
                'error': None
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'text': None,
                'error': str(e)
            }

    def extract_text_with_boxes(self, image_path):
        """
        提取文字並返回每個文字的位置信息
        """
        try:
            image = cv2.imread(image_path)
            processed_image = self.preprocess_image(image)
            
            # 獲取文字和位置信息
            d = pytesseract.image_to_data(
                processed_image, 
                output_type=pytesseract.Output.DICT,
                config=self.tesseract_config
            )
            
            # 整理有意義的文字和位置
            extracted_data = []
            n_boxes = len(d['text'])
            
            for i in range(n_boxes):
                if int(float(d['conf'][i])) > 60:  # 只保留置信度高於 60% 的文字
                    if d['text'][i].strip():  # 只保留非空文字
                        extracted_data.append({
                            'text': d['text'][i],
                            'conf': d['conf'][i],
                            'x': d['left'][i],
                            'y': d['top'][i],
                            'width': d['width'][i],
                            'height': d['height'][i]
                        })
            
            return {
                'status': 'success',
                'data': extracted_data,
                'error': None
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'data': None,
                'error': str(e)
            }
