import unittest
import os
import cv2
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ocr import MenuOCR

class TestMenuOCR(unittest.TestCase):
    def setUp(self):
        """測試前的設置"""
        self.ocr = MenuOCR()
        
        # 使用絕對路徑並打印出來確認
        self.test_image_path = os.path.abspath(os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "korean_tofu_house_meu.webp"
        ))
        print(f"\nFull image path: {self.test_image_path}")
        
        # 確認文件存在
        if not os.path.exists(self.test_image_path):
            raise FileNotFoundError(f"Image not found at: {self.test_image_path}")

    def test_extract_text(self):
        """測試文字提取和保存"""
        print(f"\nTesting image path: {self.test_image_path}")
        result = self.ocr.extract_text(self.test_image_path)
        
        # 打印完整的結果
        print(f"OCR Result: {result}")
        
        # 檢查結果
        self.assertEqual(result['status'], 'success')
        self.assertIsNotNone(result['text'])
        
        # 確認文件是否已創建
        self.assertTrue(os.path.exists('menu_text.txt'))
        
        # 讀取保存的文件內容
        with open('menu_text.txt', 'r', encoding='utf-8') as f:
            saved_text = f.read()
        
        # 確認文件內容與返回的文字相同
        self.assertEqual(saved_text, result['text'])

    def test_extract_text_with_boxes(self):
        """測試帶位置信息的文字提取"""
        result = self.ocr.extract_text_with_boxes(self.test_image_path)
        
        # 檢查結果
        self.assertEqual(result['status'], 'success')
        self.assertIsNotNone(result['data'])
        
        # 打印提取的文字和位置
        print("\n=== Extracted text with positions ===")
        for item in result['data']:
            print(f"Text: {item['text']}")
            print(f"Position: (x={item['x']}, y={item['y']})")
            print(f"Confidence: {item['conf']}%")
            print("-" * 50)
        
        # 檢查是否有提取到文字
        self.assertTrue(len(result['data']) > 0, "No text was extracted from the image")

    def test_image_preprocessing(self):
        """測試圖像預處理"""
        # 讀取原始圖像
        image = cv2.imread(self.test_image_path)
        self.assertIsNotNone(image, "Failed to load test image")
        
        # 測試預處理
        processed_image = self.ocr.preprocess_image(image)
        
        # 保存處理後的圖像以供視覺檢查
        output_path = "processed_korean_menu.png"
        cv2.imwrite(output_path, processed_image)
        print(f"\nProcessed image saved to: {output_path}")

if __name__ == '__main__':
    unittest.main() 