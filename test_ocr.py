import unittest
import os
import cv2
from backend.ocr import MenuOCR

class TestMenuOCR(unittest.TestCase):
    def setUp(self):
        self.ocr = MenuOCR()
        # 創建測試圖片目錄
        self.test_dir = os.path.join(os.path.dirname(__file__), 'test_images')
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
            
    def test_extract_text_basic(self):
        # 創建一個簡單的測試圖片
        test_image_path = os.path.join(self.test_dir, 'test_menu.png')
        # 創建一個白底黑字的圖片
        img = 255 * np.ones((100, 300, 3), dtype=np.uint8)
        # 添加文字
        text = "Coffee $5"
        cv2.putText(img, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imwrite(test_image_path, img)
        
        # 測試文字提取
        result = self.ocr.extract_text(test_image_path)
        
        # 驗證結果
        self.assertEqual(result['status'], 'success')
        self.assertIn('Coffee', result['text'])
        self.assertIn('5', result['text'])
        
    def test_extract_text_with_layout(self):
        test_image_path = os.path.join(self.test_dir, 'test_menu_layout.png')
        # 創建測試圖片
        img = 255 * np.ones((200, 400, 3), dtype=np.uint8)
        cv2.putText(img, "Menu", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, "Coffee $5", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, "Tea $4", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imwrite(test_image_path, img)
        
        # 測試帶佈局的文字提取
        result = self.ocr.extract_text_with_layout(test_image_path)
        
        # 驗證結果
        self.assertEqual(result['status'], 'success')
        self.assertTrue(len(result['data']) > 0)
        
        # 驗證是否包含預期的文字
        extracted_texts = [item['text'] for item in result['data']]
        self.assertTrue(any('Menu' in text for text in extracted_texts))
        self.assertTrue(any('Coffee' in text for text in extracted_texts))
        self.assertTrue(any('Tea' in text for text in extracted_texts))

    def test_error_handling(self):
        # 測試不存在的圖片
        result = self.ocr.extract_text('nonexistent.png')
        self.assertEqual(result['status'], 'error')
        self.assertIsNotNone(result['error'])

if __name__ == '__main__':
    unittest.main()