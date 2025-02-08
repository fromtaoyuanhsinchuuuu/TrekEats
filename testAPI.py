from openai import OpenAI
import os
from dotenv import load_dotenv

def test_api():
    """
    Test SWARM API key functionality
    """
    # 直接使用提供的 API key
    api_key = "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"

    client = OpenAI(api_key=api_key)

    try:
        # 測試 API 調用
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Hello, this is a test message for TrekEats API."}
            ]
        )

        print("API 測試成功!")
        print("回應:", response.choices[0].message.content)
        return True

    except Exception as e:
        print("API 測試失敗!")
        print("錯誤:", str(e))
        return False

if __name__ == "__main__":
    test_api()
