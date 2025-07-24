from dotenv import load_dotenv
from openai import OpenAI

import json
import os

class OpenAIClient:
    def __init__(self):
        load_dotenv() # .env 파일 로드
        self.api_key = os.getenv("OPENAI_API_KEY") # 키 읽어오기
        self.client = OpenAI(api_key=self.api_key) # 클라이언트 생성
    
    def __getattr__(self, name):
        return getattr(self.client, name) # self.client가 가지고 있는 속성이나 메서드를 직접 사용할 수 있게 위임
    
    def create_response(self, model = 'gpt-4.1', system_content = '', user_content = ''):
        response = self.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_content
                },
                {"role": "user", "content": user_content}
            ]
        )
        return response.choices[0].message.content.strip()

def dump_json(json_dir = './json', filename = '', json_data = {}):
    # JSON 파일 저장 경로 구성
    os.makedirs(f"./{json_dir}", exist_ok=True)  # 디렉토리 없으면 생성
    json_path = f"./{json_dir}/{filename}.json"

    # JSON 저장
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ JSON 파일이 저장되었습니다: {json_path}")

def read_json(json_dir = './json', filename = ''):
    json_path = f"./{json_dir}/{filename}.json"

    # JSON 읽기
    with open(json_path, "r", encoding = "utf-8") as f:
        data = json.load(f)
    
    print(f"\n✅ JSON 파일을 읽어왔습니다: {json_path}")
    return data