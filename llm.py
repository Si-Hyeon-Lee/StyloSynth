import pathlib
import textwrap
import google.generativeai as genai

# [1] 빈칸을 작성하시오.
# API 키
GOOGLE_API_KEY = '' # 지워
genai.configure(api_key=GOOGLE_API_KEY)

# 모델 초기화
def LLM(text):
  
  model = genai.GenerativeModel("gemini-1.5-flash-002")
  while True:
    user_input = text
    if user_input=="q":
      break
    else:
      instruction = "You are a designer who changes the top of a human cloth. You are given simple instructions for changing the top as a prompt, and you describe the instructions as a costume designer would describe a top. Now here is simple instruction"
      prompt = user_input
      full_prompt = f"{instruction}\n{prompt}"
      response = model.generate_content(full_prompt)

      return response
