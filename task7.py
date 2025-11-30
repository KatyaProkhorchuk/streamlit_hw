import requests
from dotenv import load_dotenv
import os
load_dotenv()

identificator = os.getenv("ID")
api_key = os.getenv("API_KEY")

def get_yandex_gpt_openai_response(question, history):

   print(identificator, api_key)
   system_prompt = """Ты - эксперт по компьютерной технике и электронике. Твоя специализация:
   # - Компьютерные компоненты (процессоры, видеокарты, память)
   # - Ноутбуки и десктопы
   # - Игровые системы
   # - Рабочие станции
   # - Технические характеристики и сравнения
   # - Рекомендации по выбору оборудования
   #
   # Отвечай подробно, технически грамотно, но понятно для обычных пользователей.
   # Используй конкретные примеры, сравнивай варианты, давай обоснованные рекомендации.
   # Если нужно, используй таблицы для сравнения характеристик."""

   messages = [{
       "role": "system",
       "text": system_prompt
   }]
   for msg in history:
      messages.append({
         "role": msg["role"],
         "text": msg["content"]
      })
   messages.append({
               "role": "user",
               "text": question
           })
   prompt = {
       "modelUri": f"gpt://{identificator}/yandexgpt-4-lite/latest",
       "completionOptions": {
           "stream": False,
           "temperature": 0.6,
           "maxTokens": "2000"
       },
       "messages": messages
   }
   try:
      url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
      headers = {
          "Content-Type": "application/json",
          "Authorization": f"Api-Key {api_key}"
      }

      response = requests.post(url, headers=headers, json=prompt)
      if response.status_code == 200:
         result = response.json()
         return result['result']['alternatives'][0]['message']['text']
      else:
         error_msg = f"API Error {response.status_code}: {response.text}"
         raise Exception(error_msg)
   except Exception as e:
      raise Exception(f"Yandex GPT error: {e}")
