import os, asyncio, nest_asyncio
import openai
import google.generativeai as genai
from openai import OpenAI
from anthropic import Anthropic
from typing import List, Dict, Optional
from loguru import logger
from asyncio import Semaphore
from utils.json_utils import maximal_parsable_json

class LLMService:
    """LLM服务类, 基于openrouter api, 支持多模型并发请求"""
    
    def __init__(
        self, 
        provider: str = "openrouter", 
        model: str = "google/gemini-2.5-pro-preview",
        max_retries: int = 3,
        rate_limit: int = 10,
        api_keys: List[str] = None
        ):
        self.provider = provider.lower()
        self.model = model  
        self.max_retries = max_retries
        self.rate_limit = rate_limit
        self.api_keys = api_keys or [os.getenv('OPENAI_API_KEY', '')]
        
        # 设置base_url
        if self.provider == "openrouter":
            self.base_url = "https://openrouter.ai/api/v1"
        elif self.provider == "openai":
            self.base_url = "https://api.openai.com/v1"
        else:
            self.base_url = "https://api.openai.com/v1"
            
        nest_asyncio.apply()

    def _split_list(self, lst: List, n: int) -> List[List]:
        """Split list into n sublists"""
        if n <= 0:
            return [lst]
        k, m = divmod(len(lst), n)
        return [lst[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]
    
    async def _chat_answer_asy(
        self,
        semaphore: Semaphore,
        message: List[Dict],
        key: str,
        top_p: float = 1,
        temperature: float = 0.7,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        response_format: str = 'text'
    ) -> Optional[str]:
        """Async helper for single chat completion"""
        async with semaphore:
            # 异步上下文管理器, 使用semaphore限制并发数
            client = OpenAI(base_url=self.base_url, api_key=key)
            retries = 0
            while retries <= self.max_retries:
                try:
                    if 'google/gemini' in self.model:
                        answer = await asyncio.to_thread(
                            client.chat.completions.create,
                            model=self.model,
                            temperature=temperature,
                            top_p=top_p,
                            messages=message
                        )
                    else:
                        answer = await asyncio.to_thread(
                            client.chat.completions.create,
                            model=self.model,
                            temperature=temperature,
                            top_p=top_p,
                            frequency_penalty=frequency_penalty,
                            presence_penalty=presence_penalty,
                            response_format={'type': response_format},
                            messages=message
                        )
                    return answer.choices[0].message.content
                    
                except Exception as e:
                    print(f"Error: {e}, Retrying...")
                    retries += 1
                    await asyncio.sleep(5)
                    if retries > self.max_retries:
                        print("Max retries reached. Aborting task.")
                        return None
            
            return None
    
    async def _get_chat_answers(
        self,
        messages: List[List[Dict[str, str]]],
        temperature: float = 0,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        response_format: str = 'text'
    ) -> List[Optional[str]]:
        """Get multiple chat answers in parallel"""
        semaphore = Semaphore(self.rate_limit)
        tasks = []
        message_lists = self._split_list(messages, len(self.api_keys))
        
        for index in range(len(message_lists)):
            messages_for_key = message_lists[index]
            current_key = self.api_keys[index]
            for message in messages_for_key:
                task = asyncio.create_task(
                    self._chat_answer_asy(
                        semaphore,
                        message,
                        current_key,
                        temperature=temperature,
                        presence_penalty=presence_penalty,
                        frequency_penalty=frequency_penalty,
                        response_format=response_format
                    )
                )
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    def generate_sync(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        response_format: str = 'text'
    ) -> str:
        """Generate a single response from LLM synchronously"""
        results = asyncio.run(self._get_chat_answers(
            [messages],
            temperature=temperature,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            response_format=response_format
        ))
        
        if results and results[0] and not isinstance(results[0], Exception):
            return results[0]
        else:
            raise Exception(f"LLM generation failed: {results[0] if results else 'Unknown error'}")
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        response_format: str = 'text'
    ) -> str:
        """Generate a single response from LLM"""
        results = await self._get_chat_answers(
            [messages],
            temperature=temperature,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            response_format=response_format
        )
        
        if results and results[0] and not isinstance(results[0], Exception):
            return results[0]
        else:
            raise Exception(f"LLM generation failed: {results[0] if results else 'Unknown error'}")
    
    def generate_batch_sync(
        self,
        messages_list: List[List[Dict[str, str]]],
        temperature: float = 0.7,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        response_format: str = 'text'
    ) -> List[Optional[str]]:
        """Generate multiple responses in parallel synchronously"""
        return asyncio.run(self._get_chat_answers(
            messages_list,
            temperature=temperature,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            response_format=response_format
        ))
    
    async def generate_batch(
        self,
        messages_list: List[List[Dict[str, str]]],
        temperature: float = 0.7,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        response_format: str = 'text'
    ) -> List[Optional[str]]:
        """Generate multiple responses in parallel"""
        return await self._get_chat_answers(
            messages_list,
            temperature=temperature,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            response_format=response_format
        )
    
    def parse_json_response(self, response: str) -> Optional[Dict]:
        """Parse JSON from LLM response"""
        return maximal_parsable_json(response)
    
    def get_embedding(self, text: str) -> List[float]:
        """获取文本嵌入向量"""
        try:
            if self.provider == "openai":
                response = openai.embeddings.create(
                    model="text-embedding-ada-002",
                    input=text
                )
                return response.data[0].embedding
                
            elif self.provider == "gemini":
                response = genai.embed_content(
                    model="models/embedding-001",
                    content=text
                )
                return response['embedding']
                
            else:
                logger.warning(f"{self.provider} 不支持嵌入功能")
                return []
                
        except Exception as e:
            logger.error(f"获取嵌入向量失败: {e}")
            return [] 