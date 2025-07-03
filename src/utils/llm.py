import re, os, time, random, datetime, asyncio, configparser, json, nest_asyncio
from openai import OpenAI
from src.utils.log import setup_logger
from typing import List
import types
import openai

logger = setup_logger(__name__)

config = configparser.ConfigParser()
config.read("./src/config/config.ini", encoding="utf-8")

OPENAI_API_KEY = "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"

pr_price_dict = {
    "gpt-4o-2024-11-20": 2.5,
    "gpt-4.1": 2,
    "gpt-4.1-mini": 0.4,
    "gpt-4.1-nano": 0.1,

    "deepseek-chat": 0.27,
    "deepseek-reasoner": 0.55,
}
rt_price_dict = {
    "gpt-4o-2024-11-20": 10,
    "gpt-4.1": 8,
    "gpt-4.1-mini": 1.6,
    "gpt-4.1-nano": 0.4,
    
    "deepseek-chat": 1.1,
    "deepseek-reasoner": 2.19,
}

def message_quailifer():
    return ["hello", "world"]

async def async_llm_reasoning(model, system_message, user_message, task_name,
                              temperature, max_tokens, max_retries = 5, 
                              LONG_MODE = False, STREAM_MODE = False):
    try:
        GEMINI_DOWN = False
        time_stamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_path = f"results/app_prompts/{task_name}_{time_stamp}.txt"
        prompt  = f"""Prompt: \nsystem_message: {system_message},\nuser_message: {user_message}"""
        
        with open(prompt_path, "w", encoding="utf-8") as file:
            file.write(prompt)
        # logger.info(f"Model in use: {model}")
        logger.info(f"llm prompt saved to {prompt_path}")

        if LONG_MODE:
            client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
            with client.beta.messages.stream(
                model = model,
                max_tokens = max_tokens,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 5000},
                system=system_message,
                messages=[{"role": "user", "content": user_message,}],
                betas=["token-efficient-tools-2025-02-19","output-128k-2025-02-19"]) as stream:
                for event in stream:
                    if event.type == "content_block_start":
                        logger.info(f"\nStarting {event.content_block.type} block...")
                    elif event.type == "content_block_delta":
                        if event.delta.type == "thinking_delta":
                            # logger.info(f"Thinking {event.delta.thinking}")
                            yield event.delta.thinking, ""
                            await asyncio.sleep(0)
                        elif event.delta.type == "text_delta":
                            yield "", event.delta.text
                            await asyncio.sleep(0)
                    elif event.type == "content_block_stop":
                        logger.info("\nBlock complete.")
        
        elif 'gemini' in model:
            retries = 0
            while retries < max_retries:
                try:
                    if 'thinking' in model:
                        client_gem = genai.Client(api_key=GEMINI_API_KEY)
                    else:
                        client_gem = genai.Client(api_key=GEMINI_API_KEY,
                                                  http_options={'api_version':'v1alpha'})
                    response = client_gem.models.generate_content(
                        model = model, 
                        contents = user_message,
                        config = {
                            'system_instruction': system_message,
                            'temperature': temperature,
                            'max_output_tokens': max_tokens,
                        }
                    )
                    result = response.text
                    if result == None:
                        GEMINI_DOWN = True
                    break
                except Exception as e:
                    retries += 1
                    logger.warning(f"Gemini调用, 错误: {e}, 第 {retries} 次重试")
                    logger.warning(f"Gemini返回, {response}")
                    if retries < max_retries:
                        time.sleep(10)
            if retries == max_retries:
                GEMINI_DOWN = True
                model = "gpt-4o-2024-11-20"

        elif 'claude' in model:
            client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
            response = client.messages.create(
                model = model, 
                max_tokens=max_tokens, 
                temperature=temperature, 
                stream=STREAM_MODE,
                system = system_message,
                messages = [{"role": "user", "content": user_message,}])
            if STREAM_MODE:
                for event in response:
                    if event.type == "content_block_delta" and event.delta.type == "text_delta":
                        yield event.delta.text
                        await asyncio.sleep(0)
            else:
                result = response.content[0].text
                yield result

        elif 'ep' in model:
            client = OpenAI(
                base_url="https://ark.cn-beijing.volces.com/api/v3/",
                api_key=DEEPSK_API_KEY)
            response = client.chat.completions.create(
                 model = model,
                 max_tokens = max_tokens, 
                 temperature = temperature, 
                 stream = STREAM_MODE,
                 messages = [
                     {"role": "system", "content": system_message},
                     {"role": "user", "content": user_message}])
            if STREAM_MODE:
                yield "Thinking..."
                await asyncio.sleep(0)
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content
                        await asyncio.sleep(0)
            else:
                result = response.choices[0].message.content
                yield result
            
        elif 'gpt' in model or GEMINI_DOWN:
            logger.info(f"GEMINI_DOWN is {GEMINI_DOWN}")
            client = OpenAI(api_key = OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream = STREAM_MODE,
            )
            if STREAM_MODE:
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content
                        await asyncio.sleep(0)
            else:
                result = response.choices[0].message.content
                yield result
        else:
            logger.error(f"wrong model input of {model}")
    except Exception as e:
        logger.error(f"{str(e)}")
        yield f"Error: {str(e)}"

async def llm_multiple_responses(model, system_message, user_message, task_name,
                                 temperature, max_tokens, n_responses):
    async def single_response(idx):
        """单个响应的生成器，带编号"""
        full_response = ""
        async for chunk in async_llm_reasoning(
            model, system_message, user_message, task_name,
            temperature, max_tokens, STREAM_MODE=True
        ):
            full_response += chunk or ""
            yield idx, chunk  # 返回编号和 chunk

    generators = [single_response(i) for i in range(n_responses)]
    tasks = [asyncio.create_task(anext(gen)) for gen in generators]

    while tasks:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        new_tasks = []

        for task in done:
            try:
                idx, chunk = task.result()
                yield idx, chunk
                gen = generators[idx]
                new_tasks.append(asyncio.create_task(anext(gen)))
            except StopAsyncIteration:
                # 当前 generator 已完成
                pass

        tasks = list(pending) + new_tasks

async def get_chat_answers_batch(folder, api_key, messages, 
                                 model="gpt-4o-2024-11-20", 
                                 temperature=0,
                                 top_p=0.9,
                                 frequency_penalty=0,
                                 presence_penalty=0,
                                 response_format="text", 
                                 base_url="https://api.openai.com/v1",
                                 batch_wait_time=60):
    token_pr_price = pr_price_dict[model]/2
    token_rt_price = rt_price_dict[model]/2
    # 创建client时设置超时时间
    client = openai.OpenAI(
        api_key=api_key, 
        base_url=base_url,
        timeout=60.0,  # 增加超时时间到60秒
        max_retries=3  # 添加重试次数
    )
    # 1. 新建 JSONL 文件
    batch_file_path = folder + "/batch_requests.jsonl"
    with open(batch_file_path, "w", encoding="utf-8") as f:
        for id, msg in enumerate(messages):
            request = {
                "custom_id": f"request-{id}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": msg,
                    "temperature": temperature,
                    "max_tokens": 8000,
                    "response_format": {"type": response_format},
                    "top_p": top_p,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty,
                }
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")

    # 2. 上传 JSONL 文件
    with open(batch_file_path, "rb") as file:
        upload_response = client.files.create(file=file, purpose="batch")
    file_id = upload_response.id
    logger.info(f"✅ 文件上传成功: {file_id}")

    # 3. 创建 Batch 任务
    batch_response = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window = "24h",
    )
    batch_id = batch_response.id
    logger.info(f"✅ Batch 任务创建成功: {batch_id}")

    # 4. 轮询任务状态
    while True:
        try:
            batch_status = client.batches.retrieve(batch_id)
            status = batch_status.status
            
            if status == "in_progress" and hasattr(batch_status, "request_counts"):
                counts = batch_status.request_counts
 
                total = counts.total if hasattr(counts, "total") else 0
                completed = counts.completed if hasattr(counts, "completed") else 0
                failed = counts.failed if hasattr(counts, "failed") else 0 
                pending = total - completed - failed
                
                logger.info(f"⏳ Batch 任务状态: {status}: {completed} 完成 - {failed} 失败 - {pending} 等待中 - {total} 总数")
            else:
                logger.info(f"⏳ Batch 任务状态: {status}")

            if status in ["completed", "failed"]:
                break
                
        except openai.APITimeoutError:
            logger.warning(f"❗ 获取批处理状态时超时，等待{batch_wait_time}秒后重试")
        except Exception as e:
            logger.warning(f"❗ 获取批处理状态时出错: {e}, 等待{batch_wait_time}秒后重试")
        
        await asyncio.sleep(batch_wait_time)

    # 5. 处理结果
    if status == "completed":
        try:
            output_file_id = batch_status.output_file_id
            output_file = client.files.content(output_file_id)
            logger.info(f"✅ 任务完成, 下载结果: {output_file_id}")

            response = [json.loads(line)["response"]["body"]["choices"][0]["message"]["content"] for line in output_file.text.strip().split("\n")]
            token_pr_usage = sum([json.loads(line)["response"]["body"]["usage"]["prompt_tokens"] for line in output_file.text.strip().split("\n")])
            token_rt_usage = sum([json.loads(line)["response"]["body"]["usage"]["completion_tokens"] for line in output_file.text.strip().split("\n")])
            token_usd_usage= token_pr_usage/1000000*token_pr_price + token_rt_usage/1000000*token_rt_price
            logger.info(f"🪙 任务消耗 {token_pr_usage} + {token_rt_usage} Token")
            logger.info(f"💵 任务消耗 {token_usd_usage} USD")
            return response
        except Exception as e:
            logger.error(f"❌ 下载或处理结果时发生错误: {e}")
            # 尝试恢复：通过batch_id查询处理结果
            logger.info("🔄 尝试通过其他方式获取批处理结果...")
            # 这里可以添加备用获取结果的方法
    
    logger.error("❌ Batch 任务失败")
    return []

def test(model, 
         api_key, 
         api_key_pool, 
         base_url = None, 
         temperature = 0.7, top_p = 1, frequency_penalty = 0.0, presence_penalty = 0.0,
         folder = "", BATCH_MODE = False):
    
    messages = list(message_quailifer())
    
    if BATCH_MODE:
        # ✅ 调用 batch 版本
        nest_asyncio.apply()
        answers = asyncio.run(get_chat_answers_batch(
            folder = folder,
            api_key=api_key,  # 取第一个 API Key
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            base_url=base_url
        ))
    else:
        nest_asyncio.apply()
        # get_chat_answers现在返回(contents, ask_result_dict)而不是(results, ask_result_dict)
        answers, _ = asyncio.run(get_chat_answers(
            api_key_pool=api_key_pool,
            base_url=base_url,
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        ))
    # # parse
    # results = []
    # for id, result in enumerate(answers):
    #     # print(result)
    #     try:
    #         extraction_result = json.loads(result)
    #         keywords = extraction_result.get('keywords', [])
    #         entities = extraction_result.get('entities', [])

    #         keywords_list = [item['keyword'] for item in keywords if item['relevance'] == '高']
    #         entities_list = [item['entity_name'] for item in entities]

    #         # print(f"成功提取关键词: {keywords_list}")
    #         # print(f"成功提取实体: {entities_list}")
    #         results.append(keywords_list + entities_list)
    #     except Exception as e:
    #         logger.error(f"LLM returns extractions error: {e}")
    #         results.append([])
    
    # return 
    return answers
    
def get_embedding_gemini(self, texts: List[str], title: str = None) -> List[List[float]]:
    """批量获取Gemini模型的文本嵌入向量
    
    Args:
        texts: 输入文本列表
        title: 文档标题
    Returns:
        嵌入向量列表，每个文本对应一个向量
    """
    if not self.gemini_client:
        raise ValueError("Gemini客户端未初始化")
    
    max_tokens = 8192  # 每个chunk的最大总长度限制
    max_text_length = max_tokens   # 单个文本的最大长度限制
    max_texts_per_chunk = 5  # 每个chunk最多包含的文本数
    rest_interval = 3       # 每处理一个chunk后的休息时间（秒）
    
    # API密钥池配置
    api_keys = self.api_keys
    if not api_keys or not api_keys[0]:
        raise ValueError("未配置有效的API密钥")
    
    current_api_index = 0
    current_api_key = api_keys[current_api_index]
    
    # 将文本分成多个chunk
    chunks = []
    current_chunk = []
    current_chunk_length = 0
    
    for text in texts:
        # 如果单个文本超过限制，需要截断
        if len(text) > max_text_length:
            logger.warning(f"单个文本长度({len(text)})超过最大限制({max_text_length})，将被截断")
            text = text[:max_text_length]
        
        text_length = len(text)
        
        # 如果当前chunk已满（文本数达到上限或总长度超过限制），创建新chunk
        if (len(current_chunk) >= max_texts_per_chunk or 
            (current_chunk_length + text_length > max_tokens and current_chunk)):
            chunks.append(current_chunk)
            current_chunk = []
            current_chunk_length = 0
        
        # 添加文本到当前chunk
        current_chunk.append(text)
        current_chunk_length += text_length
    
    # 添加最后一个chunk（如果有内容）
    if current_chunk:
        chunks.append(current_chunk)
    
    logger.info(f"将{len(texts)}个文本分成{len(chunks)}个块进行处理")
    
    # 处理每个chunk
    all_results = []
    for i, chunk in enumerate(chunks):
        logger.info(f"处理第{i+1}/{len(chunks)}个块，包含{len(chunk)}个文本，总长度约{sum(len(t) for t in chunk)}字符")
        
        # 标记是否成功处理当前chunk
        chunk_processed = False
        api_attempts = 0
        
        # 尝试使用不同的API密钥处理当前chunk
        while not chunk_processed and api_attempts < len(api_keys):
            try:
                # 更新客户端使用当前API密钥
                self.gemini_client = genai.Client(api_key=current_api_key)
                logger.info(f"使用API密钥 #{current_api_index+1} 处理第{i+1}个块")
                
                result = self.gemini_client.models.embed_content(
                    model=self.model_name,
                    contents=chunk,
                    config=types.EmbedContentConfig(
                        title=title,
                        task_type=self.task_type,
                        output_dimensionality=N_DIM)
                )
                
                # 提取嵌入向量
                chunk_embeddings = [embedding.values for embedding in result.embeddings]
                all_results.extend(chunk_embeddings)
                logger.info(f"成功处理第{i+1}个块，获取了{len(chunk_embeddings)}个嵌入向量")
                
                # 标记成功处理
                chunk_processed = True
                
            except Exception as e:
                # 记录错误并切换到下一个API密钥
                logger.warning(f"使用API密钥 #{current_api_index+1} 处理第{i+1}个块时出错: {str(e)}")
                api_attempts += 1
                
                # 更新API密钥索引，如果已经用完所有密钥，从头开始
                current_api_index = (current_api_index + 1) % len(api_keys)
                current_api_key = api_keys[current_api_index]
        
        # 检查是否成功处理
        if not chunk_processed:
            error_msg = f"所有API密钥都无法处理第{i+1}个块"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        # 处理完一个chunk后休息
        if i < len(chunks) - 1:  # 如果不是最后一个chunk
            # logger.info(f"休息{rest_interval}秒后继续处理下一个块")
            time.sleep(rest_interval)
    
    return all_results