from log import setup_logger
from llm import get_chat_answers_batch, OPENAI_API_KEY
import nest_asyncio, asyncio, json, os
from pathlib import Path
import openai
import PyPDF2
import docx
import glob
import re

logger = setup_logger(__name__)
def pdf_reader(folder):
    documents = []
    # 使用glob模块递归查找所有PDF文件
    # 确保路径格式正确，并设置recursive=True以搜索子文件夹
    pdf_files = []
    if os.path.exists(folder):
        pdf_files = glob.glob(os.path.join(folder, "**/*.pdf"), recursive=True)
        if not pdf_files:
            logger.warning(f"在{folder}中未找到PDF文件")
            # 尝试非递归方式查找
            pdf_files = glob.glob(os.path.join(folder, "*.pdf"))
    else:
        logger.error(f"文件夹{folder}不存在")
    # return pdf_files
    # 同样处理docx文件
    docx_files = glob.glob(os.path.join(folder, "**/*.docx"), recursive=True)
    # 合并两种类型的文件路径列表
    all_files = pdf_files + docx_files
    
    logger.info(f"找到 {len(all_files)} 个文档文件")
    
    for file_path in all_files:
        content = ""
        file_ext = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)
        
        # 根据文件类型选择适当的方法提取文本
        if file_ext == '.pdf':
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        content += page_text + "\n\n"
        
        elif file_ext == '.docx':
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                if para.text:
                    content += para.text + "\n"

        if content.strip():
            documents.append({
                "file_path": file_path,
                "file_name": file_name,
                "content": content
            })
            logger.info(f"成功读取 {file_name} 的内容")
        else:
            logger.warning(f"提取 {file_name} 的内容为空")
    return documents

def message_quailifer(content):
#     file = client.files.create(
#         file=open(pdf_path, "rb"),
#         purpose="user_data"
#     )

#     message = [
#         {'role': 'system',
#          'content': '''你是一位专业的学术文献分析专家。你的任务是分析给定的文献，提取关键信息，
# 并将其与用户正在开发的双模态YOLO算法进行比较分析。请以学术严谨的态度进行分析，客观指出文献中方法的优点和不足，
# 并突出用户方法的创新点和优越性。你的分析将用于用户的开题报告文献综述部分，请你返回大约500字左右的分析结果。'''},
#         {"role": "user",
#          "content": [
#              {
#                  "type": "input_file",
#                  "file_id": file.id,
#                  },
#                  {
#                      "type": "input_text",
#                      "text": f'''
# <我的研究方法>
# 本研究项目针对血液分层检测与分割任务，设计了一种创新的双模态YOLO算法，充分利用白光和蓝光图像的互补特性，
# 通过变形跨模态Transformer技术实现特征的高效对齐与融合，以提升检测精度和鲁棒性。

# 白光图像在边界细节捕捉上具有优势，而蓝光图像因其透光性可提供血液内部结构的清晰信息。算法采用双backbone结构
# 分别提取两种模态的多尺度特征，通过内容感知的空间对齐机制解决因光线折射和光学特性差异导致的空间不对齐问题。

# 其核心创新在于变形跨模态Transformer模块，该模块结合空间变形网络与跨模态注意力机制，通过预测像素级偏移场进行
# 特征空间重采样，并利用多头注意力实现模态间信息交互与增强。此外，算法提出三种灵活的特征融合策略（加权融合、
# 拼接融合、Transformer融合），并在yolo的多尺度特征层（P3/8、P4/16、P5/32）上进行双向特征增强，确保局部细节与
# 全局语义的充分整合。
# </我的研究方法>

# <分析要求>
# 1. 请分析上述文献的主要研究目标、方法和结论，重点关注其在图像处理、计算机视觉、多模态融合或医学图像分析方面的内容
# 2. 评估文献中方法的主要创新点、技术路线和解决的关键问题
# 3. 客观指出文献中方法的局限性和不足之处
# 4. 详细分析该文献的方法与我的双模态YOLO算法的异同点，特别是在以下几个方面：
#    - 多模态特征提取与融合策略
#    - 空间不对齐问题的处理方法
#    - 特征交互与增强机制
#    - 在医学图像分析特别是血液检测任务中的适用性
# 5. 突出我的方法相比该文献的优势和创新点，强调为什么我的方法能够在血液分层检测任务中取得更好的效果
# 6. 总结该文献对我的研究的启发和借鉴意义

# 请以学术严谨的方式进行分析，避免过度推断或主观评价。针对每个方面提供具体、详细的比较分析，而不是笼统的概述。
# </分析要求>
# ''',
# },]}]
#     return message

    """
    构建用于文献综述分析的GPT消息
    
    Args:
        document: 包含文件路径和内容的字典
    
    Returns:
        构建好的消息列表
    """
    # # 如果内容太长，截取一部分
    # content = document["content"]
    # if len(content) > 15000:  # 限制token数量
    #     content = content[:15000] + "...(内容已截断)"
    
    message = [
        {'role': 'system',
         'content': '''你是一位专业的学术文献分析专家。你的任务是分析给定的文献，提取关键信息和文章结构提纲。你的分析将会帮助用户了解文献的具体的工作，并用于用户的论文撰写灵感、论文提纲设计和论文投稿，请你返回大约500字左右的分析结果。'''},
        
        {'role': 'user',
         'content': f'''
<文献>
{content}
</文献>

<我的研究方法>
在本硕士研究中，我围绕血液试管的自动化识别与分类展开，设计并实现了一个基于计算机视觉的多模态检测系统 —— BloodScan。该系统以深度学习为核心，结合机械臂控制与高分辨率图像采集，实现了对血液样本中不同分层（血清层、白膜层、血浆层）的高精度识别与评估。

研究的核心创新在于提出了跨模态注意力机制（CrossModalAttention），通过蓝光与白光图像的融合，提升了血液分层边界的识别精度。在模型设计上，采用双Backbone结构分别提取不同光源下的特征，并在多个尺度上实现跨模态特征融合，兼顾检测精度与计算效率。相比传统YOLO架构，本研究提出的跨模态注意力策略在准确率、召回率上取得显著提升。

系统模块涵盖了数据采集与增强、YOLO模型训练与评估、注意力可视化、以及机械臂和摄像头的软硬件联动控制。为提升模型训练效率，还构建了6通道融合数据集，并制定了严格的“精确一次检测”医学评估标准。

本项目计划不仅在方法上具有创新性，也希望在工程实现层面达到高完整度，具备良好的可复现性和扩展性，为临床血液分析的自动化提供了有力支撑。
</我的研究方法>

<分析要求>
1. 请分析上述文献的主要研究目标、方法和结论，重点关注其在血液分析检测、图像识别、深度神经网络、图像处理、计算机视觉、多模态融合或医学图像分析等方面的内容
2. 总结该文献的提纲和关键信息

请以学术严谨的方式进行分析，避免过度推断或主观评价。针对每个方面提供具体、详细的比较分析，而不是笼统的概述。
</分析要求>
'''}
    ]
#     message = [
#         {'role': 'system',
#          'content': '''你是一位专业的学术文献分析专家。你的任务是分析给定的文献，提取关键信息和文章结构提纲，
# 并将其与用户正在开发的双模态融合YOLO算法进行比较分析。请以学术严谨的态度进行分析，客观指出文献中方法的优点和不足，
# 并突出用户方法的创新点和优越性。你的分析将用于用户的开题报告文献综述部分，请你返回大约500字左右的分析结果。'''},
        
#         {'role': 'user',
#          'content': f'''
# <文献>
# {content}
# </文献>

# <我的研究方法>
# 本研究项目针对血液分层检测与分割任务，设计了一种创新的双模态YOLO算法，充分利用白光和蓝光图像的互补特性，
# 通过变形跨模态Transformer技术实现特征的高效对齐与融合，以提升检测精度和鲁棒性。

# 白光图像在边界细节捕捉上具有优势，而蓝光图像因其透光性可提供血液内部结构的清晰信息。算法采用双backbone结构
# 分别提取两种模态的多尺度特征，通过内容感知的空间对齐机制解决因光线折射和光学特性差异导致的空间不对齐问题。

# 其核心创新在于变形跨模态Transformer模块，该模块结合空间变形网络与跨模态注意力机制，通过预测像素级偏移场进行
# 特征空间重采样，并利用多头注意力实现模态间信息交互与增强。此外，算法提出三种灵活的特征融合策略（加权融合、
# 拼接融合、Transformer融合），并在yolo的多尺度特征层（P3/8、P4/16、P5/32）上进行双向特征增强，确保局部细节与
# 全局语义的充分整合。
# </我的研究方法>

# <分析要求>
# 1. 请分析上述文献的主要研究目标、方法和结论，重点关注其在图像处理、计算机视觉、多模态融合或医学图像分析方面的内容
# 2. 评估文献中方法的主要创新点、技术路线和解决的关键问题
# 3. 客观指出文献中方法的局限性和不足之处
# 4. 详细分析该文献的方法与我的双模态YOLO算法的异同点，特别是在以下几个方面：
#    - 多模态特征提取与融合策略
#    - 空间不对齐问题的处理方法
#    - 特征交互与增强机制
#    - 在医学图像分析特别是血液检测任务中的适用性
# 5. 突出我的方法相比该文献的优势和创新点，强调为什么我的方法能够在血液分层检测任务中取得更好的效果
# 6. 总结该文献对我的研究的启发和借鉴意义

# 请以学术严谨的方式进行分析，避免过度推断或主观评价。针对每个方面提供具体、详细的比较分析，而不是笼统的概述。
# </分析要求>
# '''}
#     ]
    return message

def pdf_analyzer_batch(model = "gpt-4o-2024-11-20",
         pdf_path = "./docs/documentation",
         api_key = OPENAI_API_KEY,
         temperature = 0.7, 
         top_p = 1, 
         frequency_penalty = 0.0, 
         presence_penalty = 0.0,
         folder = "",):
    # client = openai.OpenAI(api_key=api_key)
    pdf_list = pdf_reader(pdf_path)
    messages = list(message_quailifer(pdf_list[i]["content"]) for i in range(len(pdf_list)))
    # print(messages)
    
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
    ))

    return answers

def batch_result_downloader(batch_id):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    batch_status = client.batches.retrieve(batch_id)
    output_file_id = batch_status.output_file_id
    output_file = client.files.content(output_file_id)

    response = [json.loads(line)["response"]["body"]["choices"][0]["message"]["content"] for line in output_file.text.strip().split("\n")]
    return response

if __name__ == "__main__":
    pdf_path = "./test/"
    answers = pdf_analyzer_batch(model = "gpt-4.1", 
                       pdf_path = pdf_path,
                       api_key = OPENAI_API_KEY,
                       temperature = 0.7, 
                       top_p = 1, 
                       frequency_penalty = 0.0, 
                       presence_penalty = 0.0,
                       folder = pdf_path)
    
    # client = openai.OpenAI(api_key=OPENAI_API_KEY)
    # batch_id = "batch_681a43e86dd88190b7a31715109c1018"
    # batch_status = client.batches.retrieve(batch_id)
    # output_file_id = "file-VYVtxNCXHxDsHMYgTXJY7w"
    # output_file_id = batch_status.output_file_id
    # output_file = client.files.content(output_file_id)
    # logger.info(f"✅ 任务完成, 下载结果: {output_file_id}")

    # answers = batch_result_downloader(batch_id = "batch_681a43e86dd88190b7a31715109c1018")

    with open(pdf_path + "answer.txt", "w") as f:
        for answer in answers:
            f.write(answer + "\n\n" + "="*40 + "\n\n")
    
    # export PATH="$PATH:/Users/xin99/Documents/BloodScan"