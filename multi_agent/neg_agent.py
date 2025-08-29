import requests
import json
from tqdm import *
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

query_path = ""

prompt_template = '''
[[任务定义]]
你是一位专注于发现祝福语中不足之处的专业分析师。

[[核心任务]]
你的唯一职责是对用户提供的祝福语进行严格、细致的审视，并准确无误地识别出其中存在的所有缺点或问题。

[[分析维度]]
你需要从以下几个主要方面来评估祝福语，识别潜在的缺点。但你的分析不限于这些维度，任何影响祝福语质量的因素都应被考虑：
1、文字/语法错误: 包括错别字、漏字、多字、用词不当、标点错误、语法结构错误等。
2、表达流畅性: 语句是否拗口、不自然，句子之间的衔接是否生硬，是否存在不必要的重复或啰嗦。
3、情感深度/真挚度: 情感表达是否平淡、空洞，缺乏真情实感，是否过于程式化，像套用模板，缺乏个性化温度。
4、创意性/新颖性: 内容是否陈旧、缺乏创意，与其他常见祝福语高度雷同，没有独特的想法或表达方式。
5、逻辑性/恰当性: 祝福语的内容是否符合语境、场合（如生日、节日、升迁、康复等）、送祝福的对象以及你与对方的关系。是否存在逻辑不通、用词不当或不合时宜的地方。
6、内容空泛/缺乏针对性: 祝福语是否过于通用，感觉像是可以发给任何人的模板，缺乏针对接收者个人特点、成就或当前状况的具体细节。
7、措辞不当/负面暗示: 是否使用了可能引起误解、不适、包含负面含义、带来压力（如催婚、催生、过度期望等）或听起来像讽刺的词语或表达。
8、文化或习俗禁忌: 是否触犯了接收方或当前情境下的文化禁忌或习俗上的不妥之处。
9、其他影响质量的问题: 任何其他可能降低祝福语表达效果、真诚度和质量的方面。

[[评分输出格式]]
用JSON格式呈现：
```json
{   
    "缺点1": "原因1",
    "缺点2": "原因2",
    ...
}
```

[[评分案例]]
{"query": "蛇年对爱人的祝福语", "response": "蛇年到来，愿你爱情甜蜜如初，幸福长伴左右。"}
输出：
```json
{
  "内容过于通用，缺乏针对性": "祝福语中的措辞（如'甜蜜如初'，'幸福长伴左右'）可以用于任何一对情侣，没有包含针对你们二人或对方独特之处的细节。",
  "措辞陈旧，缺乏创意": "'甜蜜如初'和'幸福长伴左右'是祝福语中非常常见且程式化的表达，缺乏新意。"
}
```

[[评估对象]]
content
'''
query_list = []
with open(query_path, 'r') as f:
    for line in f:
        line = line.strip()
        query_list.append(line)
           
import os
import json
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch_response(query, url, headers, prompt_template=""):
    payload = json.dumps({
        "model": "",
        "messages": [
            {
                "role": "user",
                "content": prompt_template.replace("content",query)
            },
        ]
    })
    # import pdb; pdb.set_trace()
    response = requests.request("POST", url, headers=headers, data=payload)
    # import pdb; pdb.set_trace()
    while 'error' in response.text:
        print(response.text)
        response = requests.request("POST", url, headers=headers, data=payload)
    payload = response.json()
    output_json = {
        "query": query,
        "request_data": payload
    }
    return output_json

def main():
    res_path_file = ""
    url = ""
    
    headers = {
        'Content-Type': '',
        'Authorization': ''
    }
    
    # 读取已有结果并收集已处理的查询
    processed_queries = set()
    if os.path.exists(res_path_file):
        with open(res_path_file, 'r') as f:
            for line in f:
                try:
                    result = json.loads(line)
                    processed_queries.add(result['query'])
                except json.JSONDecodeError:
                    continue
    else:
           
        os.makedirs(os.path.dirname(res_path_file), exist_ok=True)
        with open(res_path_file, 'w', encoding='utf-8') as f:
            pass


    # 过滤掉已经处理过的查询
    remaining_queries = [query for query in query_list if query not in processed_queries]

    # 以追加模式打开结果文件
    with open(res_path_file, 'a') as f:
        # 创建一个线程池
        with ThreadPoolExecutor(max_workers=100) as executor:
            # 提交任务给线程池
            futures = {executor.submit(fetch_response, query, url, headers, prompt_template): query for query in remaining_queries}
            
            for future in tqdm(as_completed(futures), total=len(remaining_queries)):
                query = futures[future]
                try:
                    output_json = future.result()
                    # 立即将结果写入文件
                    f.write(json.dumps(output_json, ensure_ascii=False) + '\n')
                except Exception as e:
                    print(f"查询 {query} 产生了一个异常: {e}")

# 注意：`max_workers` 参数可以根据你想要的并发线程数量进行调整。

if __name__ == '__main__':

        flag = main()

