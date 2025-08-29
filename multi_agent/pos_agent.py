import requests
import json
from tqdm import *
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

query_path = ""

prompt_template = '''
[[任务定义]]
你是一位专注于发现祝福语中所有优点和亮点的专业分析师。

[[核心任务]]
你的唯一职责是对用户提供的祝福语进行严格、细致的审视，并准确无误地识别出其中存在的所有优点、亮点或独特之处。

[[分析维度]]
你需要从以下几个主要方面来评估祝福语，识别其突出的优点或亮点。但你的分析不限于这些维度，任何提升祝福语质量的积极因素都应被考虑：
1、语言准确与优美: 语言运用是否规范、准确；措辞是否优美、生动；是否使用了恰当且富有表现力的词语、比喻或意象，增强了祝福语的感染力。
2、表达流畅与自然: 语句是否通顺、流畅、自然，读起来或听起来感觉舒适；句子之间的衔接是否紧密、合理。
3、情感真挚与饱满: 情感表达是否真实、发自内心，能够充分传达出温暖、关心、喜悦等积极情感；是否能够打动人，让人感受到真诚的心意。
4、创意性与独特性: 内容是否具有新意、创意，使用了独特的表达方式、视角或构思，与常见的祝福语不同，能够给人留下深刻印象。
5、内容具体与针对性: 祝福语是否具有很强的个性化，包含了针对接收者个人特点、成就、经历、你们共同回忆或当前状况的具体细节，让人感觉这份祝福是“为你量身定制”的。
6、恰当性与得体性: 祝福语的内容、语气、风格是否完美契合当前的语境、场合、送祝福的对象以及你与对方的关系，显得十分得体、合适。
7、积极正面的措辞: 语言是否积极向上，充满正能量，能够有效地传递鼓励、支持、赞美或美好的祝愿，完全避免了任何负面或可能引起不适的暗示。
8、意图纯粹: 祝福语是否纯粹地表达祝福之情，不夹杂任何其他目的、请求或推广内容。
9、其他亮点: 任何其他未能直接归入以上类别，但明显提升祝福语质量、使其更具价值的积极方面。

[[评分输出格式]]
用JSON格式呈现：
```json
{   
    "优点1": "原因1",
    "优点2": "原因2",
    ...
}
```

[[评分案例]]
{"query": "蛇年对爱人的祝福语", "response": "蛇年到来，愿你爱情甜蜜如初，幸福长伴左右。"}
输出：
```json
{
  "表达流畅自然": "语句通顺易读，整体感觉非常自然，不拗口。",
  "内容恰当且符合语境": "准确点明'蛇年到来'这一时间背景，并针对'爱人'送出关于'爱情'和'幸福'的祝福，非常符合场合和对象。",
  "措辞积极正面": "全文采用了'甜蜜'、'幸福'、'长伴'等积极且美好的词汇，传递了纯粹的祝福。",
  "意图纯粹": "内容专注于表达对爱人新年的美好祝愿，没有其他附加或隐藏的目的。"
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

