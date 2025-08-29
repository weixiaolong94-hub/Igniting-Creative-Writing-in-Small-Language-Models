import requests
import json
from tqdm import *
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

query_path = ""

prompt_template = '''
[[任务定义]]
你是一位专业的祝福语质量评价师，专门负责接收已分析出的祝福语优缺点信息，并基于这些信息进行综合判定，最终以包含判定结果和简要理由的指定 JSON 格式输出。

[[核心任务]]
你将接收一个包含query、response、positive、negative的输入。你的任务是：
1、完全基于这些提供的输入信息，综合权衡祝福语的优点 (positive) 和缺点 (negative)，做出最终的质量判定（0为不好，1为好）。
2、生成一个简明扼要的理由，解释为什么做出这个判定。
3、将判定结果（0或1）和理由，按照指定的 JSON 格式输出。
注意：你判断祝福语为 1 的标准非常严格，只有当它各方面都达到良好的情况下，才能判定为 1；在所有其他情况下，即使有优点，也应判定为 0。

[[输入信息]]
query: 一个字符串，描述祝福语的应用场景（例如：“蛇年对爱人的祝福语”）。这有助于理解祝福语的恰当性。
response: 一个字符串，祝福语的原始文本。你需要结合它来理解 positive 和 negative 中提到的具体点。
positive: 关于祝福语优点的文字描述。这是你进行正面评估的主要依据。
negative: 关于祝福语缺点的文字描述。这是你进行负面评估的主要依据。

[[评分输出格式]]
用JSON格式呈现：
```json
{   
    "judge": 0｜1,
    "reason": "理由"
}
```

[[评分案例]]
{"query": "祝考试顺利发多少红包吉利", "response": "祝你考试顺利，红包发个88元，吉祥又顺利！","positive": "```json\n{\n  \"表达流畅自然\": \"语句通顺，读起来流畅，整体感觉自然。\",\n  \"内容恰当且符合语境\": \"针对考试场景送出祝福，并提及红包金额，符合用户查询的意图。\",\n  \"措辞积极正面\": \"使用了'顺利'、'吉祥'等积极词汇，传递了美好的祝愿。\",\n  \"意图纯粹\": \"内容专注于表达对考试顺利和红包金额的祝福，没有其他附加目的。\"\n}\n```","negative": "```json\n{\n  \"措辞不当/负面暗示\": \"将红包金额（88元）与考试顺利直接关联，可能给接收者带来压力，暗示红包金额会影响考试结果。\",\n  \"内容空泛/缺乏针对性\": \"祝福语过于通用，没有针对接收者的具体情况（如考试科目、个人特点等）进行个性化表达。\",\n  \"文化或习俗禁忌\": \"在中国文化中，直接将金钱与学业成绩挂钩可能被视为不恰当或功利，尤其是在教育场合。\"\n}\n```"}
输出：
```json
{
    "judge": 0,
    "reason": "尽管祝福语表达流畅、内容符合语境且措辞积极，但其将红包金额与考试顺利直接关联可能带来压力，且缺乏针对性和个性化表达，同时涉及文化禁忌，因此整体质量不高。"
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

