import requests
import json
from tqdm import *
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

query_path = ""

prompt_template = '''
[[任务定义]]
你是一位专业的祝福语质量反思专家，专门负责接收已分析出的祝福语优缺点信息、最初的质量判定及其理由，并基于所有这些信息进行二次分析和反思，最终给出你的最终判定结果和理由，以包含判定结果和最终理由的指定 JSON 格式输出。

[[核心任务]]
你将接收一个包含query、response、positive、negative、initial_judge、initial_reason的输入。你的任务是：
1、完全基于这些提供的输入信息，重新审视最初的判定（initial_judge）和理由（initial_reason）。
2、结合祝福语的优点 (positive) 和缺点 (negative)，独立思考最初的判定是否合理、是否存在偏差。
3、做出你的最终质量判定（0为不好，1为好）。
4、生成一个简明扼要的最终理由，解释你做出最终判定的原因。
5、将最终判定结果（judge）和最终理由（reason），按照指定的 JSON 格式输出。
注意：你判断更改initial_judge的标准十分严格，只有明确、充分的理由认为最初的判定是错误的，才能更改判断。否则，应该维持initial_judge。

[[输入信息]]
query: 一个字符串，描述祝福语的应用场景。
response: 一个字符串，祝福语的原始文本。
positive: 关于祝福语优点的文字描述（通常为 JSON 格式的字符串，包含优点项和具体描述）。
negative: 关于祝福语缺点的文字描述（通常为 JSON 格式的字符串，包含缺点项和具体描述）。
initial_judge: 一个整数 (0或1)，由Judge Agent给出的最初判定结果。
initial_reason: 一个字符串，由Judge Agent给出的最初判定理由。

[[评分输出格式]]
用JSON格式呈现：
```json
{   
    "judge": 0｜1,
    "reason": "最终理由"
}
```

[[评分案例]]
{"query": "祝考试顺利发多少红包吉利", "response": "祝你考试顺利，红包发个88元，吉祥又顺利！","positive": "json\n{\n \"表达流畅自然\": \"语句通顺，读起来流畅，整体感觉自然。\",\n \"内容恰当且符合语境\": \"针对考试场景送出祝福，并提及红包金额，符合用户查询的意图。\",\n \"措辞积极正面\": \"使用了'顺利'、'吉祥'等积极词汇，传递了美好的祝愿。\",\n \"意图纯粹\": \"内容专注于表达对考试顺利和红包金额的祝福，没有其他附加目的。\"\n}\n","negative": "json\n{\n \"措辞不当/负面暗示\": \"将红包金额（88元）与考试顺利直接关联，可能给接收者带来压力，暗示红包金额会影响考试结果。\",\n \"内容空泛/缺乏针对性\": \"祝福语过于通用，没有针对接收者的具体情况（如考试科目、个人特点等）进行个性化表达。\",\n \"文化或习俗禁忌\": \"在中国文化中，直接将金钱与学业成绩挂钩可能被视为不恰当或功利，尤其是在教育场合。\"\n}\n","initial_judge": 0,"initial_reason": "尽管祝福语表达流畅、内容符合语境且措辞积极，但其将红包金额与考试顺利直接关联可能带来压力，且缺乏针对性和个性化表达，同时涉及文化禁忌，因此整体质量不高。"}
输出：
```json
{
    "judge": 0,
    "reason": "对优点和缺点进行复核后，认同最初判定的理由。该祝福语虽然有流畅和积极之处，但将红包金额与考试挂钩、缺乏个性化以及触及文化禁忌等缺点，使其质量不足以被判定为优秀。"
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

