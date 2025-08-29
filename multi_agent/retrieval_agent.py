import os
import json
import logging
import argparse
from typing import List, Dict

import faiss
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FewShotRetriever:

    def __init__(self, model_name: str, corpus_path: str, cache_dir: str = './cache'):
        """
        初始化检索器。

        Args:
            model_name (str): 用于编码的 SentenceTransformer 模型名称。
            corpus_path (str): 高质量语料库文件的路径 (JSONL 格式)。
            cache_dir (str): 用于存储编码向量和 Faiss 索引的缓存目录。
        """
        self.model_name = model_name
        self.corpus_path = corpus_path
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"使用设备: {self.device}")
        
        self.model = SentenceTransformer(self.model_name, device=self.device)
        
        self.corpus_data = None
        self.corpus_embeddings = None
        self.index = None

    def _load_corpus(self):
        """加载并解析语料库文件。"""
        logging.info(f"从 {self.corpus_path} 加载语料库...")
        self.corpus_data = []
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    self.corpus_data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    logging.warning(f"跳过格式错误的行: {line.strip()}")
        logging.info(f"语料库加载完成，共 {len(self.corpus_data)} 条记录。")

    def _build_or_load_index(self):
        """构建 Faiss 索引或从缓存加载。"""
        embedding_cache_path = os.path.join(self.cache_dir, f"{os.path.basename(self.model_name)}_embeddings.npy")
        index_cache_path = os.path.join(self.cache_dir, f"{os.path.basename(self.model_name)}_index.faiss")

        if os.path.exists(embedding_cache_path) and os.path.exists(index_cache_path):
            logging.info("从缓存加载编码向量和 Faiss 索引...")
            self.corpus_embeddings = np.load(embedding_cache_path)
            self.index = faiss.read_index(index_cache_path)
            logging.info("加载完成。")
        else:
            logging.info("未找到缓存，开始构建新的索引...")
            if self.corpus_data is None:
                self._load_corpus()

            texts_to_encode = [item['normalized_query'] for item in self.corpus_data]
            
            logging.info(f"开始使用 {self.model_name} 编码 {len(texts_to_encode)} 条语料库文本...")
            self.corpus_embeddings = self.model.encode(
                texts_to_encode, 
                show_progress_bar=True,
                batch_size=128 
            )
            
            faiss.normalize_L2(self.corpus_embeddings)
            
            np.save(embedding_cache_path, self.corpus_embeddings)
            logging.info(f"编码向量已缓存至 {embedding_cache_path}")

            d = self.corpus_embeddings.shape[1]
            self.index = faiss.IndexFlatIP(d)
            self.index.add(self.corpus_embeddings)
            
            faiss.write_index(self.index, index_cache_path)
            logging.info(f"Faiss 索引已缓存至 {index_cache_path}")
            
        logging.info(f"索引构建完成，包含 {self.index.ntotal} 个向量。")

    def retrieve(self, queries: List[str], k_search: int, k_final: int) -> List[Dict]:
        """
        为查询列表执行检索和重排。

        Args:
            queries (List[str]): 需要检索的查询字符串列表。
            k_search (int): Faiss 搜索时召回的候选数量，应大于 k_final。
            k_final (int): 最终需要返回的高质量示例数量。

        Returns:
            List[Dict]: 格式化后的结果列表。
        """
        if self.index is None:
            self._build_or_load_index()
            
        logging.info(f"开始为 {len(queries)} 条查询进行检索...")
        query_embeddings = self.model.encode(
            queries, 
            show_progress_bar=True,
            batch_size=128
        )
        faiss.normalize_L2(query_embeddings)

        _, I = self.index.search(query_embeddings, k_search)
        
        results = []

        for i, query in enumerate(tqdm(queries, desc="处理查询结果")):
            candidate_indices = I[i]

            candidates = [self.corpus_data[idx] for idx in candidate_indices if idx != -1]

            candidates.sort(key=lambda x: x.get('click_num', 0), reverse=True)

            top_k_candidates = candidates[:k_final]
            
            few_shot_list = [
                {"query": item['normalized_query'], "response": item['copy_content']}
                for item in top_k_candidates
            ]
            
            results.append({
                "query": query,
                "few_shot": few_shot_list
            })
            
        return results

    @staticmethod
    def save_results(results: List[Dict], output_path: str):
        logging.info(f"将 {len(results)} 条结果保存到 {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        logging.info("保存完成。")

def main():
    parser = argparse.ArgumentParser(description="为查询检索高质量的 few-shot 示例。")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--query_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--k_search", type=int)
    parser.add_argument("--k_final", type=int)
    parser.add_argument("--cache_dir", type=str, default="./retrieval_cache")

    args = parser.parse_args()

    try:
        retriever = FewShotRetriever(
            model_name=args.model_name,
            corpus_path=args.corpus_path,
            cache_dir=args.cache_dir
        )
        
        with open(args.query_path, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
            
        retrieval_results = retriever.retrieve(
            queries=queries,
            k_search=args.k_search,
            k_final=args.k_final
        )
        
        FewShotRetriever.save_results(retrieval_results, args.output_path)
        
        logging.info("检索任务成功完成！")

    except Exception as e:
        logging.error(f"处理过程中发生错误: {e}", exc_info=True)


if __name__ == "__main__":
    main()