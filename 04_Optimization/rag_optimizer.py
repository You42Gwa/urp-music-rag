import numpy as np
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm
from langchain_community.embeddings import OllamaEmbeddings

class RAGOptimizer:
    # model_name: Ollama에서 쓸 임베딩 모델명
    # base_url: Ollama 서버 URL
    def __init__(self, model_name="nomic-embed-text", base_url="http://localhost:11434"):
        self.embeddings = OllamaEmbeddings(
            model=model_name, 
            base_url=base_url
        )
        print(f"Initialized OllamaEmbeddings with model: {model_name} at {base_url}")

    # [배치 프로세싱] for 속도개선, text 묶어 embedding
    def get_batch_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        print(f"Processing {len(texts)} texts in {total_batches} batches of size {batch_size}...")

        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Batches"):
            batch_texts = texts[i : i + batch_size]
            try:
                # [수정] self.embed_model -> self.embeddings
                batch_embs = self.embeddings.embed_documents(batch_texts)
                # [수정] extent -> extend
                embeddings.extend(batch_embs)
            except Exception as e:
                print(f"Error embedding batch starting at index {i}: {e}")
            
        return np.array(embeddings)
    
    #[hybrid reranking] vector score(유사도) + Popularity 결합
    def hybrid_rerank(self,
                      retrieved_docs: List[Dict[str, Any]], # FAISS에서 검색된 문서 리스트
                      alpha: float = 0.7, # vector score 가중치 (basic 0.7)
                      beta: float = 0.3) -> List[Dict[str, Any]]: # popularity 가중치 (basic 0.3)
        
        reranked_results = []

        for doc in retrieved_docs:
            # 1. Vector Score (유샤도)
            vector_score = doc.get("score", 0.0)

            # 2. Popularity nomalization! (0~100 -> 0.0~1.0)
            popularity = doc.get('metadata', {}).get('popularity', 0.0)
            norm_popularity = float(popularity) / 100.0

            # 3. Hybrid Scoring (가중치 합산)
            final_score = (vector_score * alpha) + (norm_popularity * beta)

            # 4. 결과 업데이트
            doc['rerank_score'] = final_score
            doc['original_score'] = vector_score
            reranked_results.append(doc)
        
        # 5. 3번 기반으로 내림차순 정렬
        reranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)

        return reranked_results

# =============================
# [사용 예시] 
# =============================

if __name__ == "__main__":
    
    #======================
    # 1. [csv file loadd]
    #======================
    merged_filename = 'final_preprocessed_music_data.csv'
    try:
        merged_df = pd.read_csv(merged_filename)
        print(f" 파일 로드 성공! 총 데이터: {len(merged_df)}개")
        print(f"📋컬럼 목록: {list(merged_df.columns)}") # 컬럼 이름 확인용
    except FileNotFoundError:
        print("파일을 찾을 수 없습니다. 파일 이름을 확인해주세요")
        exit()

    


    #=====================================================
    # 3. [optimizer initialization & batch embeding test]
    #======================================================
        #터미널에서 'ollama pull llama3' 먼저 실행!!!!!!
    optimizer = RAGOptimizer(model_name="nomic-embed-text", base_url="http://localhost:11434")

    lyrics_list = merged_df['lyrics'].head(30).tolist() # 테스트용으로 30개만 사용 (다쓰려면 head지우면됨)


    print("\n[Test 1] Batch Embedding Processing (속도테스트)")
    vectors = optimizer.get_batch_embeddings(lyrics_list, batch_size=10)
    print(f"Complete Embedding ! Vectors shape: {vectors.shape}")

    #=================================
    # 4. [Hybrid Reranking Test]
    #=================================

    print("\n[Test 2] Hybrid Reranking Processing")

    all_docs = []

    for idx, row in merged_df.iterrows():
        # 2번파트 FAISS구현되면 수정 필요
        fake_vector_score = np.random.uniform(0, 1)  # 0~1 사이의 임의의 유사도 점수 생성
        doc = {
            "content": f"{row['track_name']} - {row['artist_name']}",
            "score": fake_vector_score, # 벡터 유사도 (가정)
            "metadata": {
                "popularity": row['popularity'] # 실제 인기도
            }
        }
        all_docs.append(doc)

    # alpha(유사도 반영비율)=0.6, beta(인기도 반영비율)=0.4
    reranked_results = optimizer.hybrid_rerank(all_docs, alpha=0.6, beta=0.4)

    print(f"\n 최종 랭킹 Top 10 (인기도 반영 결과)")
    print(f"{'Rank':<5} | {'Song Title':<35} | {'Popularity':<10} | {'Final Score':<10}")
    print("-" * 75)

    for i, r in enumerate(reranked_results[:10]):
        # 인기도가 높아서 점수가 확 뛴 경우 표시
        pop_mark = "HIT" if r['metadata']['popularity'] >= 80 else ""
        
        print(f"{i+1:<5} | {r['content'][:35]:<35} | {r['metadata']['popularity']:<10} | {r['rerank_score']:.4f} {pop_mark}")