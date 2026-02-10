import numpy as np
import matplotlib.pyplot as plt
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.vectorstores.utils import DistanceStrategy

# Numpy 기반 Dense Retrieval - 쿼리와 가장 가까운 k개의 문서 인덱스와 점수를 반환(L2이용)
def retrieve_dense(document_embs, query_emb, topk=5):
    # 벡터 거리 계산 (L2)
    distances = np.linalg.norm(document_embs - query_emb, axis=1)
    
    # 거리가 짧은 순으로 정렬
    top_indices = np.argsort(distances)[:topk]
    top_scores = distances[top_indices]
    
    return top_indices, top_scores

# 리트리버 생성(L2, Cos, FAISS, Chroma)
def create_retrievers(docs, embedding_model):
    # FAISS 설정
    faiss_l2 = FAISS.from_documents(docs, embedding_model, distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE)
    faiss_cos = FAISS.from_documents(docs, embedding_model, distance_strategy=DistanceStrategy.COSINE)
    
    # Chroma 설정
    chroma_l2 = Chroma.from_documents(docs, embedding_model, collection_name="m_l2", collection_metadata={"hnsw:space": "l2"})
    chroma_cos = Chroma.from_documents(docs, embedding_model, collection_name="m_cos", collection_metadata={"hnsw:space": "cosine"})
    
    retrievers = {
        "FAISS (L2)": faiss_l2.as_retriever(search_kwargs={"k": 5}),
        "FAISS (Cos)": faiss_cos.as_retriever(search_kwargs={"k": 5}),
        "Chroma (L2)": chroma_l2.as_retriever(search_kwargs={"k": 5}),
        "Chroma (Cos)": chroma_cos.as_retriever(search_kwargs={"k": 5})
    }
    
    return retrievers

# 성능 평가
def calculate_recall_at_k(retriever, eval_data, k_list=[1, 3, 5, 10]):
    recalls = []
    for k in k_list:
        hits = 0
        # 리트리버의 k값 동적 변경
        retriever.search_kwargs["k"] = k
        
        for item in eval_data:
            query = item['query'].split(":")[-1].strip().replace('"', '')
            true_content = item['ground_truth']
            
            docs = retriever.invoke(query)
            if any(true_content == d.page_content for d in docs):
                hits += 1
        
        recall = hits / len(eval_data)
        recalls.append(recall)
    return recalls

# 최적의 리트리버 반환
def get_best_retriever(retrievers, eval_data, target_k=5):
    best_score = -1.0
    best_name = ""
    best_retriever_obj = None
    
    k_list = [1, 3, 5, 10]
    target_idx = k_list.index(target_k) if target_k in k_list else 2
    
    for name, retriever in retrievers.items():
        # 각 리트리버 평가
        scores = calculate_recall_at_k(retriever, eval_data, k_list=k_list)
        current_score = scores[target_idx]
        print(f"> {name} Score: {current_score:.2%}")
        
        # 최고 점수 업데이트
        if current_score > best_score:
            best_score = current_score
            best_name = name
            best_retriever_obj = retriever
        # 동일한 경우가 많은데 대용량이 아니므로 파이스 우선 선택
        elif current_score == best_score and "FAISS" in name:
            best_name = name
            best_retriever_obj = retriever

    print(f"선정된 최적 리트리버: {best_name} (Hit Rate: {best_score:.2%})")
    return best_retriever_obj

# 성능 시각화
def visualize_performance(results, k_values):
    plt.figure(figsize=(10, 6))
    colors = {'FAISS (L2)': 'blue', 'FAISS (Cos)': 'cyan', 'Chroma (L2)': 'red', 'Chroma (Cos)': 'orange'}
    
    for name, recalls in results.items():
        plt.plot(k_values, recalls, marker='o', label=name, color=colors.get(name, 'black'), linestyle='--')
    
    plt.title('Retriever Performance Comparison: L2 vs Cosine', fontsize=14)
    plt.xlabel('Number of Retrieved Documents (K)', fontsize=12)
    plt.ylabel('Hit Rate (Recall)', fontsize=12)
    plt.xticks(k_values)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# 테스트용
if __name__ == "__main__":
    pass