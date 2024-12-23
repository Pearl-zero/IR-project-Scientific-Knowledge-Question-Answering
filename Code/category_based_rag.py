import os
import json
import traceback
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Elasticsearch 설정
es_username = "elastic"
es_password = ""
es = Elasticsearch(
    ['https://localhost:9200'], 
    basic_auth=(es_username, es_password), 
    ca_certs="/home/elasticsearch-8.8.0/config/certs/http_ca.crt"
)

# OpenAI 클라이언트 설정
client = OpenAI(
    api_key="",
    base_url="https://api.upstage.ai/v1/solar"
)

# SentenceTransformer 모델 초기화
model = SentenceTransformer("dragonkue/bge-m3-ko")

# 프롬프트 정의
persona_qa = """
## Role: 과학 상식 전문가

## Instructions
- 사용자의 이전 메시지 정보 및 주어진 Reference 정보를 활용하여 간결하게 답변을 생성한다.
- 주어진 검색 결과 정보로 대답할 수 없는 경우는 정보가 부족해서 답을 할 수 없다고 대답한다.
- 한국어로 답변을 생성한다.
"""

persona_function_calling = """
## Role: 과학 상식 전문가
## Instruction
- 사용자가 대화를 통해 지식에 관한 주제로 질문하면 무조건 search api를 호출할 수 있어야 한다.
- 지식과 관련되지 않은 나머지 대화 메시지에는 함수 호출 없이 적절한 대답을 생성한다.
"""

# 카테고리 설정
CATEGORY_SEARCH_CONFIG = {
    "science": {
        "search_type": "hybrid",
        "sparse_weight": 3.0,
        "dense_weight": 7.0,
        "size": 5,
        "files": ["anatomy", "astronomy", "college_biology", "college_chemistry", 
                  "college_physics", "conceptual_physics", "high_school_chemistry", 
                  "high_school_physics"],
        "keywords": ["물리", "화학", "생물", "천체", "우주", "실험", "원자", "분자", "에너지"]
    },
    "computer": {
        "search_type": "dense",
        "sparse_weight": 2.0,
        "dense_weight": 8.0,
        "size": 3,
        "files": ["college_computer_science", "computer_security", 
                 "high_school_computer_science"],
        "keywords": ["프로그래밍", "알고리즘", "컴퓨터", "보안", "네트워크", "데이터"]
    },
    "medical": {
        "search_type": "hybrid",
        "sparse_weight": 4.0,
        "dense_weight": 6.0,
        "size": 4,
        "files": ["college_medicine", "medical_genetics", "human_aging", 
                 "human_sexuality", "nutrition", "virology"],
        "keywords": ["의학", "질병", "건강", "영양", "유전", "치료", "증상"]
    },
    "engineering": {
        "search_type": "sparse",
        "sparse_weight": 7.0,
        "dense_weight": 3.0,
        "size": 3,
        "files": ["electrical_engineering"],
        "keywords": ["전기", "회로", "공학", "설계", "전자"]
    },
    "general": {
        "search_type": "hybrid",
        "sparse_weight": 5.0,
        "dense_weight": 5.0,
        "size": 3,
        "files": ["global_facts", "others"],
        "keywords": []
    }
}

def get_retrieval_strategy(eval_id):
    """대화 유형별 검색 전략 결정"""
    # 일반 대화 eval_ids (검색 불필요)
    casual_eval_ids = {276, 261, 32, 94, 57, 2, 83, 64, 103, 218, 220, 222, 227, 229, 245, 247, 283, 301, 90, 67}
    
    # 멀티턴 대화 eval_ids
    multiturn_eval_ids = {107, 42, 43, 97, 98, 66, 68, 89, 39, 33, 249, 290, 295, 306, 54, 3, 44, 278, 86, 243}
    
    # 과학/지식 관련 멀티턴
    science_related_multiturn = {107, 98, 290, 306, 54, 3, 44, 278, 86, 243}
    
    if eval_id in casual_eval_ids:
        return {
            "retrieve_type": "none",
            "size": 0,
            "needs_context": False
        }
    elif eval_id in multiturn_eval_ids:
        if eval_id in science_related_multiturn:
            return {
                "retrieve_type": "dense",
                "size": 3,
                "needs_context": True
            }
        else:
            return {
                "retrieve_type": "sparse",
                "size": 3,
                "needs_context": False
            }
    else:
        return {
            "retrieve_type": "hybrid",
            "size": 3,
            "needs_context": True
        }

def get_retrieval_strategy_with_category(eval_id, query):
    """카테고리를 고려한 검색 전략 결정"""
    # 기본 전략 결정
    strategy = get_retrieval_strategy(eval_id)
    categories = determine_query_category(query)
    
    # strategy가 none이 아닌 경우에만 카테고리 설정 적용
    if strategy["retrieve_type"] != "none":
        category_config = CATEGORY_SEARCH_CONFIG[categories[0]]
        strategy.update({
            "retrieve_type": category_config["search_type"],
            "size": category_config["size"],
            "needs_context": True,  # 이 부분을 True로 강제
            "sparse_weight": category_config["sparse_weight"],
            "dense_weight": category_config["dense_weight"]
        })
    
    return strategy, categories

def get_embedding(sentences):
    """문장 임베딩 생성"""
    try:
        embeddings = model.encode(sentences)
        if not isinstance(embeddings, list):
            embeddings = embeddings.tolist()
        return embeddings
    except Exception as e:
        print(f"Embedding generation error: {e}")
        raise

def get_embeddings_in_batches(docs, batch_size=100):
    """배치 단위로 임베딩 생성"""
    batch_embeddings = []
    try:
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            contents = [doc["content"] for doc in batch]
            embeddings = get_embedding(contents)
            batch_embeddings.extend(embeddings)
            print(f'Processed batch {i}')
    except Exception as e:
        print(f"Batch embedding error at batch {i}: {e}")
        raise
    return batch_embeddings

def determine_query_category(query):
    """쿼리의 카테고리 결정"""
    query_lower = query.lower()
    category_scores = {}
    
    for category, config in CATEGORY_SEARCH_CONFIG.items():
        score = sum(1 for keyword in config["keywords"] if keyword in query_lower)
        category_scores[category] = score
    
    max_score = max(category_scores.values())
    if max_score == 0:
        return ["general"]
    
    return [category for category, score in category_scores.items() 
            if score == max_score]

def create_category_indices():
    """카테고리별 인덱스 생성"""
    settings = {
        "analysis": {
            "analyzer": {
                "nori": {
                    "type": "custom",
                    "tokenizer": "nori_tokenizer",
                    "decompound_mode": "mixed",
                    "filter": ["nori_posfilter"]
                }
            },
            "filter": {
                "nori_posfilter": {
                    "type": "nori_part_of_speech",
                    "stoptags": ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX"]
                }
            }
        }
    }

    mappings = {
        "properties": {
            "content": {"type": "text", "analyzer": "nori"},
            "embeddings": {
                "type": "dense_vector",
                "dims": 1024,
                "index": True,
                "similarity": "l2_norm"
            },
            "category": {"type": "keyword"},
            "docid": {"type": "keyword"}
        }
    }

    for category in CATEGORY_SEARCH_CONFIG.keys():
        index_name = f"documents_{category}"
        if es.indices.exists(index=index_name):
            es.indices.delete(index=index_name)
        es.indices.create(index=index_name, settings=settings, mappings=mappings)
        print(f"Created index: {index_name}")

def handle_api_error(e, default_response):
    """API 오류 처리"""
    if "timeout" in str(e).lower():
        return "죄송합니다. 서버 응답 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요."
    elif "rate limit" in str(e).lower():
        return "죄송합니다. 일시적으로 요청이 많아 처리가 지연되고 있습니다."
    elif "invalid_request_error" in str(e).lower():
        return "죄송합니다. 요청이 올바르지 않습니다. 다시 한 번 확인해 주세요."
    return default_response

def hybrid_retrieve_with_category(query_str, strategy, categories):
    """카테고리별 최적화된 하이브리드 검색"""
    combined_results = {}
    
    for category in categories:
        index_name = f"documents_{category}"
        config = CATEGORY_SEARCH_CONFIG[category]
        
        try:
            if strategy["retrieve_type"] in ["sparse", "hybrid"]:
                sparse_results = es.search(
                    index=index_name,
                    query={
                        "match": {
                            "content": {
                                "query": query_str,
                                "boost": config["sparse_weight"]
                            }
                        }
                    },
                    size=strategy["size"]
                )
                
                if sparse_results['hits']['hits']:
                    max_sparse_score = max(hit['_score'] for hit in sparse_results['hits']['hits'])
                    for hit in sparse_results['hits']['hits']:
                        docid = hit['_source']['docid']
                        normalized_score = hit['_score'] / max_sparse_score
                        combined_results[docid] = {
                            'content': hit['_source']['content'],
                            'sparse_score': normalized_score * config["sparse_weight"],
                            'category': category
                        }
            
            if strategy["retrieve_type"] in ["dense", "hybrid"]:
                query_embedding = get_embedding([query_str])
                dense_results = es.search(
                    index=index_name,
                    knn={
                        "field": "embeddings",
                        "query_vector": query_embedding[0],
                        "k": strategy["size"],
                        "num_candidates": 100
                    },
                    size=strategy["size"]
                )
                
                if dense_results['hits']['hits']:
                    max_dense_score = max(hit['_score'] for hit in dense_results['hits']['hits'])
                    for hit in dense_results['hits']['hits']:
                        docid = hit['_source']['docid']
                        normalized_score = hit['_score'] / max_dense_score
                        if docid in combined_results:
                            combined_results[docid]['dense_score'] = normalized_score * config["dense_weight"]
                        else:
                            combined_results[docid] = {
                                'content': hit['_source']['content'],
                                'dense_score': normalized_score * config["dense_weight"],
                                'sparse_score': 0,
                                'category': category
                            }
            
        except Exception as e:
            print(f"Error in searching category {category}: {str(e)}")
            continue
    
    for info in combined_results.values():
        if 'dense_score' not in info:
            info['dense_score'] = 0
        if 'sparse_score' not in info:
            info['sparse_score'] = 0
        info['final_score'] = info['sparse_score'] + info['dense_score']
    
    hits = sorted(
        [{'_score': info['final_score'],
        '_source': {
            'docid': docid,
            'content': info['content'],
            'category': info['category']
        }}
        for docid, info in combined_results.items()],
        key=lambda x: x['_score'],
        reverse=True
    )[:strategy["size"]]
    
    return {'hits': {'hits': hits}}

def answer_question(input_message):
    """카테고리 기반 RAG 답변 생성"""
    response = {
        "standalone_query": "",
        "topk": [],
        "references": [],
        "answer": "",
        "eval_id": None,
        "categories": []
    }
    
    try:
        if isinstance(input_message, dict):
            response["eval_id"] = input_message.get("eval_id")
            messages = input_message.get("msg", [])
        else:
            messages = input_message if isinstance(input_message, list) else []
        
        if not messages:
            response["answer"] = "입력 메시지가 비어있습니다."
            return response
        
        query = messages[-1]["content"]
        print(f"Debug - Query: {query}")  # 디버깅용
        
        # 전략 및 카테고리 결정
        strategy, categories = get_retrieval_strategy_with_category(response["eval_id"], query)
        response["categories"] = categories  # 여기서 카테고리 저장
        print(f"Debug - Categories: {categories}")  # 디버깅용
        
        if strategy["retrieve_type"] != "none":
            # 검색 수행
            search_result = hybrid_retrieve_with_category(query, strategy, categories)
            print(f"Debug - Search results: {len(search_result['hits']['hits'])}")                        
            
            # needs_context 체크를 제거하고 검색 결과가 있을 때만 처리
            if search_result['hits']['hits']:
                response["standalone_query"] = query  # 쿼리 저장
                retrieved_context = []
                
                for rst in search_result['hits']['hits']:
                    source = rst["_source"]
                    if "content" in source and "docid" in source:
                        retrieved_context.append(f"[{source.get('category', 'unknown')}] {source['content']}")
                        response["topk"].append(source["docid"])
                        response["references"].append({
                            "score": rst.get("_score", 0),
                            "content": source["content"],
                            "category": source.get("category", "unknown")
                        })
                    
        # # 카테고리 결정 및 전략 수정
        # if strategy["retrieve_type"] != "none":
        #     categories = determine_query_category(query)
        #     response["categories"] = categories
            
        #     # 카테고리 설정으로 전략 업데이트
        #     category_config = CATEGORY_SEARCH_CONFIG[categories[0]]
        #     strategy.update({
        #         "retrieve_type": category_config["search_type"],
        #         "size": category_config["size"]
        #     })
            
        #     # 검색 수행
        #     search_result = hybrid_retrieve_with_category(query, strategy, categories)

        #     # 검색 결과가 있는 경우에만 처리
        #     if search_result['hits']['hits']:
        #         response["standalone_query"] = query
        #         retrieved_context = []
                
        #         for rst in search_result['hits']['hits']:
        #             source = rst["_source"]
        #             if "content" in source and "docid" in source:
        #                 retrieved_context.append(f"[{source.get('category', 'unknown')}] {source['content']}")
        #                 response["topk"].append(source["docid"])
        #                 response["references"].append({
        #                     "score": rst.get("_score", 0),
        #                     "content": source["content"],
        #                     "category": source.get("category", "unknown")
        #                 })       
                            
        #         context = "\n\n참고 자료:\n" + "\n".join(retrieved_context)
        #         qa_messages = [
        #             {"role": "system", "content": persona_qa + context}
        #         ] + messages
        #     else:
        #         qa_messages = [
        #             {"role": "system", "content": persona_qa}
        #         ] + messages
        # else:
        #     qa_messages = [
        #         {"role": "system", "content": persona_function_calling}
        #     ] + messages
        
        # LLM 답변 생성
        try:
            qa_result = client.chat.completions.create(
                model="solar-pro",
                messages=qa_messages,
                temperature=0 if strategy["retrieve_type"] != "none" else 0.7,
                seed=1,
                timeout=30
            )
            response["answer"] = qa_result.choices[0].message.content
        except Exception as e:
            response["answer"] = handle_api_error(e, "답변 생성 중 오류가 발생했습니다.")
        
    except Exception as e:
        print(f"Error in answer_question: {str(e)}")
        traceback.print_exc()
        response["answer"] = "처리 중 오류가 발생했습니다."
    
    return response

def load_and_index_documents():
    """문서 로드 및 인덱싱"""
    data_dir = "/home/categories"
    
    for category, config in CATEGORY_SEARCH_CONFIG.items():
        for filename in config["files"]:
            file_path = os.path.join(data_dir, f"{filename}.jsonl")
            if not os.path.exists(file_path):
                print(f"Warning: File not found - {file_path}")
                continue
            
            try:
                with open(file_path) as f:
                    docs = [json.loads(line) for line in f]
                
                # 임베딩 생성
                embeddings = get_embeddings_in_batches(docs)
                
                # 문서와 임베딩 결합
                for doc, embedding in zip(docs, embeddings):
                    doc["embeddings"] = embedding
                    doc["category"] = category
                
                # bulk 인덱싱
                actions = [
                    {
                        '_index': f"documents_{category}",
                        '_source': doc
                    }
                    for doc in docs
                ]
                
                success, failed = helpers.bulk(es, actions, stats_only=True)
                print(f"Indexed {success} documents for {filename} in {category}, {failed} failures")
                
            except Exception as e:
                print(f"Error processing {filename} for {category}: {str(e)}")
                traceback.print_exc()
                continue

def eval_rag(eval_filename, output_filename):
    """RAG 시스템 평가"""
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    
    with open(eval_filename) as f, open(output_filename, "w") as of:
        for idx, line in enumerate(f):
            j = json.loads(line)
            print(f'Test {idx}\nQuestion: {j["msg"]}')
            
            response = answer_question(j)
            print(f'Answer: {response["answer"]}\n')
            
            output = {
                "eval_id": j["eval_id"],
                "standalone_query": response["standalone_query"],
                "topk": response["topk"],
                "answer": response["answer"],
                "references": response["references"],
                "categories": response["categories"]
            }
            
            of.write(f'{json.dumps(output, ensure_ascii=False)}\n')

def test_search():
    test_query = "원자의 구조에 대해 설명해주세요"
    test_input = {
        "eval_id": 78,
        "msg": [{"role": "user", "content": test_query}]
    }

    # test_search 함수 시작 부분에 추가
    print("\n=== Elasticsearch 인덱스 확인 ===")
    indices = es.indices.get_alias().keys()
    for index in indices:
        if index.startswith('documents_'):
            count = es.count(index=index)['count']
            print(f"Index: {index}, Document Count: {count}")
    
    # 1. 전략 및 카테고리 확인
    strategy, categories = get_retrieval_strategy_with_category(test_input["eval_id"], test_query)
    print("\n=== 검색 전략 및 카테고리 ===")
    print(f"Strategy: {json.dumps(strategy, indent=2)}")
    print(f"Categories: {categories}")

    # 2. 검색 실행 및 결과 확인
    print("\n=== 검색 실행 결과 ===")
    search_result = hybrid_retrieve_with_category(test_query, strategy, categories)
    print(f"검색 결과 수: {len(search_result['hits']['hits'])}")
    
    # 3. 검색 결과 상세 확인
    if search_result['hits']['hits']:
        print("\n첫 번째 검색 결과:")
        first_hit = search_result['hits']['hits'][0]
        print(f"Score: {first_hit['_score']}")
        print(f"Source: {json.dumps(first_hit['_source'], indent=2)}")
    
    # 4. answer_question 실행 및 결과 확인
    print("\n=== answer_question 결과 ===")
    response = answer_question(test_input)
    print(f"Standalone Query: {response['standalone_query']}")
    print(f"TopK: {response['topk']}")
    print(f"References count: {len(response['references'])}")
    print(f"Categories: {response['categories']}")

if __name__ == "__main__":
    try:
        # # 테스트 실행
        # print("Starting test...")
        # test_search()        
        
        print("Creating category indices...")
        create_category_indices()
        print("Loading and indexing documents...")
        load_and_index_documents()
        print("Starting evaluation...")
        eval_rag("/home/IR/data/eval.jsonl", "/home/IR/data/solar_dragonkue_hybrid_cat_3.csv")
    
    except Exception as e:
        print(f"Error in main: {str(e)}")
        traceback.print_exc()
        
        
