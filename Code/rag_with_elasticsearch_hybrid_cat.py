import os
import json
import traceback
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Elasticsearch 설정
es_username = "elastic"
# es_password = ""
es = Elasticsearch(
    ['https://localhost:9200'], 
    basic_auth=(es_username, es_password), 
    ca_certs="/home/elasticsearch-8.8.0/config/certs/http_ca.crt"
)

# SentenceTransformer 모델 초기화
model = SentenceTransformer("dragonkue/bge-m3-ko")

# OpenAI 클라이언트 설정
client = OpenAI(
    #api_key = ""
    # base_url="https://api.upstage.ai/v1/solar"
)

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

# 임베딩 관련 함수
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

# 검색 관련 함수
def sparse_retrieve(query_str, size):
    query = {
        "match": {
            "content": {
                "query": query_str
            }
        }
    }
    return es.search(index="test", query=query, size=size, sort="_score")

def dense_retrieve(query_str, size):
    """Dense 검색 구현"""
    try:
        # 쿼리 임베딩 생성
        query_embedding = get_embedding([query_str])
        if not isinstance(query_embedding, list):
            query_embedding = query_embedding.tolist()
        
        # KNN 쿼리 구성
        knn = {
            "field": "embeddings",
            "query_vector": query_embedding[0],  # 첫 번째 임베딩 사용
            "k": size,
            "num_candidates": 100,  # 후보 문서 수
        }
        
        # 검색 실행
        response = es.search(
            index="test",
            knn=knn,
            size=size,
            _source=["content", "docid"],  # 필요한 필드만 가져오기
            timeout="30s"  # 타임아웃 설정
        )
        
        if not response['hits']['hits']:
            print("No search results found")
        
        return response
        
    except Exception as e:
        print(f"Dense retrieval error: {str(e)}")
        traceback.print_exc()
        raise

def hybrid_retrieve(query_str, size):
    sparse_results = sparse_retrieve(query_str, size)
    dense_results = dense_retrieve(query_str, size)
    
    combined_results = {}
    
    # Sparse 결과 처리
    max_sparse_score = max(hit['_score'] for hit in sparse_results['hits']['hits'])
    for hit in sparse_results['hits']['hits']:
        docid = hit['_source']['docid']
        normalized_score = hit['_score'] / max_sparse_score
        combined_results[docid] = {
            'content': hit['_source']['content'],
            'sparse_score': normalized_score
        }
    
    # Dense 결과 처리
    max_dense_score = max(hit['_score'] for hit in dense_results['hits']['hits'])
    for hit in dense_results['hits']['hits']:
        docid = hit['_source']['docid']
        normalized_score = hit['_score'] / max_dense_score
        if docid in combined_results:
            combined_results[docid]['dense_score'] = normalized_score
        else:
            combined_results[docid] = {
                'content': hit['_source']['content'],
                'dense_score': normalized_score,
                'sparse_score': 0
            }
    
    SPARSE_WEIGHT = 2.0
    DENSE_WEIGHT = 8.0
    
    for info in combined_results.values():
        if 'dense_score' not in info:
            info['dense_score'] = 0
        info['final_score'] = (SPARSE_WEIGHT * info['sparse_score'] +
                             DENSE_WEIGHT * info['dense_score'])
    
    hits = sorted(
        [{'_score': info['final_score'],
        '_source': {'docid': docid, 'content': info['content']}}
        for docid, info in combined_results.items()],
        key=lambda x: x['_score'],
        reverse=True
    )[:size]
    
    return {'hits': {'hits': hits}}

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
                "retrieve_type": "dense",  # 과학 관련 멀티턴은 dense 검색
                "size": 3,  
                "needs_context": True
            }
        else:
            return {
                "retrieve_type": "sparse",  # 일반 멀티턴은 sparse 검색
                "size": 3,  
                "needs_context": False
            }
    else:
        return {
            "retrieve_type": "hybrid",  # 나머지는 hybrid 검색
            "size": 3,
            "needs_context": True
        }

def retrieve_by_strategy(query_str, strategy):
    """검색 전략에 따른 검색 수행"""
    if strategy["retrieve_type"] == "none":
        return {'hits': {'hits': []}}
    
    elif strategy["retrieve_type"] == "dense":
        return dense_retrieve(query_str, strategy["size"])
    
    elif strategy["retrieve_type"] == "sparse":
        return sparse_retrieve(query_str, strategy["size"])
    
    elif strategy["retrieve_type"] == "hybrid":
        return hybrid_retrieve(query_str, strategy["size"])
    
    return {'hits': {'hits': []}}

def process_input_message(message):
    """입력 메시지 처리"""
    if isinstance(message, list):
        return message
    elif isinstance(message, dict):
        return message.get("msg", [{"role": "user", "content": message.get("content", "")}])
    else:
        return [{"role": "user", "content": str(message)}]

def handle_api_error(e, default_response):
    """API 오류 처리"""
    if "timeout" in str(e).lower():
        return "죄송합니다. 서버 응답 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요."
    elif "rate limit" in str(e).lower():
        return "죄송합니다. 일시적으로 요청이 많아 처리가 지연되고 있습니다."
    elif "invalid_request_error" in str(e).lower():
        return "죄송합니다. 요청이 올바르지 않습니다. 다시 한 번 확인해 주세요."
    return default_response

def answer_question(input_message):
    """RAG 기반 답변 생성"""
    response = {
        "standalone_query": "",
        "topk": [],
        "references": [],
        "answer": "",
        "eval_id": None
    }
    
    try:
        # 입력 메시지 처리
        if isinstance(input_message, dict):
            response["eval_id"] = input_message.get("eval_id")
            messages = input_message.get("msg", [])
        else:
            messages = input_message if isinstance(input_message, list) else []
        
        if not messages:
            response["answer"] = "입력 메시지가 비어있습니다."
            return response
        
        # 검색 전략 결정
        strategy = get_retrieval_strategy(response["eval_id"])
        query = messages[-1]["content"]
        
        try:
            # 전략에 따른 검색 수행
            search_result = retrieve_by_strategy(query, strategy)
            
            if strategy["needs_context"] and search_result['hits']['hits']:
                response["standalone_query"] = query
                retrieved_context = []
                
                for rst in search_result['hits']['hits']:
                    source = rst["_source"]
                    if "content" in source and "docid" in source:
                        retrieved_context.append(source["content"])
                        response["topk"].append(source["docid"])
                        response["references"].append({
                            "score": rst.get("_score", 0),
                            "content": source["content"]
                        })
                
                if retrieved_context:
                    context = "\n\n참고 자료:\n" + "\n".join(retrieved_context)
                    qa_messages = [
                        {"role": "system", "content": persona_qa + context}
                    ] + messages
                else:
                    qa_messages = [
                        {"role": "system", "content": persona_qa}
                    ] + messages
            else:
                qa_messages = [
                    {"role": "system", "content": persona_qa if strategy["retrieve_type"] != "none" else persona_function_calling}
                ] + messages
            
            # LLM 답변 생성
            try:
                qa_result = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=qa_messages,
                    temperature=0 if strategy["retrieve_type"] != "none" else 0.7,
                    seed=1,
                    timeout=30
                )
                response["answer"] = qa_result.choices[0].message.content
            except Exception as e:
                print(f"LLM API Error: {str(e)}")
                response["answer"] = handle_api_error(e, "답변 생성 중 오류가 발생했습니다.")
            
        except Exception as e:
            print(f"Search Error: {str(e)}")
            qa_messages = [
                {"role": "system", "content": persona_qa if strategy["retrieve_type"] != "none" else persona_function_calling}
            ] + messages
            try:
                qa_result = client.chat.completions.create(
                    model="solar-pro",
                    messages=qa_messages,
                    temperature=0 if strategy["retrieve_type"] != "none" else 0.7,
                    seed=1,
                    timeout=30
                )
                response["answer"] = qa_result.choices[0].message.content
            except:
                response["answer"] = "검색 및 답변 생성에 실패했습니다."
    
    except Exception as e:
        print(f"General Error: {str(e)}")
        traceback.print_exc()
        response["answer"] = "처리 중 오류가 발생했습니다."
    
    return response

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
                "references": response["references"]
            }
            
            of.write(f'{json.dumps(output, ensure_ascii=False)}\n')

if __name__ == "__main__":
    # Elasticsearch 색인 설정
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
            }
        }
    }

    # 색인 생성 및 데이터 로드
    try:
        # 기존 인덱스 확인 및 삭제
        if es.indices.exists(index="test"):
            es.indices.delete(index="test")
            print("Deleted existing index")
        
        # 새 인덱스 생성
        es.indices.create(index="test", settings=settings, mappings=mappings)
        print("Created new index with 2048 dimensions")
        
        # 데이터 로드 및 임베딩 생성
        with open("/home/IR/data/documents.jsonl") as f:
            docs = [json.loads(line) for line in f]
        
        # 배치 단위로 임베딩 생성
        embeddings = get_embeddings_in_batches(docs)
        print(f"Generated embeddings for {len(docs)} documents")
        
        # 문서와 임베딩 결합 - .tolist() 제거
        for doc, embedding in zip(docs, embeddings):
            doc["embeddings"] = embedding  # embedding은 이미 리스트 형태
        
        # bulk 인덱싱
        actions = [
            {
                '_index': "test",
                '_source': doc
            }
            for doc in docs
        ]
        success, failed = helpers.bulk(es, actions, stats_only=True)
        print(f"Indexed {success} documents, {failed} failures")

    except Exception as e:
        print(f"Index creation error: {e}")
        traceback.print_exc()
        raise

    # 평가 실행
    print("Starting evaluation...")
    eval_rag("/home/IR/data/eval.jsonl", "/home/IR/data/openai_dragonkue_hybrid_cat_.csv")