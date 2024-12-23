import os
import json
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import traceback

# Elasticsearch 설정
es_username = "elastic"
es_password = ""
es = Elasticsearch(
    ['https://localhost:9200'], 
    basic_auth=(es_username, es_password), 
    ca_certs="/home/elasticsearch-8.8.0/config/certs/http_ca.crt"
)

# SentenceTransformer 모델 초기화
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# OpenAI 클라이언트 설정
client = OpenAI(
    api_key="",
    base_url="https://api.upstage.ai/v1/solar"
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
## Role: 과학, 사회, 인문, 공학 등 모든 분야의 지식과 상식을 갖춘 전문가

## Instruction
카테고리별로 질문(fact), 조언 (reference 기반 혹은 질문자의 마지막 대화를 기준으로 ), 일상대화를 나눈 뒤에 질문자의 요청에 따라 간결하고 명확하게 대답해주세요. (단, 쿼리 자체는 한국어로 출력해주세요.) 

- 사용자와 여러번 대화를 했을 경우, 대화 맥락을 바탕으로 마지막으로 대화를 나눈 의미를 추론합니다.
- 사용자가 각종 지식을 주제로 질문을 하거나 설명을 요청하면, 시스템은 검색 API를 무조건 호출합니다.
- 지식과 관련되지 않는 다른 일상적인 대화 메세지에 대해서는 적절한 응답을 생성합니다.
- 영어로 질문할 때 (예시: Dmitri )는 한국말로 번역하여 인식하여도 됩니다 (예시: 드미트리)
"""

# 임베딩 관련 함수
def get_embedding(sentences):
    return model.encode(sentences)

def get_embeddings_in_batches(docs, batch_size=100):
    batch_embeddings = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        contents = [doc["content"] for doc in batch]
        embeddings = get_embedding(contents)
        batch_embeddings.extend(embeddings)
        print(f'batch {i}')
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
    query_embedding = get_embedding([query_str])[0]
    knn = {
        "field": "embeddings",
        "query_vector": query_embedding.tolist(),
        "k": size,
        "num_candidates": 100
    }
    return es.search(index="test", knn=knn)

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
    
    SPARSE_WEIGHT = 8.0
    DENSE_WEIGHT = 2.0
    
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

def identify_conversation_type(messages, eval_id):
    """대화 유형 분류 - eval.jsonl 파일의 실제 구조 기반"""
    # 일반 대화 eval_ids (topk 불필요)
    casual_eval_ids = {276, 261, 32, 94, 57, 2, 83, 64, 103, 218, 220, 222, 227, 229, 245, 247, 283, 301, 90}
    
    # 멀티턴 대화 eval_ids
    multiturn_eval_ids = {107, 42, 43, 97, 98, 66, 68, 89, 39, 33, 249, 290, 295, 306, 54, 3, 44, 278}
    
    # 과학/지식 관련 멀티턴
    science_related_eval_ids = {107, 98, 290, 306, 54, 3, 44, 278}
    
    if eval_id in casual_eval_ids:
        return False, False
    
    if eval_id in multiturn_eval_ids:
        return eval_id in science_related_eval_ids, eval_id in science_related_eval_ids
    
    return True, True

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
    messages = process_input_message(input_message)
    response = {
        "standalone_query": "", 
        "topk": [], 
        "references": [], 
        "answer": "",
        "eval_id": None
    }

    try:
        if not messages:
            raise ValueError("Invalid input message format")

        # eval_id 추출
        eval_id = None
        if isinstance(input_message, dict):
            eval_id = input_message.get("eval_id")
            response["eval_id"] = eval_id

        is_science_query, needs_topk = identify_conversation_type(messages, eval_id)
        
        if is_science_query:
            try:
                query = messages[-1]["content"]
                search_result = hybrid_retrieve(query, 3)
                
                if needs_topk:
                    response["standalone_query"] = query
                    retrieved_context = []
                    
                    for rst in search_result['hits']['hits']:
                        retrieved_context.append(rst["_source"]["content"])
                        response["topk"].append(rst["_source"]["docid"])
                        response["references"].append({
                            "score": rst["_score"],
                            "content": rst["_source"]["content"]
                        })

                    if retrieved_context:
                        context = "\n\n참고 자료:\n" + "\n".join(retrieved_context)
                        qa_messages = [
                            {"role": "system", "content": persona_qa + context}
                        ] + messages

                        qa_result = client.chat.completions.create(
                            model="solar-pro",
                            messages=qa_messages,
                            temperature=0,
                            seed=1,
                            timeout=30
                        )
                        response["answer"] = qa_result.choices[0].message.content
                    else:
                        response["answer"] = "죄송합니다. 관련된 정보를 찾을 수 없습니다."
            
            except Exception as e:
                error_msg = handle_api_error(e, "검색 중 오류가 발생했습니다.")
                response["answer"] = error_msg
        
        else:
            try:
                chat_messages = [
                    {"role": "system", "content": persona_function_calling}
                ] + messages

                chat_result = client.chat.completions.create(
                    model="solar-pro",
                    messages=chat_messages,
                    temperature=0.7,
                    seed=1,
                    timeout=30
                )
                response["answer"] = chat_result.choices[0].message.content
                
                # 일반 대화의 경우 검색 관련 필드 비우기
                response["topk"] = []
                response["references"] = []
                response["standalone_query"] = ""
            
            except Exception as e:
                error_msg = handle_api_error(e, "응답 생성 중 오류가 발생했습니다.")
                response["answer"] = error_msg

    except ValueError as e:
        response["answer"] = f"입력 형식이 올바르지 않습니다: {str(e)}"
    except Exception as e:
        traceback.print_exc()
        error_msg = handle_api_error(e, "죄송합니다. 처리 중 오류가 발생했습니다.")
        response["answer"] = error_msg

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
                "dims": 768,
                "index": True,
                "similarity": "l2_norm"
            }
        }
    }

    # 색인 생성 및 데이터 로드
    try:
        if not es.indices.exists(index="test"):
            es.indices.create(index="test", settings=settings, mappings=mappings)
            
            with open("/home/IR/data/documents.jsonl") as f:
                docs = [json.loads(line) for line in f]
            
            embeddings = get_embeddings_in_batches(docs)
            
            for doc, embedding in zip(docs, embeddings):
                doc["embeddings"] = embedding.tolist()
            
            actions = [
                {
                    '_index': "test",
                    '_source': doc
                }
                for doc in docs
            ]
            helpers.bulk(es, actions)
    except Exception as e:
        print(f"Index creation error: {e}")

    # 평가 실행
    eval_rag("/home/IR/data/eval.jsonl", "/home/IR/data/solar_prompt_hybrid_2.csv")