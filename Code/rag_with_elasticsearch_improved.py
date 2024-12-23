import os
import json
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

# Sentence Transformer 모델 초기화 (한국어 임베딩 생성 가능한 어떤 모델도 가능)
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# SetntenceTransformer를 이용하여 임베딩 생성
def get_embedding(sentences):
    return model.encode(sentences)


# 주어진 문서의 리스트에서 배치 단위로 임베딩 생성
def get_embeddings_in_batches(docs, batch_size=100):
    batch_embeddings = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        contents = [doc["content"] for doc in batch]
        embeddings = get_embedding(contents)
        batch_embeddings.extend(embeddings)
        print(f'batch {i}')
    return batch_embeddings


# 새로운 index 생성
def create_es_index(index, settings, mappings):
    # 인덱스가 이미 존재하는지 확인
    if es.indices.exists(index=index):
        # 인덱스가 이미 존재하면 설정을 새로운 것으로 갱신하기 위해 삭제
        es.indices.delete(index=index)
    # 지정된 설정으로 새로운 인덱스 생성
    es.indices.create(index=index, settings=settings, mappings=mappings)


# 지정된 인덱스 삭제
def delete_es_index(index):
    es.indices.delete(index=index)


# Elasticsearch 헬퍼 함수를 사용하여 대량 인덱싱 수행
def bulk_add(index, docs):
    # 대량 인덱싱 작업을 준비
    actions = [
        {
            '_index': index,
            '_source': doc
        }
        for doc in docs
    ]
    return helpers.bulk(es, actions)


# 역색인을 이용한 검색
def sparse_retrieve(query_str, size):
    query = {
        "match": {
            "content": {
                "query": query_str
            }
        }
    }
    return es.search(index="test", query=query, size=size, sort="_score")


# Vector 유사도를 이용한 검색
def dense_retrieve(query_str, size):
    # 벡터 유사도 검색에 사용할 쿼리 임베딩 가져오기
    query_embedding = get_embedding([query_str])[0]

    # KNN을 사용한 벡터 유사성 검색을 위한 매개변수 설정
    knn = {
        "field": "embeddings",
        "query_vector": query_embedding.tolist(),
        "k": size,
        "num_candidates": 100
    }

    # 지정된 인덱스에서 벡터 유사도 검색 수행
    return es.search(index="test", knn=knn)

def improved_hybrid_retrieve(query_str, size):
    # 1. Query Expansion: 쿼리 확장
    expanded_query = query_expansion(query_str)
    
    # 2. 각각의 검색 수행
    sparse_results = sparse_retrieve(expanded_query, size * 2)  # 더 많은 후보 검색
    dense_results = dense_retrieve(query_str, size * 2)
    
    # 3. 결과 통합을 위한 딕셔너리
    combined_results = {}
    
    # 4. Sparse 결과 처리 - BM25 스코어 정규화
    sparse_scores = [hit['_score'] for hit in sparse_results['hits']['hits']]
    max_sparse_score = max(sparse_scores)
    min_sparse_score = min(sparse_scores)
    
    for hit in sparse_results['hits']['hits']:
        docid = hit['_source']['docid']
        # Min-Max 정규화 적용
        if max_sparse_score != min_sparse_score:
            normalized_score = (hit['_score'] - min_sparse_score) / (max_sparse_score - min_sparse_score)
        else:
            normalized_score = 1.0
            
        combined_results[docid] = {
            'content': hit['_source']['content'],
            'sparse_score': normalized_score,
            'position_score': 0
        }
    
    # 5. Dense 결과 처리 - 코사인 유사도 스코어 정규화
    dense_scores = [hit['_score'] for hit in dense_results['hits']['hits']]
    max_dense_score = max(dense_scores)
    min_dense_score = min(dense_scores)
    
    for idx, hit in enumerate(dense_results['hits']['hits']):
        docid = hit['_source']['docid']
        if max_dense_score != min_dense_score:
            normalized_score = (hit['_score'] - min_dense_score) / (max_dense_score - min_dense_score)
        else:
            normalized_score = 1.0
            
        if docid in combined_results:
            combined_results[docid]['dense_score'] = normalized_score
        else:
            combined_results[docid] = {
                'content': hit['_source']['content'],
                'dense_score': normalized_score,
                'sparse_score': 0,
                'position_score': 0
            }
    
    # 6. Position-aware 스코링: 검색 결과 순위 반영
    for idx, hit in enumerate(sparse_results['hits']['hits']):
        docid = hit['_source']['docid']
        position_score = 1.0 / (idx + 1)
        combined_results[docid]['position_score'] = position_score
    
    # 7. 최종 스코어 계산 - 가중치 조정 및 position 점수 반영
    SPARSE_WEIGHT = 0.5  # BM25
    DENSE_WEIGHT = 0.3   # Dense Embedding
    POSITION_WEIGHT = 0.2  # Position-aware score
    
    for info in combined_results.values():
        if 'dense_score' not in info:
            info['dense_score'] = 0
        
        info['final_score'] = (
            SPARSE_WEIGHT * info['sparse_score'] +
            DENSE_WEIGHT * info['dense_score'] +
            POSITION_WEIGHT * info['position_score']
        )
    
    # 8. 최종 결과 생성
    hits = sorted(
        [{'_score': info['final_score'],
        '_source': {'docid': docid, 'content': info['content']}}
        for docid, info in combined_results.items()],
        key=lambda x: x['_score'],
        reverse=True
    )[:size]
    
    return {'hits': {'hits': hits}}

def query_expansion(query_str):
    """쿼리 확장을 위한 함수"""
    # 1. 형태소 분석
    tokenized = es.indices.analyze(
        index="test",
        body={
            "analyzer": "nori",
            "text": query_str
        }
    )
    
    # 2. 핵심 키워드 추출
    keywords = [token['token'] for token in tokenized['tokens']]
    
    # 3. 동의어 확장 (예시 - 실제로는 동의어 사전이나 워드넷 등을 활용)
    synonyms = {
        "이유": ["원인", "까닭"],
        "방법": ["방식", "절차"],
        "차이": ["차이점", "구별"],
        # 필요한 동의어들 추가
    }
        
    # 4. 확장된 쿼리 생성
    expanded_terms = []
    for keyword in keywords:
        expanded_terms.append(keyword)
        if keyword in synonyms:
            expanded_terms.extend(synonyms[keyword])
    
    # 5. 가중치를 부여한 쿼리 생성
    expanded_query = " ".join(expanded_terms)
    return expanded_query

es_username = "elastic"
es_password = ""

# Elasticsearch client 생성
es = Elasticsearch(['https://localhost:9200'], basic_auth=(es_username, es_password), ca_certs="/home/elasticsearch-8.8.0/config/certs/http_ca.crt")

# Elasticsearch client 정보 확인
print(es.info())

# 색인을 위한 setting 설정
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
                # 어미, 조사, 구분자, 줄임표, 지정사, 보조 용언 등
                "stoptags": ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX"]
            }
        }
    }
}

# 색인을 위한 mapping 설정 (역색인 필드, 임베딩 필드 모두 설정)
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

# settings, mappings 설정된 내용으로 'test' 인덱스 생성
create_es_index("test", settings, mappings)

# 문서의 content 필드에 대한 임베딩 생성
index_docs = []
with open("/home/IR/data/documents.jsonl") as f:
    docs = [json.loads(line) for line in f]
embeddings = get_embeddings_in_batches(docs)
                
# 생성한 임베딩을 색인할 필드로 추가
for doc, embedding in zip(docs, embeddings):
    doc["embeddings"] = embedding.tolist()
    index_docs.append(doc)

# 'test' 인덱스에 대량 문서 추가
ret = bulk_add("test", index_docs)

# 색인이 잘 되었는지 확인 (색인된 총 문서수가 출력되어야 함)
print(ret)

test_query = "금성이 다른 행성들보다 밝게 보이는 이유는 무엇인가요?"

# 역색인을 사용하는 검색 예제
search_result_retrieve = sparse_retrieve(test_query, 3)

# 결과 출력 테스트
for rst in search_result_retrieve['hits']['hits']:
    print('score:', rst['_score'], 'source:', rst['_source']["content"])

# Vector 유사도 사용한 검색 예제
search_result_retrieve = dense_retrieve(test_query, 3)

# 결과 출력 테스트
for rst in search_result_retrieve['hits']['hits']:
    print('score:', rst['_score'], 'source:', rst['_source']["content"])



# 아래부터는 실제 RAG를 구현하는 코드입니다.
from openai import OpenAI
import traceback

# OpenAI API 키를 환경변수에 설정
os.environ["OPENAI_API_KEY"] = ""

client = OpenAI(
    api_key="",
    base_url="https://api.upstage.ai/v1/solar"
)
# 사용할 모델을 설정
llm_model = "solar-pro"

# RAG 구현에 필요한 Question Answering을 위한 LLM  프롬프트
persona_qa = """
## Role: 과학 상식 전문가

## Instructions
- 사용자의 이전 메시지 정보 및 주어진 Reference 정보를 활용하여 간결하게 답변을 생성한다.
- 주어진 검색 결과 정보로 대답할 수 없는 경우는 정보가 부족해서 답을 할 수 없다고 대답한다.
- 한국어로 답변을 생성한다.
"""

# RAG 구현에 필요한 질의 분석 및 검색 이외의 일반 질의 대응을 위한 LLM 프롬프트
persona_function_calling = """
## Role: 과학 상식 전문가

## Instruction
- 사용자가 대화를 통해 과학 지식에 관한 주제로 질문하면 search api를 호출할 수 있어야 한다.
- 과학 상식과 관련되지 않은 나머지 대화 메시지에는 적절한 대답을 생성한다.
"""

# Function calling에 사용할 함수 정의
tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "search relevant documents",
            "parameters": {
                "properties": {
                    "standalone_query": {
                        "type": "string",
                        "description": "Final query suitable for use in search from the user messages history."
                    }
                },
                "required": ["standalone_query"],
                "type": "object"
            }
        }
    },
]

# LLM과 검색엔진을 활용한 RAG 구현
def answer_question(messages):
    # 함수 출력 초기화
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

    # 마지막 사용자 메시지를 검색 쿼리로 사용
    for msg in reversed(messages):
        if msg["role"] == "user":
            standalone_query = msg["content"]
            break

    try:
        # 개선된 Hybrid 검색 실행
        search_result = improved_hybrid_retrieve(standalone_query, 3)

        # 검색 결과 처리
        response["standalone_query"] = standalone_query
        retrieved_context = []
        for rst in search_result['hits']['hits']:
            retrieved_context.append(rst["_source"]["content"])
            response["topk"].append(rst["_source"]["docid"])
            response["references"].append({
                "score": rst["_score"],
                "content": rst["_source"]["content"]
            })

        # 프롬프트 구성
        context = "\n\n참고 자료:\n" + "\n".join(retrieved_context)
        system_message = persona_qa + context
        
        # LLM으로 답변 생성
        messages_for_completion = [
            {"role": "system", "content": system_message}
        ] + messages

        result = client.chat.completions.create(
            model="solar-pro",
            messages=messages_for_completion,
            temperature=0,
            seed=1,
            timeout=30
        )
        response["answer"] = result.choices[0].message.content

    except Exception as e:
        traceback.print_exc()
        return response

    return response

# 평가를 위한 파일을 읽어서 각 평가 데이터에 대해서 결과 추출후 파일에 저장
def eval_rag(eval_filename, output_filename):
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(eval_filename) as f, open(output_filename, "w") as of:
        idx = 0
        for line in f:
            j = json.loads(line)
            print(f'Test {idx}\nQuestion: {j["msg"]}')
            response = answer_question(j["msg"])
            print(f'Answer: {response["answer"]}\n')

            # 대회 score 계산은 topk 정보를 사용, answer 정보는 LLM을 통한 자동평가시 활용
            output = {"eval_id": j["eval_id"], "standalone_query": response["standalone_query"], "topk": response["topk"], "answer": response["answer"], "references": response["references"]}
            of.write(f'{json.dumps(output, ensure_ascii=False)}\n')
            idx += 1

# 평가 데이터에 대해서 결과 생성 - 파일 포맷은 jsonl이지만 파일명은 csv 사용
eval_rag("/home/IR/data/eval.jsonl", "/home/IR/data/improved_hybrid.csv")

