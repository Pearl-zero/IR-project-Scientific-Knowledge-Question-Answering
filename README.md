# Scientific Knowledge Question Answering
## Team

| <img src="https://avatars.githubusercontent.com/u/102230809?s=60&v=4" width="200"> | <img src="https://avatars.githubusercontent.com/u/17960812?s=60&v=4" width="200"> | <img src="https://avatars.githubusercontent.com/u/45289805?s=60&v=4" width="200"> | <img src="https://avatars.githubusercontent.com/u/51690185?s=60&v=4" width="200"> | <img src="https://avatars.githubusercontent.com/u/173867600?s=60&v=4" width="200"> |
|:---:|:---:|:---:|:---:|:---:|
| [김동규](https://github.com/Lumiere001) | [김지환](https://github.com/jihwanK) | [이주하](https://github.com/jl3725) | [한성범](https://github.com/winterbeom) | [진주영](https://github.com/Pearl-zero) |
| LLM Model<br>Selection<br>Query Routing<br>Team Leader | Embedding<br>Model Testing<br>Evaluation | Prompt<br>Engineering<br>Query Expansion<br>Chunking | Dense Retrieve<br>Reranking<br>Evaluation | Hybrid Retrieve<br>Reranking<br>Testing set<br>EDA |

## 0. Overview

### Requirements
- sentence_transformers==2.2.2
- elasticsearch==8.8.0
- openai==1.7.2

## 1. Competiton Info

### Overview

- 질문과 이전 대화 히스토리를 보고 참고할 문서를 검색엔진에서 추출 후 이를 활용하여 질문에 적합한 대답을 생성하는 태스크

### Timeline

- Dec 16, 2024 - Start Date
- Dec 19, 2024 - Final submission deadline

## 2. Components

### Directory

- _Insert your directory structure_

e.g.
```
 |-Code
 | |-category_based_rag.py
 | |-EDA.ipynb
 | |-hybrid_cat_rerank.py
 | |-rag_with_elasticsearch_hybrid.py
 | |-rag_with_elasticsearch_hybrid_cat.py
 | |-rag_with_elasticsearch_improved.py
 | |-rag_with_elasticsearch_prompt.py
 |-IR_경진대회-1조-발표자료.pdf
 |-Test set
 | |-compare_modified.ipynb
 | |-compare_submission.ipynb
 |-Private LV.png
 |-requirements.txt
 |-README.md
```

## 3. Data descrption

### Dataset overview

- 과학 상식 정보를 담고 있는 순수 색인 대상 문서 4200여개
- doc_id'에는 uuid로 문서별 id가 부여되어 있고 'src'는 출처를 나타내는 필드입니다. 그리고 실제 RAG에서 레퍼런스로 참고할 지식 정보는 'content' 필드에 저장되어 있음.
- MMLU, ARC 데이터를 기반으로 생성

## 4. Modeling

### Model descrition

- embedding model : dragonkue/bge-m3-ko
- LLM model : gpt 3.5-turbo

### Modeling Process

- reranking, hybrid search 진행.

## 5. Result

### Leader Board

<p align="center">
  <img src="Private LV.png" alt="Private LV" width="600">
</p>

### Presentation

[발표 자료](IR_경진대회-1조-발표자료.pdf)

## etc

### Reference

- hugging face etc.
