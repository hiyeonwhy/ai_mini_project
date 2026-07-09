# AI Shopping Assistant

패션 이미지를 업로드하면 AI가 이미지 속 의류 아이템을 탐지하고, 색상·카테고리·이미지 유사도·예산 조건을 기반으로 관련 상품을 추천하는 AI 쇼핑 어시스턴트 프로젝트입니다.

본 프로젝트는 YOLO 기반 패션 아이템 탐지, 색상 인식, 네이버 쇼핑 API, ChromaDB 벡터 검색, CLIP 이미지 유사도 검색, LLM 기반 대화형 추천 기능을 통합하여 사용자가 상품명을 직접 입력하지 않아도 이미지와 자연어 대화만으로 원하는 패션 상품을 탐색할 수 있도록 구현했습니다.

---

## 1. 프로젝트 개요

사용자가 패션 이미지를 업로드하면 다음과 같은 흐름으로 상품을 추천합니다.

```text
이미지 업로드
→ YOLO로 패션 아이템 탐지
→ 탐지된 의류 영역에서 대표 색상 추출
→ 색상 + 의류 카테고리 기반 검색어 생성
→ 네이버 쇼핑 API 상품 검색
→ 상품 정보를 ChromaDB에 저장
→ 상품 이미지를 CLIP 벡터로 변환
→ 업로드 이미지와 시각적으로 유사한 상품 검색
→ 텍스트 검색 결과와 이미지 유사도 검색 결과 병합
→ Gradio UI에서 추천 상품 표시
→ 사용자의 예산 조건에 따라 가격 필터링
```

---

## 2. 주요 기능

### 2.1 패션 아이템 탐지

DeepFashion2 기반 YOLOv8 모델을 사용하여 업로드된 이미지에서 패션 아이템을 탐지합니다.

탐지된 객체는 이미지 위에 바운딩 박스로 표시되며, 탐지된 클래스명을 기반으로 상품 검색어를 생성합니다.

탐지 가능한 항목 예시는 다음과 같습니다.

```text
상의
셔츠
아우터
바지
스커트
원피스
조끼
```

---

### 2.2 옷 색상 인식 기능

YOLO가 탐지한 바운딩 박스 영역을 기준으로 의류 영역을 crop한 뒤, 해당 영역의 대표 색상을 추출합니다.

구현 방식은 다음과 같습니다.

```text
YOLO bbox 좌표 추출
→ 의류 영역 crop
→ RGB 이미지를 HSV 색공간으로 변환
→ 채도와 명도 기준으로 색상 분류
→ 한국어 색상명으로 변환
→ 색상 + 의류명 검색어 생성
```

지원 색상 예시는 다음과 같습니다.

```text
흰색
검은색
회색
빨간색
주황색
노란색
초록색
하늘색
파란색
보라색
분홍색
갈색
```

예시 결과:

```text
YOLO 탐지 결과: skirt
색상 인식 결과: 회색 스커트
검색어: 회색 스커트
```

---

### 2.3 의류 클래스명 한국어 변환

YOLO 모델에서 반환하는 영어 클래스명을 네이버 쇼핑 검색에 적합한 한국어 키워드로 변환합니다.

예시:

```text
skirt → 스커트
trousers → 바지
short_sleeved_shirt → 반팔 셔츠
long_sleeved_shirt → 긴팔 셔츠
long_sleeved_outwear → 긴팔 아우터
short sleeve dress → 반팔 원피스
```

이를 통해 검색어 품질을 높이고, 네이버 쇼핑 API에서 더 적절한 상품 결과를 가져올 수 있도록 했습니다.

---

### 2.4 네이버 쇼핑 API 기반 상품 검색

탐지된 의류명과 색상 정보를 조합하여 네이버 쇼핑 검색 API로 관련 상품을 검색합니다.

예시 검색어:

```text
흰색 반팔 원피스
회색 스커트
파란색 긴팔 아우터
파란색 바지
```

검색 결과에서 사용하는 정보는 다음과 같습니다.

```text
상품명
상품 이미지
최저가
쇼핑몰명
구매 링크
브랜드
카테고리
상품 ID
```

---

### 2.5 ChromaDB 기반 텍스트 벡터 검색

네이버 쇼핑 API로 검색된 상품 정보를 텍스트 형태로 변환한 뒤 ChromaDB에 저장합니다.

저장되는 텍스트 정보 예시는 다음과 같습니다.

```text
상품명
쇼핑몰명
가격
브랜드
제조사
카테고리
```

검색 API 호출 횟수가 제한에 도달하거나, 기존에 저장된 상품 데이터에서 검색이 필요한 경우 ChromaDB 기반 유사도 검색을 수행합니다.

사용한 임베딩 모델:

```text
jhgan/ko-sroberta-multitask
```

---

### 2.6 CLIP 이미지 유사도 검색

실습 과제 3에서 구현한 기능입니다.

기존 텍스트 검색뿐 아니라, 상품 이미지를 CLIP 벡터로 변환하여 사용자가 업로드한 이미지와 시각적으로 유사한 상품을 검색합니다.

구현 흐름은 다음과 같습니다.

```text
네이버 쇼핑 API 검색 결과 상품 이미지 다운로드
→ 상품 이미지를 CLIP 벡터로 변환
→ ChromaDB 이미지 전용 Collection에 저장
→ 사용자 업로드 이미지를 CLIP 벡터로 변환
→ 이미지 벡터 간 cosine similarity 기반 유사도 검색
→ 텍스트 검색 결과와 CLIP 이미지 검색 결과 병합
→ 중복 상품 제거 후 추천 결과 출력
```

사용한 모델:

```text
openai/clip-vit-base-patch32
```

중요 구현 포인트:

텍스트 임베딩과 CLIP 이미지 임베딩은 벡터 차원이 다르기 때문에 동일한 ChromaDB Collection에 저장하지 않고, 이미지 검색 전용 Collection을 별도로 생성했습니다.

```text
텍스트 상품 검색: KO-SRoBERTa 기반 ChromaDB
이미지 유사도 검색: CLIP 기반 ChromaDB Collection
```

---

### 2.7 예산 맞춤 가격 필터링

사용자의 자연어 예산 요청을 분석하여 상품 결과를 가격 조건에 맞게 필터링합니다.

지원하는 요청 예시는 다음과 같습니다.

```text
5만원 이하로 추천해줘
10만원 안으로 보여줘
30000원 이하
20000원 이상
3만원 초과
더 저렴한 걸로 보여줘
싼 순으로 보여줘
가격 낮은 순으로 보여줘
최저가 위주로 추천해줘
```

구현 기능:

```text
정규표현식 기반 가격 숫자 추출
만원 / 원 단위 처리
이하 / 미만 / 이상 / 초과 조건 처리
상대적 가격 요청 처리
상품 가격 기준 필터링
가격 낮은 순 정렬
```

예시:

```text
사용자 입력:
5만원 이하로 추천해줘

응답:
50,000원 이하 상품을 가격 낮은 순으로 정렬했습니다.
```

```text
사용자 입력:
20000원 이상

응답:
20,000원 이상 상품을 가격 낮은 순으로 정렬했습니다.
```

---

### 2.8 LLM 기반 대화형 쇼핑 어시스턴트

LLM을 사용하여 사용자의 질문에 대화형으로 응답합니다.

사용자가 예산, 스타일, 브랜드 선호 등을 입력하면 현재 검색된 상품 정보를 바탕으로 답변을 생성합니다.

사용한 모델:

```text
Bllossom/llama-3.2-Korean-Bllossom-3B
```

대화 히스토리는 토큰 수를 관리하면서 유지됩니다.

구현 내용:

```text
대화 히스토리 관리
TokenManager 기반 토큰 수 제한
검색 결과를 LLM context로 전달
추천 상품 요약 응답 생성
```

---

## 3. 기술 스택

### Language

```text
Python
```

### Computer Vision

```text
YOLOv8
Ultralytics
DeepFashion2 사전학습 모델
OpenCV
Pillow
NumPy
```

### Image Similarity

```text
CLIP
openai/clip-vit-base-patch32
Transformers
PyTorch
```

### LLM / RAG

```text
LangChain
HuggingFacePipeline
ConversationBufferMemory
ConversationalRetrievalChain
ChromaDB
KO-SRoBERTa
Bllossom Llama 3.2 Korean 3B
```

### API / Data

```text
Naver Shopping Search API
Requests
Pandas
JSON
CSV
BeautifulSoup
Selenium
```

### UI

```text
Gradio
```

---

## 4. 프로젝트 구조

```text
ai-shopping-assistant/
├── app.py                 # 메인 Gradio 애플리케이션
├── config.py              # API 키, 모델 경로, 검색 설정
├── utils.py               # 색상 인식, 예산 파싱, 토큰 관리 등 유틸 함수
├── crawl_products.py      # 네이버 쇼핑 API 기반 초기 상품 데이터 수집
├── requirements.txt       # 의존 패키지 목록
├── README.md              # 프로젝트 설명 문서
├── test-image.png         # 테스트용 이미지
├── chroma_db/             # ChromaDB 저장소
├── models/                # 모델 캐시 폴더
├── fashion_products.csv   # 크롤링 결과 CSV
└── fashion_products.json  # 크롤링 결과 JSON
```

---

## 5. 주요 파일 설명

### app.py

프로젝트의 메인 실행 파일입니다.

주요 역할:

```text
Gradio UI 생성
YOLO 모델 로드
CLIP 모델 로드
LLM 모델 로드
이미지 업로드 처리
패션 아이템 탐지
색상 포함 검색어 생성
네이버 쇼핑 API 검색
ChromaDB 텍스트 검색
CLIP 이미지 유사도 검색
채팅 응답 처리
예산 필터링 결과 출력
```

주요 함수:

```text
load_models()
detect_fashion_objects()
extract_clip_features()
search_products()
save_products_to_vectorstore()
search_from_vectorstore()
save_products_to_clip_collection()
search_similar_products_by_image()
merge_and_deduplicate_products()
chat_response()
create_interface()
```

---

### utils.py

공통 유틸리티 함수가 들어있는 파일입니다.

주요 역할:

```text
대화 토큰 관리
텍스트 정리
가격 포맷팅
의류 색상 인식
YOLO 클래스명 한국어 변환
예산 조건 파싱
가격 필터링
```

주요 함수:

```text
TokenManager
clean_text()
format_price()
detect_clothing_color()
build_colored_item_name()
parse_budget_request()
apply_budget_filter()
generate_budget_response()
```

---

### config.py

모델 경로, API 설정, 검색 조건 등을 관리하는 설정 파일입니다.

주요 설정:

```text
NAVER_CLIENT_ID
NAVER_CLIENT_SECRET
YOLO_MODEL_PATH
LLM_MODEL_PATH
EMBEDDING_MODEL_PATH
CLIP_MODEL_PATH
CHROMA_PERSIST_DIR
MAX_SEARCH_COUNT
SEARCH_DISPLAY_COUNT
VECTOR_SEARCH_K
CLIP_IMAGE_COLLECTION_NAME
CLIP_SEARCH_K
CLIP_INDEX_MAX_PRODUCTS
GRADIO_SERVER_PORT
GRADIO_SERVER_NAME
```

---

### crawl_products.py

초기 상품 데이터를 수집하기 위한 파일입니다.

주요 역할:

```text
네이버 쇼핑 API로 패션 카테고리별 상품 수집
CSV 파일 저장
JSON 파일 저장
ChromaDB 텍스트 벡터 DB 생성
```

수집 카테고리 예시:

```text
셔츠
바지
원피스
자켓
스커트
니트
코트
```

---

## 6. 설치 및 실행 방법

### 6.1 가상환경 생성

Miniforge 또는 Anaconda 환경 기준입니다.

```bash
conda create -n ai-shopping python=3.10 -y
conda activate ai-shopping
```

---

### 6.2 프로젝트 폴더 이동

```bash
cd C:\ai_week1\15th-ai1\ai-shopping-assistant
```

---

### 6.3 패키지 설치

```bash
pip install -r requirements.txt
```

Hugging Face 모델 다운로드 속도가 느린 경우 선택적으로 설치할 수 있습니다.

```bash
pip install hf_xet
```

---

### 6.4 네이버 쇼핑 API 설정

네이버 개발자 센터에서 애플리케이션을 등록하고 검색 API 사용 신청을 해야 합니다.

발급받은 Client ID와 Client Secret을 `config.py`에 설정합니다.

```python
NAVER_CLIENT_ID = "발급받은 Client ID"
NAVER_CLIENT_SECRET = "발급받은 Client Secret"
```

GitHub에 업로드할 경우 API 키를 직접 올리지 않는 것을 권장합니다.

권장 방식:

```python
import os

NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
```

---

### 6.5 초기 상품 데이터 생성

선택 사항입니다.

```bash
python crawl_products.py
```

실행하면 다음 파일과 DB가 생성됩니다.

```text
fashion_products.csv
fashion_products.json
chroma_db/
```

---

### 6.6 애플리케이션 실행

```bash
python app.py
```

실행 후 브라우저에서 접속합니다.

```text
http://localhost:7860
```

---

## 7. 사용 방법

1. 웹 페이지 접속
2. 패션 이미지 업로드
3. `상품 탐지` 버튼 클릭
4. 탐지된 아이템, 색상, 검색어 확인
5. 추천 상품 목록 확인
6. 채팅창에서 추가 요청 입력

예시 입력:

```text
5만원 이하로 추천해줘
더 저렴한 걸로 보여줘
20000원 이상
신상으로 찾아줘
가격 낮은 순으로 보여줘
```

---

## 8. 구현 결과 예시

### 이미지 탐지 결과

```text
탐지된 아이템

회색 스커트, 회색 긴팔 셔츠

텍스트 검색어: 회색 스커트 회색 긴팔 셔츠

CLIP 유사 이미지 검색: 5개 반영
```

---

### 예산 필터링 결과

```text
사용자 입력:
5만원 이하로 추천해줘

응답:
50,000원 이하 상품을 가격 낮은 순으로 정렬했습니다.
```

```text
사용자 입력:
20000원 이상

응답:
20,000원 이상 상품을 가격 낮은 순으로 정렬했습니다.
```

---

## 9. 실습 과제 구현 내용

### 실습 과제 1. 옷 색상 인식 기능

구현 내용:

```text
YOLO 탐지 bbox 기준 의류 영역 crop
HSV 색공간 기반 대표 색상 추출
한국어 색상명 변환
색상 + 의류명 조합 검색어 생성
네이버 쇼핑 API 검색어에 색상 반영
```

예시:

```text
short sleeve dress
→ 흰색 반팔 원피스
```

---

### 실습 과제 2. 예산 맞춤 가격 필터링 기능

구현 내용:

```text
정규표현식 기반 자연어 예산 파싱
만원 / 원 단위 변환
이하 / 미만 / 이상 / 초과 조건 처리
더 저렴한 / 최저가 / 가격 낮은 순 요청 처리
상품 가격 필터링
낮은 가격순 정렬
```

예시:

```text
10만원 이하
→ max_price = 100000
```

```text
20000원 이상
→ min_price = 20000
```

---

### 실습 과제 3. CLIP 유사도 검색 기능

구현 내용:

```text
상품 이미지 URL 다운로드
상품 이미지를 CLIP 벡터로 변환
CLIP 이미지 전용 ChromaDB Collection 생성
사용자 업로드 이미지를 CLIP 벡터로 변환
cosine similarity 기반 유사 상품 검색
텍스트 검색 결과와 이미지 검색 결과 병합
중복 상품 제거
```

예시 흐름:

```text
사용자 업로드 이미지
→ CLIP 벡터 변환
→ 이미지 Collection 검색
→ 시각적으로 유사한 상품 반환
```

---

## 10. 실행 시 참고 사항

처음 실행할 때는 Hugging Face 모델 다운로드 때문에 시간이 오래 걸릴 수 있습니다.

다운로드되는 주요 모델:

```text
YOLO DeepFashion2 모델
openai/clip-vit-base-patch32
Bllossom/llama-3.2-Korean-Bllossom-3B
jhgan/ko-sroberta-multitask
```

한 번 다운로드가 완료되면 이후 실행부터는 로컬 캐시를 사용하므로 더 빠르게 실행됩니다.

---

## 11. GitHub 업로드 시 주의사항

다음 파일과 폴더는 GitHub에 업로드하지 않는 것을 권장합니다.

```gitignore
.venv/
__pycache__/
*.pyc
.env
chroma_db/
models/
fashion_products.csv
fashion_products.json
```

특히 네이버 API 키는 절대 GitHub에 직접 업로드하지 않도록 주의해야 합니다.

권장 `.gitignore` 예시:

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.pyd

# Virtual Environment
.venv/
env/
venv/

# Environment Variables
.env

# Model / DB Cache
chroma_db/
models/
.cache/

# Crawled Data
fashion_products.csv
fashion_products.json

# OS
.DS_Store
Thumbs.db
```

---

## 12. 향후 개선 방향

```text
상품 추천 결과에 브랜드 / 가격대별 필터 추가
CLIP 이미지 검색 결과의 유사도 점수 표시
상품 이미지 벡터를 사전에 batch indexing하여 검색 속도 개선
Gradio UI 디자인 개선
네이버 API 키를 .env 기반으로 관리
LLM 응답 속도 개선을 위한 경량 모델 적용
의류 클래스명 한국어 매핑 확장
색상 인식 정확도 향상을 위한 K-Means 기반 대표 색상 추출 적용
```

---

## 13. 핵심 성과

본 프로젝트에서는 단순 이미지 분류가 아니라 다음 기능들을 하나의 쇼핑 추천 파이프라인으로 통합했습니다.

```text
컴퓨터 비전 기반 객체 탐지
색상 인식 기반 검색어 확장
네이버 쇼핑 API 연동
텍스트 임베딩 기반 상품 검색
CLIP 기반 이미지 유사도 검색
LLM 기반 대화형 추천
자연어 예산 조건 필터링
Gradio 기반 웹 UI 구현
```

이를 통해 사용자는 상품명을 직접 입력하지 않아도 이미지만 업로드하여 유사한 패션 상품을 탐색하고, 예산 조건에 맞는 상품을 추천받을 수 있습니다.
