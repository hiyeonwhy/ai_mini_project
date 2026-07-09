# utils.py
"""
AI 쇼핑 어시스턴트 유틸리티 모듈
토큰 관리 및 기타 유틸리티 함수들
"""

import re
import torch
from typing import List, Dict, Tuple
import numpy as np
import cv2

FASHION_CLASS_KO_MAP = {
    "short sleeve top": "반팔 상의",
    "long sleeve top": "긴팔 상의",
    "short sleeve outwear": "반팔 아우터",
    "long sleeve outwear": "긴팔 아우터",
    "vest": "조끼",
    "sling": "민소매 상의",
    "shorts": "반바지",
    "trousers": "바지",
    "skirt": "스커트",
    "short sleeve dress": "반팔 원피스",
    "long sleeve dress": "긴팔 원피스",
    "vest dress": "민소매 원피스",
    "sling dress": "슬링 원피스",

    "short_sleeved_shirt": "반팔 셔츠",
    "long_sleeved_shirt": "긴팔 셔츠",
    "short_sleeved_outwear": "반팔 아우터",
    "long_sleeved_outwear": "긴팔 아우터",
    "sling_dress": "슬링 원피스",
    "vest_dress": "민소매 원피스",
}


class TokenManager:
    """
    대화 히스토리의 토큰을 관리하는 클래스
    토큰 수를 추적하고 컨텍스트 윈도우 내에서 대화를 유지합니다.
    """
    
    def __init__(self, tokenizer, max_context_tokens=2048):
        """
        TokenManager 초기화
        
        Args:
            tokenizer: Hugging Face tokenizer 객체
            max_context_tokens: 최대 컨텍스트 토큰 수
        """
        self.tokenizer = tokenizer
        self.max_context_tokens = max_context_tokens
        
    def count_tokens(self, text: str) -> int:
        """
        텍스트의 토큰 수를 계산합니다.
        
        Args:
            text: 토큰을 계산할 텍스트
            
        Returns:
            int: 토큰 수
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    
    def prepare_prompt(self, system_prompt: str, conversation_history: List[Dict], 
                      current_query: str, context: str = None) -> Tuple[str, int]:
        """
        대화 프롬프트를 준비하고 토큰 수를 계산합니다.
        
        Args:
            system_prompt: 시스템 프롬프트
            conversation_history: 대화 히스토리
            current_query: 현재 사용자 쿼리
            context: 추가 컨텍스트 (검색 결과 등)
            
        Returns:
            Tuple[str, int]: (준비된 프롬프트, 토큰 수)
        """
        # 프롬프트 구성
        prompt_parts = [f"시스템: {system_prompt}\n"]
        
        # 대화 히스토리 추가
        for msg in conversation_history:
            role = "사용자" if msg["role"] == "user" else "어시스턴트"
            prompt_parts.append(f"{role}: {msg['content']}\n")
        
        # 컨텍스트가 있으면 추가
        if context:
            prompt_parts.append(f"검색 결과:\n{context}\n")
        
        # 현재 쿼리 추가
        prompt_parts.append(f"사용자: {current_query}\n")
        prompt_parts.append("어시스턴트:")
        
        full_prompt = "".join(prompt_parts)
        token_count = self.count_tokens(full_prompt)
        
        return full_prompt, token_count
    
    def manage_conversation_history(self, history: List[Dict], 
                                  new_message: Dict) -> Tuple[List[Dict], int]:
        """
        대화 히스토리를 관리하고 토큰 한계 내에서 유지합니다.
        
        Args:
            history: 현재 대화 히스토리
            new_message: 추가할 새 메시지
            
        Returns:
            Tuple[List[Dict], int]: (업데이트된 히스토리, 총 토큰 수)
        """
        # 새 메시지 추가
        updated_history = history + [new_message]
        
        # 토큰 수 계산
        total_tokens = sum(self.count_tokens(msg['content']) for msg in updated_history)
        
        # 토큰 한계 초과 시 오래된 메시지 제거
        while total_tokens > self.max_context_tokens and len(updated_history) > 2:
            # 가장 오래된 사용자-어시스턴트 쌍 제거
            if updated_history[0]['role'] == 'user' and len(updated_history) > 1:
                updated_history = updated_history[2:]
            else:
                updated_history = updated_history[1:]
            
            # 토큰 수 재계산
            total_tokens = sum(self.count_tokens(msg['content']) for msg in updated_history)
        
        return updated_history, total_tokens
    
    def get_token_stats(self, history: List[Dict]) -> Dict[str, int]:
        """
        대화 히스토리의 토큰 통계를 계산합니다.
        
        Args:
            history: 대화 히스토리
            
        Returns:
            Dict[str, int]: 토큰 통계 정보
        """
        total_tokens = sum(self.count_tokens(msg['content']) for msg in history)
        message_count = len(history)
        
        return {
            'total': total_tokens,
            'messages': message_count,
            'average': total_tokens // message_count if message_count > 0 else 0,
            'remaining': self.max_context_tokens - total_tokens
        }
    
    def truncate_text(self, text: str, max_tokens: int) -> str:
        """
        텍스트를 최대 토큰 수에 맞게 잘라냅니다.
        
        Args:
            text: 자를 텍스트
            max_tokens: 최대 토큰 수
            
        Returns:
            str: 잘린 텍스트
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= max_tokens:
            return text
        
        # 토큰을 자르고 디코드
        truncated_tokens = tokens[:max_tokens]
        truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        
        return truncated_text + "..."


def crop_image_by_bbox(image, bbox):
    """
    PIL Image와 YOLO bbox를 받아 옷 영역만 crop합니다.

    Args:
        image: PIL Image
        bbox: (x1, y1, x2, y2)

    Returns:
        cropped: RGB numpy array
    """
    img_array = np.array(image.convert("RGB"))

    h, w = img_array.shape[:2]
    x1, y1, x2, y2 = bbox

    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return img_array[y1:y2, x1:x2]


def convert_color_to_korean_name(rgb):
    """
    RGB 값을 한국어 색상명으로 변환합니다.

    Args:
        rgb: (r, g, b)

    Returns:
        color_name: 한국어 색상명
    """
    r, g, b = rgb

    color = np.uint8([[[r, g, b]]])
    hsv = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)[0][0]

    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    # 무채색 계열
    if v < 45:
        return "검은색"
    if s < 35 and v > 190:
        return "흰색"
    if s < 45:
        return "회색"

    # 유채색 계열
    if h < 10 or h >= 170:
        return "빨간색"
    elif 10 <= h < 20:
        if v < 150:
            return "갈색"
        return "주황색"
    elif 20 <= h < 35:
        return "노란색"
    elif 35 <= h < 85:
        return "초록색"
    elif 85 <= h < 100:
        return "하늘색"
    elif 100 <= h < 130:
        return "파란색"
    elif 130 <= h < 155:
        return "보라색"
    elif 155 <= h < 170:
        return "분홍색"

    return "기타색"


def detect_clothing_color(image, bbox):
    """
    탐지된 옷 영역에서 대표 색상을 추출합니다.

    Args:
        image: PIL Image
        bbox: YOLO bbox 좌표

    Returns:
        color_name: 한국어 색상명
    """
    cropped = crop_image_by_bbox(image, bbox)

    if cropped is None or cropped.size == 0:
        return "색상미상"

    # bbox 가장자리에는 배경이 섞일 수 있으므로 중앙 영역 위주로 사용
    h, w = cropped.shape[:2]

    if h > 20 and w > 20:
        y_margin = int(h * 0.08)
        x_margin = int(w * 0.08)
        cropped = cropped[y_margin:h - y_margin, x_margin:w - x_margin]

    hsv_img = cv2.cvtColor(cropped, cv2.COLOR_RGB2HSV)
    pixels_rgb = cropped.reshape(-1, 3)
    pixels_hsv = hsv_img.reshape(-1, 3)

    # 너무 어두운 픽셀 제거
    valid_mask = pixels_hsv[:, 2] > 35
    valid_rgb = pixels_rgb[valid_mask]

    if len(valid_rgb) == 0:
        return "색상미상"

    # 픽셀 일부만 샘플링해서 속도 개선
    if len(valid_rgb) > 10000:
        indices = np.random.choice(len(valid_rgb), 10000, replace=False)
        valid_rgb = valid_rgb[indices]

    color_counts = {}

    for rgb in valid_rgb:
        color_name = convert_color_to_korean_name(tuple(rgb))
        color_counts[color_name] = color_counts.get(color_name, 0) + 1

    # 가장 많이 나온 색상 반환
    dominant_color = max(color_counts, key=color_counts.get)

    return dominant_color


def build_colored_item_name(color_name, class_name):
    """
    색상명과 YOLO 클래스명을 합쳐 네이버 쇼핑 검색어를 만듭니다.

    예:
        흰색 + short sleeve dress -> 흰색 반팔 원피스
    """
    korean_item_name = FASHION_CLASS_KO_MAP.get(class_name, class_name)

    if color_name and color_name != "색상미상":
        return f"{color_name} {korean_item_name}"

    return korean_item_name

def clean_text(text: str) -> str:
    """
    텍스트를 정리합니다.
    
    Args:
        text: 정리할 텍스트
        
    Returns:
        str: 정리된 텍스트
    """
    # 여러 개의 공백을 하나로
    text = ' '.join(text.split())
    
    # 앞뒤 공백 제거
    text = text.strip()
    
    return text


def format_price(price: str) -> str:
    """
    가격을 포맷팅합니다.
    
    Args:
        price: 가격 문자열
        
    Returns:
        str: 포맷팅된 가격
    """
    try:
        price_int = int(price)
        return f"{price_int:,}원"
    except:
        return 
    
def parse_price_to_int(price) -> int:
    """
    상품 가격 문자열을 int로 변환합니다.
    예: "35,000", "35000", 35000 -> 35000
    """
    if price is None:
        return 0

    try:
        if isinstance(price, int):
            return price

        price_str = str(price)
        price_str = price_str.replace(",", "")
        price_str = price_str.replace("원", "")
        price_str = price_str.strip()

        if not price_str:
            return 0

        return int(float(price_str))
    except:
        return 0


def parse_budget_request(message: str):
    """
    사용자 메시지에서 예산 조건을 추출합니다.

    처리 예:
    - "5만원 이하"
    - "10만원 안으로"
    - "30000원 이하"
    - "20000원 이상"
    - "3만원 초과"
    - "더 저렴한 걸로"
    - "싼 순으로 보여줘"
    """
    if not message:
        return None

    text = clean_text(message)

    cheaper_keywords = [
        "더 저렴",
        "저렴한",
        "싼",
        "싸게",
        "가격 낮",
        "낮은 가격",
        "낮은 순",
        "최저가",
        "가성비",
        "저가",
    ]

    has_cheaper_intent = any(keyword in text for keyword in cheaper_keywords)

    # 가격 조건 방향 판단
    is_min_condition = any(keyword in text for keyword in ["이상", "초과", "부터", "넘는", "넘어"])
    is_max_condition = any(keyword in text for keyword in ["이하", "미만", "안으로", "까지", "내로", "아래"])

    # 기본값: 숫자만 있으면 "이하"로 처리
    if not is_min_condition and not is_max_condition:
        is_max_condition = True

    price_value = None
    operator = "lte"

    # "5만원", "5.5만원", "10 만원" 처리
    manwon_match = re.search(r"(\d+(?:\.\d+)?)\s*만\s*원?", text)
    if manwon_match:
        price_value = int(float(manwon_match.group(1)) * 10000)

    # "50000원", "50,000원" 처리
    won_match = re.search(r"(\d{1,3}(?:,\d{3})+|\d+)\s*원", text)
    if won_match and price_value is None:
        price_value = parse_price_to_int(won_match.group(1))

    if price_value is None and not has_cheaper_intent:
        return None

    min_price = None
    max_price = None

    if price_value is not None:
        if is_min_condition:
            min_price = price_value
            operator = "gte"
            if "초과" in text or "넘는" in text or "넘어" in text:
                operator = "gt"
        else:
            max_price = price_value
            operator = "lte"
            if "미만" in text or "아래" in text:
                operator = "lt"

    return {
        "min_price": min_price,
        "max_price": max_price,
        "operator": operator,
        "sort_low_to_high": True,
        "relative_cheaper": price_value is None and has_cheaper_intent,
        "has_cheaper_intent": has_cheaper_intent,
        "original_message": message,
    }

def apply_budget_filter(products, budget_info):
    """
    상품 리스트에 예산 조건을 적용합니다.
    lprice 기준으로 필터링하고, 가격 낮은 순으로 정렬합니다.
    """
    if not products:
        return []

    if not budget_info:
        return products

    priced_products = []

    for product in products:
        price = parse_price_to_int(
            product.get("lprice") or product.get("price") or product.get("hprice")
        )

        if price <= 0:
            continue

        product_copy = product.copy()
        product_copy["_parsed_price"] = price
        priced_products.append(product_copy)

    min_price = budget_info.get("min_price")
    max_price = budget_info.get("max_price")
    operator = budget_info.get("operator", "lte")

    if min_price is not None:
        if operator == "gt":
            priced_products = [
                product for product in priced_products
                if product["_parsed_price"] > min_price
            ]
        else:
            priced_products = [
                product for product in priced_products
                if product["_parsed_price"] >= min_price
            ]

    if max_price is not None:
        if operator == "lt":
            priced_products = [
                product for product in priced_products
                if product["_parsed_price"] < max_price
            ]
        else:
            priced_products = [
                product for product in priced_products
                if product["_parsed_price"] <= max_price
            ]

    if budget_info.get("sort_low_to_high"):
        priced_products.sort(key=lambda product: product["_parsed_price"])

    for product in priced_products:
        product.pop("_parsed_price", None)

    return priced_products

def generate_budget_response(products, budget_info):
    """
    예산 필터링 결과 안내 문구를 생성합니다.
    """
    if not budget_info:
        return ""

    min_price = budget_info.get("min_price")
    max_price = budget_info.get("max_price")
    operator = budget_info.get("operator", "lte")

    if not products:
        if min_price:
            condition_text = "초과" if operator == "gt" else "이상"
            return f"{format_price(min_price)} {condition_text} 조건에 맞는 상품을 찾지 못했습니다."
        if max_price:
            condition_text = "미만" if operator == "lt" else "이하"
            return f"{format_price(max_price)} {condition_text} 조건에 맞는 상품을 찾지 못했습니다."
        return "현재 추천 목록에서 더 저렴한 상품을 찾지 못했습니다."

    if min_price:
        condition_text = "초과" if operator == "gt" else "이상"
        return f"{format_price(min_price)} {condition_text} 상품을 가격 낮은 순으로 정렬했습니다."

    if max_price:
        condition_text = "미만" if operator == "lt" else "이하"
        return f"{format_price(max_price)} {condition_text} 상품을 가격 낮은 순으로 정렬했습니다."

    return "현재 추천 목록에서 더 저렴한 상품 위주로 다시 정렬했습니다."