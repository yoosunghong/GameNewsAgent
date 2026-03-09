import os
import sys
import json
import requests
import feedparser
import re
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone

from google import genai
from google.genai import types

# 1. 환경 변수 및 설정
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TARGET_REPO_PATH = os.environ.get("TARGET_REPO_PATH", ".")
KST = timezone(timedelta(hours=9))
TODAY = datetime.now(KST)
TODAY_STR = TODAY.strftime("%Y-%m-%d")

if not GEMINI_API_KEY:
    print("🚨 GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
    sys.exit(1)

# genai 클라이언트 초기화
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = 'gemini-2.5-flash' 

# 안전 필터 설정
safety_settings = [
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
]

def clean_generated_text(text: str) -> str:
    """AI가 생성한 텍스트 정제 및 마크다운 복구"""
    if not text:
        return text
    
    # 1. [P], [R] 등의 불필요한 말머리 제거
    cleaned_text = re.sub(r'\[[A-Z]{1,2}\]\s*', '', text)
    
    # 2. (2 lines) 같은 가이드 문구 제거
    cleaned_text = re.sub(r'\s*\(\d+\s*lines?\)', '', cleaned_text)
    
    # 3. 깨진 마크다운 링크 자동 복구
    cleaned_text = re.sub(
        r'^(?:###\s*)?(\d+)\.\s*([^\[\n]+?)\s*\((https?://[^\)]+)\)',
        r'### \1. [\2](\3)',
        cleaned_text,
        flags=re.MULTILINE
    )
    
    return cleaned_text

def call_gemini(prompt: str, is_json=False) -> str:
    """Gemini API 호출 함수"""
    config_args = {"safety_settings": safety_settings}
    if is_json:
        config_args["response_mime_type"] = "application/json"
        
    config = types.GenerateContentConfig(**config_args)
    
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=config
        )
        return response.text
    except Exception as e:
        print(f"    ⚠️ Gemini API 호출 오류: {e}")
        raise

def fetch_recent_rss_entries() -> list:
    """최근 24시간 이내의 고품질 기술 RSS 피드 수집"""
    urls = [
        "https://news.ycombinator.com/rss",                           # Hacker News
        "https://www.reddit.com/r/MachineLearning/new/.rss",          # ML/AI 전문
        "https://www.reddit.com/r/GameDev/new/.rss",                 # 게임 개발 (엔지니어링 중심)
        "https://huggingface.co/feeds/papers.xml",                   # Hugging Face Daily Papers (핵심 AI 논문)
        "https://rss.arxiv.org/rss/cs.GR",                           # ArXiv Graphics (렌더링, 시뮬레이션)
        "https://rss.arxiv.org/rss/cs.AI"                            # ArXiv AI (최신 아키텍처)
    ]
    yesterday = TODAY - timedelta(days=1)
    entries = []

    for url in urls:
        print(f"  📥 RSS 파싱 중: {url}")
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                published_tuple = entry.get('published_parsed', entry.get('updated_parsed'))
                if published_tuple:
                    published_dt = datetime(*published_tuple[:6], tzinfo=timezone.utc)
                    if published_dt > yesterday:
                        entries.append({
                            "title": entry.title,
                            "link": entry.link,
                            "summary": entry.get('summary', '')[:500] 
                        })
        except Exception as e:
            print(f"    ⚠️ RSS 로드 실패 ({url}): {e}")
            
    return entries

def extract_webpage_text(url: str) -> str:
    """URL에서 기술 본문 추출"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 불필요한 요소 제거
        for script in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            script.extract()
            
        text = soup.get_text(separator=' ', strip=True)
        return text[:4000] 
    except Exception as e:
        print(f"    ⚠️ 본문 추출 실패: {e}")
        return ""

def main():
    print(f"🚀 [1/5] 기사 수집 및 선별...")
    rss_entries = fetch_recent_rss_entries()
    if not rss_entries:
        print("🚨 신규 기사 없음"); sys.exit(0)

    rss_text = "\n".join([f"- {e['title']}\n  {e['link']}" for e in rss_entries])
    step1_prompt = f"Select TOP 5 technical articles for Game/AI Engineers. Avoid business news. Return JSON only: [{{'title': '...', 'link': '...'}}]. Articles:\n{rss_text}"
    selected_articles = json.loads(call_gemini(step1_prompt, is_json=True))

    print(f"🚀 [2/5] 기사 분석 및 요약...")
    summaries = []
    for article in selected_articles:
        content = extract_webpage_text(article['link'])
        # 변경점: 제목에서 링크를 빼고, 별도의 Link 섹션을 명시함
        step2_prompt = f"""Summarize this technical article for a Senior Engineer.
        Strictly follow this format:
        
        ### {article['title']}
        **Link:** {article['link']}
        
        * **Core Content:** (2-3 sentences about the technical mechanism)
        * **Technical Significance:** (Why this matters for engineering)
        * **Practical Application:** (How to use this in real-world projects)

        Content to summarize:
        {content if content else 'N/A'}
        """
        summaries.append(call_gemini(step2_prompt))

    print(f"🚀 [3/5] 영문 마크다운 생성...")
    combined_summaries = "\n\n---\n\n".join(summaries)
    
    # 변경점: 프론트매터가 '문서 전체에서 딱 한 번'만 나와야 함을 강조
    step3_en_prompt = f"""
    You are a professional technical blogger. Combine the provided summaries into a single cohesive blog post.
    
    [STRICT RULES]
    1. The Markdown Frontmatter (the section between ---) MUST appear ONLY ONCE at the very top of the document.
    2. Do NOT repeat the frontmatter for each article.
    3. Start the content immediately after the second --- of the frontmatter.
    4. Maintain the 'Link:' section for every article as provided.
    
    [Format]
    ---
    title: "[Single Technical Title for the Entire Post]"
    date: {TODAY.strftime("%Y-%m-%dT09:00:00+09:00")}
    draft: false
    description: "[Concise 2-sentence summary of all 5 topics]"
    tags: ["Tech", "AI", "GameDev"]
    categories: ["Tech"]
    ---

    {combined_summaries}
    """
    final_markdown_en = clean_generated_text(call_gemini(step3_en_prompt))

    print(f"🚀 [4/5] 한글 버전 번역...")
    # 변경점: 링크 주소는 번역하지 말고 그대로 유지하도록 지시
    step4_ko_prompt = f"""
    Translate the following technical blog post into Korean.
    
    [Rules]
    1. Translate the 'title' and 'description' in the frontmatter.
    2. Keep technical terms like 'Rendering', 'LLM', 'Pipeline' in English if they are standard industry terms.
    3. IMPORTANT: Never translate or modify the URLs in the '**Link:**' section.
    4. Ensure the Frontmatter structure (---) remains intact at the top.
    
    [English Post]
    {final_markdown_en}
    """
    final_markdown_ko = clean_generated_text(call_gemini(step4_ko_prompt))

    print(f"🚀 [4/5] 한글 버전 번역 (제목 포함 전문 번역)...")
    step4_ko_prompt = f"""
    Translate this technical post into Korean.
    - IMPORTANT: Translate the 'title' and 'description' in the Frontmatter into Korean.
    - IMPORTANT: Do NOT include any introductory text like '게임 및 AI 엔지니어링 동향...'.
    - Keep the technical terms (Rendering, LLM, etc.) and the structure (including '---' separators) exactly as they are.
    
    [English Post]
    {final_markdown_en}
    """
    final_markdown_ko = clean_generated_text(call_gemini(step4_ko_prompt))

    print(f"🚀 [5/5] 저장 중...")
    target_dir = os.path.join(TARGET_REPO_PATH, "content", "journal")
    os.makedirs(target_dir, exist_ok=True)
    
    with open(os.path.join(target_dir, f"{TODAY_STR}_news.md"), "w", encoding="utf-8") as f:
        f.write(final_markdown_en)
    with open(os.path.join(target_dir, f"{TODAY_STR}_news.ko.md"), "w", encoding="utf-8") as f:
        f.write(final_markdown_ko)
        
    print(f"🎉 완료: {TODAY_STR}_news.ko.md")

if __name__ == "__main__":
    main()
