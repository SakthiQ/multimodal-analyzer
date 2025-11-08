# multimodal_infer.py
import os
import requests
from io import BytesIO
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from PIL import Image
import numpy as np

# Transformers & feature extraction
import torch
from transformers import BertTokenizer, BertModel, pipeline

# Keras ResNet for image features
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.preprocessing import image as keras_image

# Keyword extraction (TF-IDF)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import _stop_words
import re

# ---------- Step 1: Helpers to fetch page, text, image URL ----------

def fetch_page(url, timeout=15):
    import re, time

    # 1️⃣ Try with cloudscraper first (bypasses many firewalls)
    try:
        import cloudscraper
        scraper = cloudscraper.create_scraper(
            browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False}
        )
        print(f"[INFO] Using cloudscraper to fetch: {url}")
        html = scraper.get(url, timeout=timeout).text
        # detect NDTV error pattern
        if re.search(r"Reference #[0-9a-f\.]+", html):
            raise ValueError("Got Akamai error page (NDTV block). Trying browser mode...")
        return html
    except Exception as e:
        print(f"[WARN] cloudscraper failed: {e}")

    # 2️⃣ Fallback: try normal requests with fake headers
    try:
        import requests
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/122.0.0.0 Safari/537.36'
            ),
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.google.com/'
        }
        print("[INFO] Using requests fallback...")
        resp = requests.get(url, headers=headers, timeout=timeout)
        html = resp.text
        if re.search(r"Reference #[0-9a-f\.]+", html):
            raise ValueError("Got Akamai error again. Trying full browser simulation...")
        return html
    except Exception as e:
        print(f"[WARN] requests fallback failed: {e}")

    # 3️⃣ Final fallback: headless browser (selenium-stealth)
    try:
        print("[INFO] Launching headless browser...")
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium_stealth import stealth

        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--window-size=1920,1080")
        driver = webdriver.Chrome(options=options)

        stealth(driver,
            languages=["en-US", "en"],
            vendor="Google Inc.",
            platform="Win32",
            webgl_vendor="Intel Inc.",
            renderer="Intel Iris OpenGL",
            fix_hairline=True,
        )

        driver.get(url)
        time.sleep(3)
        html = driver.page_source
        driver.quit()
        return html
    except Exception as e:
        raise RuntimeError(f"All fetch methods failed for {url}: {e}")

    try:
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/122.0.0.0 Safari/537.36'
            ),
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Referer': 'https://www.google.com/',
            'Connection': 'keep-alive'
        }
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code == 403:
            import cloudscraper
            scraper = cloudscraper.create_scraper()
            html = scraper.get(url, timeout=timeout).text
            return html
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        raise RuntimeError(f"Failed to fetch page {url}: {e}")

    try:
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/122.0.0.0 Safari/537.36'
            ),
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Referer': 'https://www.google.com/',
            'Connection': 'keep-alive'
        }
        resp = requests.get(url, headers=headers, timeout=timeout)

        # If blocked, use cloudscraper
        if resp.status_code == 403:
            try:
                import cloudscraper
                scraper = cloudscraper.create_scraper()
                html = scraper.get(url, timeout=timeout).text
                return html
            except Exception as e:
                print(f"[WARN] cloudscraper fallback failed: {e}")
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        raise RuntimeError(f"Failed to fetch page {url}: {e}")


def extract_text_from_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    # join visible paragraph text
    paragraphs = [p.get_text(separator=' ', strip=True) for p in soup.find_all('p')]
    text = ' '.join(paragraphs).strip()
    # fallback: page title
    if not text:
        title = soup.title.string if soup.title else ''
        return (title or '').strip()
    return text

def find_first_image_url(html, base_url):
    soup = BeautifulSoup(html, 'html.parser')
    # prefer og:image if present
    og = soup.find('meta', property='og:image')
    if og and og.get('content'):
        img_url = og['content']
        return make_absolute_url(img_url, base_url)
    # else first <img> with src
    img = soup.find('img')
    if img and img.get('src'):
        return make_absolute_url(img['src'], base_url)
    # no image found
    return None

def make_absolute_url(link, base_url):
    if not link:
        return None
    link = link.strip()
    if link.startswith('//'):
        return 'https:' + link
    if link.startswith('http://') or link.startswith('https://'):
        return link
    return urljoin(base_url, link)

# ---------- Step 2: download image and preprocess for ResNet ----------

def download_image_as_pil(url, timeout=10):
    try:
        r = requests.get(url, timeout=timeout, headers={'User-Agent':'Mozilla/5.0'})
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert('RGB')
        return img
    except Exception as e:
        print(f"[WARN] Failed to download image: {e}")
        return None

def pil_to_resnet_input(pil_img, target_size=(224,224)):
    pil_img = pil_img.resize(target_size)
    arr = keras_image.img_to_array(pil_img)
    arr = np.expand_dims(arr, axis=0)
    arr = resnet_preprocess(arr)
    return arr

# ---------- Step 3: Feature extractors (pretrained) ----------

# Initialize models once (global) for speed
print("Loading pretrained models (BERT and ResNet50). This may take a bit...")
_BERT_TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
_BERT_MODEL = BertModel.from_pretrained('bert-base-uncased')
# Use CPU or GPU automatically via transformers/pytorch
_SENTIMENT = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    device=0 if torch.cuda.is_available() else -1
)


_RESNET = ResNet50(weights='imagenet', include_top=False, pooling='avg')  # returns vector

def extract_text_feature(text, max_len=128):
    # returns numpy vector (pooler_output)
    encoding = _BERT_TOKENIZER(text, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
    with torch.no_grad():
        out = _BERT_MODEL(**encoding)
    # pooler_output shape: (batch, hidden) if available; else mean of last_hidden_state
    if hasattr(out, 'pooler_output') and out.pooler_output is not None:
        vec = out.pooler_output[0].cpu().numpy()
    else:
        # fallback: mean pooling
        vec = out.last_hidden_state.mean(dim=1)[0].cpu().numpy()
    return vec  # shape (768,)

def extract_image_feature_from_pil(pil_img):
    arr = pil_to_resnet_input(pil_img)
    feat = _RESNET.predict(arr)  # shape (1, 2048)
    return feat[0]  # (2048,)

# ---------- Step 4: Fusion ----------

def fuse_features(text_feat, image_feat):
    # simple concatenation fusion; returns 1D numpy vector
    if image_feat is None:
        return text_feat
    return np.concatenate([text_feat, image_feat])

# ---------- Step 5: No-training rule-based emotional prediction ----------

def predict_emotional_from_text(text):
    """
    Smart emotional prediction:
    Adds neutral filtering + topic awareness + intensity classification.
    """
    try:
        snippet = text[:512] if len(text) > 512 else text
        res = _SENTIMENT(snippet)[0]  # {'label': 'NEGATIVE', 'score': 0.98}
        label = res['label'].upper()
        score = res['score']

        # --- Intensity scale ---
        if score >= 0.85:
            intensity = "High"
        elif score >= 0.6:
            intensity = "Medium"
        else:
            intensity = "Low"

        # --- Topic-based neutral detection ---
        neutral_keywords = [
            "temperature", "forecast", "weather", "humidity", "rainfall",
            "partly", "cloudy", "sunny", "air quality", "wind", "showers",
            "monsoon", "rain", "clear sky"
        ]
        text_lower = text.lower()
        if any(word in text_lower for word in neutral_keywords):
            emotion = "NEUTRAL"
            intensity = "Low"
            emotional_flag = False
        else:
            # --- Emotion logic ---
            if label == "NEGATIVE" and score > 0.85:
                emotion = "NEGATIVE"
                emotional_flag = True
            elif label == "POSITIVE" and score > 0.85:
                emotion = "POSITIVE"
                emotional_flag = True
            else:
                emotion = "NEUTRAL"
                emotional_flag = False

        res["emotion"] = emotion
        res["intensity"] = intensity
        res["emotional_impact"] = "Yes" if emotional_flag else "No"

        return emotional_flag, res

    except Exception as e:
        print(f"[WARN] Sentiment pipeline failed: {e}")
        return False, {"emotion": "UNKNOWN", "intensity": "N/A", "emotional_impact": "No"}


# ---------- Step 6: Keyword extraction using TF-IDF ----------

def extract_top_keywords(text, n=7):
    # cleanup text a bit
    text_clean = re.sub(r'\s+', ' ', text)
    # use english stop words from sklearn
    stop_words = list(_stop_words.ENGLISH_STOP_WORDS)
    vec = TfidfVectorizer(stop_words=stop_words, max_df=1.0, min_df=1, ngram_range=(1,2))

    try:
        X = vec.fit_transform([text_clean])
        feature_names = np.array(vec.get_feature_names_out())
        if X.nnz == 0:
            return []
        # get top n features by tfidf score
        scores = X.toarray()[0]
        topn_ids = scores.argsort()[::-1][:n]
        top_features = feature_names[topn_ids]
        # filter out very short tokens
        top_features = [t for t in top_features if len(t) > 1][:n]
        return top_features
    except Exception as e:
        print(f"[WARN] keyword extraction failed: {e}")
        return []

# ---------- Step 7: Main pipeline function ----------

def analyze_url(url):
    base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
    print(f"[INFO] Fetching: {url}")
    html = fetch_page(url)
    text = extract_text_from_html(html)
    if not text:
        print("[WARN] No text extracted from page.")
    img_url = find_first_image_url(html, base)
    pil_img = None
    if img_url:
        print(f"[INFO] Found image URL: {img_url}")
        pil_img = download_image_as_pil(img_url)
    else:
        print("[INFO] No image found on page.")

    # Feature extraction
    print("[INFO] Extracting text feature (BERT)...")
    text_feat = extract_text_feature(text if text else " ")
    img_feat = None
    if pil_img:
        print("[INFO] Extracting image feature (ResNet50)...")
        img_feat = extract_image_feature_from_pil(pil_img)
    else:
        print("[INFO] Skipping image feature (no image).")

    # Fusion
    fused = fuse_features(text_feat, img_feat)
    print(f"[INFO] Fused feature vector length: {fused.shape[0]}")

    # Emotional decision (no training) using pretrained sentiment
    emotional_flag, sentiment_result = predict_emotional_from_text(text if text else "")
    emotional_str = "Yes" if emotional_flag else "No"

    # Keywords
    keywords = extract_top_keywords(text if text else "", n=7)

    # Return structured output
    return {
        'url': url,
        'text_snippet': (text[:500] + '...') if text and len(text) > 500 else text,
        'image_url': img_url,
        'emotional': emotional_str,
        'sentiment_raw': sentiment_result,
        'keywords': keywords,
        'fused_vector_shape': fused.shape
    }

# ---------- If run as script ----------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multimodal inference (no training) from URL")
    parser.add_argument('--url', type=str, required=True, help='Webpage URL to analyze')
    args = parser.parse_args()
    out = analyze_url(args.url)
    print("\n--- RESULT ---")
    #print(f"URL: {out['url']}")
    print(f"Emotional Impact (Yes/No): {out['sentiment_raw']['emotional_impact']}")
    print(f"Sentiment Type: {out['sentiment_raw']['emotion']}")
    print(f"Emotion Intensity: {out['sentiment_raw'].get('intensity', 'N/A')}")
    print(f"Sentiment raw: {out['sentiment_raw']}")
    print(f"Top Keywords: {out['keywords']}")
    print(f"Image URL: {out['image_url']}")
    print(f"Text snippet: {out['text_snippet']}")
    print(f"Fused vector shape: {out['fused_vector_shape']}")

