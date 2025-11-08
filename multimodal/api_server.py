from flask import Flask, request, jsonify
from multimodal_infer import analyze_url
import traceback

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        "message": "✅ Multimodal Sentiment Analysis API is running!",
        "usage": {
            "POST /analyze": "Send JSON { 'url': '<your target URL>' } to analyze."
        }
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    POST endpoint for multimodal inference.
    Request JSON format:
        { "url": "https://example.com/article" }
    Response JSON format:
        {
            "url": "...",
            "emotional": "...",
            "sentiment_raw": {...},
            "keywords": [...],
            "image_url": "...",
            "text_snippet": "...",
            "fused_vector_shape": [2816]
        }
    """
    try:
        data = request.get_json(force=True)
        url = data.get('url')

        if not url:
            return jsonify({"error": "Missing 'url' in JSON body"}), 400

        print(f"[API] Received URL: {url}")

        result = analyze_url(url)
        print(f"[API] Analysis complete for: {url}")

        return jsonify({
            "status": "success",
            "url": result.get('url'),
            "emotional": result.get('emotional'),
            "sentiment_raw": result.get('sentiment_raw'),
            "keywords": result.get('keywords'),
            "image_url": result.get('image_url'),
            "text_snippet": result.get('text_snippet'),
            "fused_vector_shape": result.get('fused_vector_shape')
        })

    except Exception as e:
        print("[ERROR]", str(e))
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500


if __name__ == "__main__":
    # Host=0.0.0.0 allows access from your Chrome extension or other devices on the same network
    app.run(host="0.0.0.0", port=5000, debug=True)
