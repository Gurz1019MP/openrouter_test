import os
from openai import OpenAI
from dotenv import load_dotenv

# .env ファイルから環境変数をロード
load_dotenv()

# OpenRouter の API キーを取得
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    print("エラー: OPENROUTER_API_KEY が設定されていません。")
    print(".env ファイルに OPENROUTER_API_KEY=\"sk-or-v1-...\" の形式でAPIキーを記述してください。")
    exit(1)

# OpenAI クライアントを OpenRouter のエンドポイントに設定
# base_url を OpenRouter の API URL に指定する点が重要です
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

def chat_with_openrouter(prompt_text, model_name="openai/gpt-4o-mini"):
    """
    OpenRouter を介して指定されたモデルとチャットする関数

    Args:
        prompt_text (str): ユーザーからのプロンプト
        model_name (str): 使用するモデルの名前 (例: "openai/gpt-4o", "anthropic/claude-3-sonnet", "google/gemini-pro")

    Returns:
        str: モデルからの応答メッセージ
    """
    try:
        print(f"モデル: {model_name} を使用して応答を生成中...")
        completion = client.chat.completions.create(
            # OpenRouter 固有のヘッダー (オプション):
            # あなたのサイトを OpenRouter のランキングに表示させるために使用
            extra_headers={
                "HTTP-Referer": "https://your-app-url.com", # あなたのアプリケーションのURL
                "X-Title": "My OpenRouter App", # あなたのアプリケーションのタイトル
            },
            model=model_name,
            messages=[
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.7, # 応答のランダム性を制御
            max_tokens=150,  # 生成されるトークンの最大数
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"エラーが発生しました: {e}"

if __name__ == "__main__":
    print("OpenRouter チャットアプリケーションへようこそ！")
    print("終了するには 'exit' と入力してください。")

    while True:
        user_input = input("\nあなた: ")
        if user_input.lower() == 'exit':
            print("アプリケーションを終了します。")
            break

        # 利用可能なモデルの例 (OpenRouter のドキュメントで最新のリストを確認してください)
        # model_to_use = "openai/gpt-4o"
        # model_to_use = "anthropic/claude-3-sonnet"
        model_to_use = "google/gemini-2.5-flash-preview-05-20" # OpenRouterで利用可能なGeminiモデル

        response = chat_with_openrouter(user_input, model_to_use)
        print(f"AI ({model_to_use}): {response}")
