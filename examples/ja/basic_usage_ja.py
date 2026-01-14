"""Basic usage example for RLM."""

import os
import sys
import warnings

# Ignore Pydantic warnings from litellm
warnings.filterwarnings("ignore", module="pydantic")

# Add src to path to import rlm_ja
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from dotenv import load_dotenv
from rlm_ja import RLM

# Load environment variables from .env file
load_dotenv()

# Sample long document (Japanese translation of the original text)
long_document = """
人工知能の歴史

はじめに
過去数十年の間に、人工知能（AI）は理論的な概念から実用的な現実へと変化しました。
この文書では、AI開発における主要なマイルストーンを探ります。

1950年代：AIの誕生
1950年、アラン・チューリングは「計算機と知性」を発表し、有名なチューリング・テストを導入しました。
「人工知能」という用語は、1956年のダートマス会議でジョン・マッカーシー、マービン・ミンスキーらによって作られました。

1960年代-1970年代：初期の楽観主義
この時期、研究者たちはELIZA（1966年）のような初期のAIプログラムやエキスパートシステムを開発しました。
しかし、計算能力の限界により、1970年代には最初の「AIの冬」が訪れました。

1980年代-1990年代：エキスパートシステムとニューラルネットワーク
1980年代にはエキスパートシステムが商業的に成功しました。
1986年にはバックプロパゲーション・アルゴリズムによってニューラルネットワーク研究が活性化しました。

2000年代-2010年代：機械学習の革命
ビッグデータの台頭と強力なGPUにより、ディープラーニングのブレークスルーが可能になりました。
2012年、AlexNetがImageNetコンペティションで優勝し、ディープラーニングの転換点となりました。

2020年代：大規模言語モデル
GPT-3（2020年）やChatGPT（2022年）は、前例のない言語理解能力を示しました。
これらのモデルは何十億ものパラメータを持ち、膨大な量のテキストデータで訓練されています。

結論
AIは医療、交通、教育など数え切れないほどの分野で応用され、急速に進化し続けています。
未来にはさらにエキサイティングな発展が約束されています。
""" * 10  # Multiply to make it longer


def main():
    """Run basic RLM example."""
    rlm = RLM(
        model="gpt-4o",  # Use mini for cheaper testing
        max_iterations=15
    )

    query = "この文書によると、AI開発における主要なマイルストーンは何ですか？年代ごとにまとめてください。"

    try:
        result = rlm.completion(query, long_document)
        print(result)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Make sure to set your API key in .env file or as environment variable:
    # OPENAI_API_KEY=sk-...

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found!")
        print()
        print("Please set up your API key:")
        print("  1. Copy .env.example to .env")
        print("  2. Add your OpenAI API key to .env")
        print("  3. Or run: python setup_env.py")
        exit(1)

    main()
