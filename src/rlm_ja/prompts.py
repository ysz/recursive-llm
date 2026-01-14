"""System prompt templates for RLM."""


def build_system_prompt(context_size: int, depth: int = 0) -> str:
    """
    Build system prompt for RLM.

    Args:
        context_size: Size of context in characters
        depth: Current recursion depth

    Returns:
        System prompt string
    """
    # Minimal prompt (paper-style)
    prompt = f"""あなたはRecursive Language Model(RLM)です。Python REPL環境を通じてコンテキストと対話します。

コンテキストは変数 `context` に格納されています（このプロンプト内ではありません）。サイズ: {context_size:,} 文字。

環境で使用可能なもの:
- context: str (分析対象のドキュメント)
- query: str (質問: "{"{"}query{"}"}")
- recursive_llm(sub_query, sub_context) -> str (サブコンテキストを再帰的に処理する)
- re: インポート済みの正規表現モジュール (re.findall, re.search などを使用)

質問に答えるためのPythonコードを書いてください。最後の式または print() の出力が表示されます。

例:
- print(context[:100])  # 最初の100文字を確認
- errors = re.findall(r'ERROR', context)  # 'ERROR'をすべて検索
- count = len(errors); print(count)  # カウントして表示

答えが得られたら、FINAL("answer") を使用してください。これは関数ではありません。テキストとして書いてください。

深さ: {depth}"""

    return prompt


def build_user_prompt(query: str) -> str:
    """
    Build user prompt.

    Args:
        query: User's question

    Returns:
        User prompt string
    """
    return query
