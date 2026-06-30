import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


load_dotenv()


# レビューで承認されない場合でも、無限にAPIを呼び続けないように上限を設けます。
MAX_LOOPS = 3
THREAD_ID = "translation-review-demo"
MODEL_NAME = os.getenv("GOOGLE_MODEL", "gemini-3.1-flash-lite")


class AgentState(TypedDict):
    # LangGraphの各ノード間で共有する状態です。
    # 翻訳結果、レビュー指摘、承認状態、ループ回数をここに集約します。
    original_text: str
    current_translation: str
    review_comment: str
    is_approved: bool
    loop_count: int


def require_google_api_key() -> None:
    # APIキーはコードに直接書かず、.envまたは環境変数から読み込みます。
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "[YOUR_GEMINI_API_KEY]":
        raise ValueError("適切な GOOGLE_API_KEY を .env に設定してください。")


def get_message_text(response) -> str:
    # モデルの応答形式が文字列・リストのどちらでも扱えるように本文だけを取り出します。
    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts: list[str] = []
        for block in content:
            if isinstance(block, str):
                texts.append(block)
            elif isinstance(block, dict):
                texts.append(str(block.get("text", block.get("content", block))))
            else:
                texts.append(str(block))
        return "\n".join(texts)

    return str(content)


require_google_api_key()
# temperature=0にして、翻訳・レビュー結果の揺れを抑えます。
model = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)


def translator_node(state: AgentState) -> dict:
    print("\n--- [Translator] 翻訳中... ---")

    # 初回は指摘なし、2回目以降はReviewerからの修正指示を反映して再翻訳します。
    review_instruction = state.get("review_comment") or "なし"
    prompt = (
        "あなたはプロの翻訳家です。以下の日本語を自然な英語に翻訳してください。\n"
        "レビューの修正指示がある場合は、その内容を反映してください。\n"
        f"元テキスト: {state['original_text']}\n"
        f"修正指示: {review_instruction}\n"
    )

    response = model.invoke([HumanMessage(content=prompt)])
    translation = get_message_text(response).strip()

    # ノードは更新したいState項目だけを返します。
    # current_translationは上書きされ、loop_countは翻訳実行ごとに加算されます。
    return {
        "current_translation": translation,
        "loop_count": state.get("loop_count", 0) + 1,
    }


def reviewer_node(state: AgentState) -> dict:
    print("--- [Reviewer] レビュー中... ---")

    # Reviewerは翻訳品質を判定し、承認時はAPPROVED、修正時は日本語の改善案を返します。
    prompt = (
        "あなたは英語記事の編集者です。以下の翻訳結果が自然なビジネス英語になっているかレビューしてください。\n"
        "もし完璧であれば、最初の1行に 'APPROVED' とだけ記述してください。\n"
        "修正が必要な場合は、具体的な修正理由と改善案を日本語で記述してください。\n"
        "もし翻訳結果が複数帰ってきた場合は最も良いものだけを改善案に記載して１つだけなるように指示して下さい\n"
        f"翻訳結果: {state['current_translation']}\n"
    )

    response = model.invoke([HumanMessage(content=prompt)])
    content = get_message_text(response).strip()
    # 先頭行のAPPROVEDを終了判定に使い、それ以外は次ループの修正指示として保持します。
    is_approved = content.startswith("APPROVED")
    review_comment = "" if is_approved else content

    print(f"--- [Review Result: Loop {state['loop_count']}] ---")
    print("APPROVED" if is_approved else "DENIED")

    return {
        "is_approved": is_approved,
        "review_comment": review_comment,
    }


def should_continue(state: AgentState) -> str:
    # 条件付きエッジ用のルーターです。承認済みならENDへ進みます。
    if state["is_approved"]:
        print("--- [Router] レビュー承認: 終了します。 ---")
        return END

    # 承認されていなくても、ループ上限に達したらAPIコスト抑制のため終了します。
    if state["loop_count"] >= MAX_LOOPS:
        print(f"--- [Router] ループ上限({MAX_LOOPS}回)に達したため終了します。 ---")
        return END

    # 未承認かつ上限未満の場合は、Translatorへ戻して修正ループを続けます。
    print(
        f"--- [Router] 再調整が必要です。次のループへ進みます。"
        f"(回数: {state['loop_count']}/{MAX_LOOPS}) ---"
    )
    return "translator"


def build_graph():
    # AgentStateを共有状態として、TranslatorとReviewerの2ノードを持つグラフを定義します。
    workflow = StateGraph(AgentState)

    workflow.add_node("translator", translator_node)
    workflow.add_node("reviewer", reviewer_node)

    # START -> Translator -> Reviewer の順に実行し、レビュー後だけ条件分岐します。
    workflow.add_edge(START, "translator")
    workflow.add_edge("translator", "reviewer")
    workflow.add_conditional_edges(
        "reviewer",
        should_continue,
        {
            "translator": "translator",
            END: END,
        },
    )

    # InMemorySaverにより、thread_id単位で実行中の状態をメモリ上に保存できます。
    checkpointer = InMemorySaver()
    return workflow.compile(checkpointer=checkpointer)


def main() -> None:
    app = build_graph()

    # original_textだけを入力として与え、その他のState項目は初期値から開始します。
    input_data: AgentState = {
        "original_text": "昨日は遅くまで開発をしていましたが、LangGraphのおかげで作業がスムーズに進みました！",
        "current_translation": "",
        "review_comment": "",
        "is_approved": False,
        "loop_count": 0,
    }
    # thread_idはチェックポイントの識別子、recursion_limitは想定外の深い再帰を防ぐ保険です。
    config = {
        "configurable": {"thread_id": THREAD_ID},
        "recursion_limit": 10,
    }

    # グラフを実行すると、承認または上限到達までStateが更新されながら進みます。
    final_state = app.invoke(input_data, config=config)

    print("\n==================================")
    print(f"model: {MODEL_NAME}")
    print(f"thread_id: {THREAD_ID}")
    print(f"元文章: {final_state['original_text']}")
    print(f"最終翻訳結果: {final_state['current_translation']}")
    print(f"ループ総回数: {final_state['loop_count']}")
    print("==================================")


if __name__ == "__main__":
    main()
