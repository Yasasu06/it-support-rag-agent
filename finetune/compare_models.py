"""
Model comparison harness — closes the fine-tuning loop.

Runs the full 20-question eval set through TWO models (a base model and an
optional fine-tuned model ID) using the exact same eval-harness metrics
(groundedness, relevance, citation accuracy, refusal correctness, latency), so
a fine-tuned model can be judged the moment one exists. With no --finetuned
arg it scores only the base model, producing a fresh current baseline.

Usage:
    python3 finetune/compare_models.py
    python3 finetune/compare_models.py --finetuned ft:gpt-4o-mini:...
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0,
    os.path.dirname(os.path.dirname(__file__)))

from eval_harness.metrics import (
    score_groundedness,
    score_relevance,
    score_citation_accuracy,
    score_refusal_correctness,
    score_latency
)
from eval import TEST_CASES


def run_pipeline_with_model(
    question: str,
    model_name: str
) -> dict:
    """
    Runs a single question through the RAG pipeline
    using a SPECIFIC model override (not the config
    default), so we can compare models side by side
    without affecting production config.
    """
    import time
    from langchain_openai import ChatOpenAI
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings
    from dotenv import load_dotenv
    load_dotenv()

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    db = Chroma(
        collection_name="it_support_tickets",
        embedding_function=embeddings,
        persist_directory=os.path.join(
            os.path.dirname(
                os.path.dirname(__file__)
            ), "chroma_db"
        )
    )

    start = time.time()
    results = db.similarity_search(question, k=3)
    context = "\n\n".join([
        r.page_content for r in results
    ])
    source_ids = [
        r.metadata.get("ticket_id", "")
        for r in results
    ]

    # Reuse the EXACT production answer prompt (agent_pipeline.answer_prompt),
    # which carries the "definition of sufficient context" guidance. Using a
    # barer prompt here makes the base model over-refuse and produces a baseline
    # that does not reflect the real system — defeating the purpose of a
    # model-vs-model comparison. Only the model swaps; the prompt/retrieval stay
    # identical to production so the delta is attributable to the model alone.
    from agent_pipeline import answer_prompt
    from langchain_core.output_parsers import StrOutputParser

    llm = ChatOpenAI(model=model_name, temperature=0)
    chain = answer_prompt | llm | StrOutputParser()
    answer_text = chain.invoke(
        {"context": context, "question": question}
    )
    latency = time.time() - start

    return {
        "answer": answer_text,
        "source_ids": source_ids,
        "context": context,
        "latency": latency
    }


def run_comparison(
    base_model: str = "gpt-4o-mini",
    finetuned_model: str = None
) -> dict:
    """
    Runs the full 20-question test set through both
    models and scores each on all 5 dimensions. If
    finetuned_model is None, only scores the base model
    (useful for establishing a fresh baseline even
    without a fine-tuned model available yet).
    """
    from agent_pipeline import _get_chat_llm

    models_to_test = {"base": base_model}
    if finetuned_model:
        models_to_test["finetuned"] = finetuned_model

    all_results = {}

    for label, model_name in models_to_test.items():
        print(f"\nTesting model: {label} ({model_name})")
        results = []

        for case in TEST_CASES:
            question = case["question"]
            expected = case["should_answer"]

            pipeline_result = run_pipeline_with_model(
                question, model_name
            )
            answer = pipeline_result["answer"]
            context = pipeline_result["context"]
            source_ids = pipeline_result["source_ids"]
            latency = pipeline_result["latency"]

            scores = {
                "question": question,
                "answer": answer[:200],
                "groundedness": score_groundedness(
                    answer, context, _get_chat_llm
                ),
                "relevance": score_relevance(
                    question, answer, _get_chat_llm
                ),
                "citation_accuracy":
                    score_citation_accuracy(
                        answer, source_ids
                    ),
                "refusal_correctness":
                    score_refusal_correctness(
                        question, answer, expected
                    ),
                "latency": score_latency(latency)
            }
            results.append(scores)
            print(f"  Scored: {question[:50]}...")

        all_results[label] = results

    comparison = {
        "timestamp": datetime.utcnow().isoformat(),
        "base_model": base_model,
        "finetuned_model": finetuned_model,
        "results": {}
    }

    for label, results in all_results.items():
        dims = ["groundedness", "relevance",
                "citation_accuracy",
                "refusal_correctness", "latency"]
        summary = {}
        for dim in dims:
            scores = [
                r[dim]["score"] for r in results
                if r[dim].get("score") is not None
            ]
            summary[dim] = round(
                sum(scores) / len(scores), 3
            ) if scores else None
        comparison["results"][label] = {
            "summary": summary,
            "detailed": results
        }

    output_path = os.path.join(
        os.path.dirname(__file__),
        "model_comparison_results.json"
    )
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)

    return comparison


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=(
            "Compare base model against a fine-tuned "
            "model using the eval harness metrics."
        )
    )
    parser.add_argument(
        "--finetuned",
        default=None,
        help=(
            "Fine-tuned model ID to compare against "
            "base (e.g. ft:gpt-4o-mini:...). If omitted, "
            "only scores the current base model as a "
            "fresh baseline."
        )
    )
    parser.add_argument(
        "--base",
        default="gpt-4o-mini",
        help="Base model to use as comparison point."
    )
    args = parser.parse_args()

    result = run_comparison(
        base_model=args.base,
        finetuned_model=args.finetuned
    )

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    for label, data in result["results"].items():
        print(f"\n{label}:")
        for dim, score in data["summary"].items():
            print(f"  {dim}: {score}")
