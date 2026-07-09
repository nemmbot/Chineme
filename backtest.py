#!/usr/bin/env python3
"""
Lightweight backtest on resolved Metaculus questions.

Usage:
    poetry run python backtest.py --tournament-id minibench --limit 10

Requires METACULUS_TOKEN and VULTR_SERVERLESS_INFERENCE_API_KEY.
Does not publish forecasts or incur Metaculus submission costs.
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from forecasting_tools import MetaculusClient
from forecasting_tools.data_models.questions import BinaryQuestion
from forecasting_tools.helpers.metaculus_client import ApiFilter

from main import ConservativeHybridBot

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("backtest")


def brier_score(prediction: float, outcome: float) -> float:
    return (prediction - outcome) ** 2


async def run_backtest(tournament_id: str, limit: int) -> None:
    client = MetaculusClient()
    api_filter = ApiFilter(
        allowed_tournaments=[tournament_id],
        allowed_statuses=["resolved"],
        group_question_mode="exclude",
    )
    questions = await client.get_questions_matching_filter(
        api_filter, num_questions=limit
    )
    binary_questions = [q for q in questions if isinstance(q, BinaryQuestion)]
    if not binary_questions:
        logger.warning("No resolved binary questions found.")
        return

    bot = ConservativeHybridBot(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=False,
        skip_previously_forecasted_questions=False,
    )

    scores: list[float] = []
    for q in binary_questions:
        resolution = getattr(q, "resolution", None)
        if resolution is None:
            continue
        outcome = 1.0 if resolution else 0.0
        try:
            report = await bot.forecast_question(q)
            pred = report.prediction_value
            if isinstance(pred, float):
                score = brier_score(pred, outcome)
                scores.append(score)
                logger.info(
                    f"Brier={score:.4f} pred={pred:.3f} outcome={outcome:.0f} | "
                    f"{q.question_text[:70]}"
                )
        except Exception as exc:
            logger.error(f"Failed on Q{q.id_of_question}: {exc}")

    if scores:
        avg = sum(scores) / len(scores)
        logger.info(f"Average Brier over {len(scores)} questions: {avg:.4f}")
    else:
        logger.warning("No scores computed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest bot on resolved questions.")
    parser.add_argument("--tournament-id", default="minibench")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(run_backtest(args.tournament_id, args.limit))
