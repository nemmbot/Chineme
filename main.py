import argparse
import asyncio
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import dotenv

from forecasting_tools import (
    BinaryPrediction,
    BinaryQuestion,
    ConditionalPrediction,
    ConditionalQuestion,
    DatePercentile,
    DateQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    PredictionAffirmed,
    PredictionTypes,
    PredictedOptionList,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)

dotenv.load_dotenv()
logger = logging.getLogger(__name__)


class TavilySearcher:
    def __init__(
        self,
        api_key: str,
        max_results: int = 6,
        search_depth: str = "basic",
        include_answer: bool = False,
        include_raw_content: bool = False,
        include_images: bool = False,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        timeout_s: int = 25,
    ):
        self.api_key = api_key
        self.max_results = max_results
        self.search_depth = search_depth
        self.include_answer = include_answer
        self.include_raw_content = include_raw_content
        self.include_images = include_images
        self.include_domains = include_domains
        self.exclude_domains = exclude_domains
        self.timeout_s = timeout_s

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=self.timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw)

    async def search(self, query: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "api_key": self.api_key,
            "query": query,
            "max_results": self.max_results,
            "search_depth": self.search_depth,
            "include_answer": self.include_answer,
            "include_raw_content": self.include_raw_content,
            "include_images": self.include_images,
        }
        if self.include_domains:
            payload["include_domains"] = self.include_domains
        if self.exclude_domains:
            payload["exclude_domains"] = self.exclude_domains
        return await asyncio.to_thread(self._post_json, "https://api.tavily.com/search", payload)


@dataclass
class ExtremizationConfig:
    enabled: bool = True
    factor: float = 1.25
    floor: float = 0.01
    ceil: float = 0.99


def _logit(p: float) -> float:
    p = min(1.0 - 1e-12, max(1e-12, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def extremize_probability(p: float, cfg: ExtremizationConfig) -> float:
    if not cfg.enabled:
        return max(cfg.floor, min(cfg.ceil, p))
    x = _logit(p) * cfg.factor
    out = _sigmoid(x)
    return max(cfg.floor, min(cfg.ceil, out))


class SpringTemplateBot2026(ForecastBot):
    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 2

    _min_seconds_between_search_calls = 1.2
    _min_seconds_between_llm_calls = 0.35

    _last_search_call_ts = 0.0
    _last_llm_call_ts = 0.0

    def __init__(self, *args, **kwargs):
        llms = kwargs.pop("llms", None)
        if llms is None:
            free_model = GeneralLlm(
                model="openrouter/openrouter/free",
                temperature=0.2,
                timeout=45,
                allowed_tries=2,
            )
            llms = {
                "default": free_model,
                "summarizer": free_model,
                "researcher": free_model,
                "parser": free_model,
            }
        super().__init__(*args, llms=llms, **kwargs)
        self._research_cache: dict[str, str] = {}
        self._tavily_api_key = os.getenv("TAVILY_API_KEY", "").strip()
        self._tavily = TavilySearcher(
            api_key=self._tavily_api_key,
            max_results=6,
            search_depth="basic",
            include_answer=False,
            include_raw_content=False,
            include_images=False,
            timeout_s=25,
        )
        self._ext_cfg = ExtremizationConfig(
            enabled=os.getenv("EXTREMIZE_ENABLED", "true").lower() in ["1", "true", "yes", "y"],
            factor=float(os.getenv("EXTREMIZE_FACTOR", "1.25")),
            floor=float(os.getenv("EXTREMIZE_FLOOR", "0.01")),
            ceil=float(os.getenv("EXTREMIZE_CEIL", "0.99")),
        )

    async def _throttle_search(self) -> None:
        now = time.time()
        wait = (self._last_search_call_ts + self._min_seconds_between_search_calls) - now
        if wait > 0:
            await asyncio.sleep(wait + random.random() * 0.15)
        self._last_search_call_ts = time.time()

    async def _throttle_llm(self) -> None:
        now = time.time()
        wait = (self._last_llm_call_ts + self._min_seconds_between_llm_calls) - now
        if wait > 0:
            await asyncio.sleep(wait + random.random() * 0.10)
        self._last_llm_call_ts = time.time()

    async def _llm_invoke(self, model_key: str, prompt: str) -> str:
        await self._throttle_llm()
        return await self.get_llm(model_key, "llm").invoke(prompt)

    async def _decompose_question(self, question: MetaculusQuestion) -> list[str]:
        prompt = clean_indents(
            f"""
            You are helping build a research plan for forecasting.

            Return 3 to 5 web-search queries that would most improve a forecast for the question below.
            Queries should be short, specific, and cover: base rates, key drivers, timelines/milestones, and prediction markets if relevant.
            Output ONLY a JSON array of strings.

            Question:
            {question.question_text}

            Resolution criteria:
            {question.resolution_criteria}

            Fine print:
            {question.fine_print}
            """
        )
        try:
            raw = await self._llm_invoke("researcher", prompt)
            raw = raw.strip()
            start = raw.find("[")
            end = raw.rfind("]")
            if start != -1 and end != -1 and end > start:
                raw = raw[start : end + 1]
            queries = json.loads(raw)
            if isinstance(queries, list):
                out = []
                for q in queries:
                    if isinstance(q, str) and q.strip():
                        out.append(q.strip())
                return out[:5]
        except Exception:
            pass
        return [
            f"{question.question_text} latest updates",
            f"{question.question_text} base rate historical frequency",
            f"{question.question_text} prediction market probability",
        ]

    def _format_tavily_results(self, query: str, results: dict[str, Any]) -> str:
        items = results.get("results", []) or []
        lines = [f"Query: {query}"]
        for r in items[: self._tavily.max_results]:
            title = (r.get("title") or "").strip()
            url = (r.get("url") or "").strip()
            snippet = (r.get("content") or "").strip()
            if title or url or snippet:
                lines.append(f"- {title}".strip())
                if url:
                    lines.append(f"  URL: {url}")
                if snippet:
                    lines.append(f"  Notes: {snippet}")
        return "\n".join(lines).strip()

    async def _tavily_research_bundle(self, question: MetaculusQuestion) -> str:
        if not self._tavily_api_key:
            return ""

        queries = await self._decompose_question(question)
        market_queries = [
            f"metaforecast {question.question_text}",
            f"prediction market odds {question.question_text}",
        ]
        merged = []
        for q in queries + market_queries:
            q2 = q.strip()
            if q2 and q2 not in merged:
                merged.append(q2)
        merged = merged[:5]

        blocks: list[str] = []
        for q in merged:
            await self._throttle_search()
            try:
                res = await self._tavily.search(q)
                blocks.append(self._format_tavily_results(q, res))
            except Exception as e:
                blocks.append(f"Query: {q}\n- Search failed: {type(e).__name__}")
        return "\n\n".join([b for b in blocks if b.strip()]).strip()

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            if question.page_url in self._research_cache:
                return self._research_cache[question.page_url]

            base = clean_indents(
                f"""
                Question:
                {question.question_text}

                Resolution criteria:
                {question.resolution_criteria}

                Fine print:
                {question.fine_print}
                """
            ).strip()

            tavily_bundle = await self._tavily_research_bundle(question)
            if tavily_bundle:
                research = clean_indents(
                    f"""
                    {base}

                    --- TAVILY WEB RESEARCH (SNIPPETS) ---
                    {tavily_bundle}
                    """
                ).strip()
            else:
                research = base

            summarize_prompt = clean_indents(
                f"""
                You are an assistant to a superforecaster.
                Summarize the most relevant evidence for forecasting the question below.
                Include: status quo, key drivers, base rates if any, timelines/milestones, and any market probabilities found.
                Be concise but information-dense.

                {research}
                """
            )
            try:
                summary = await self._llm_invoke("summarizer", summarize_prompt)
                final = clean_indents(
                    f"""
                    {base}

                    --- RESEARCH SUMMARY ---
                    {summary}

                    --- RAW RESEARCH SNIPPETS ---
                    {tavily_bundle}
                    """
                ).strip() if tavily_bundle else clean_indents(
                    f"""
                    {base}

                    --- RESEARCH SUMMARY ---
                    {summary}
                    """
                ).strip()
            except Exception:
                final = research

            self._research_cache[question.page_url] = final
            logger.info(f"Found Research for URL {question.page_url}:\n{final}")
            return final

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster.

            Question:
            {question.question_text}

            Background:
            {question.background_info}

            Resolution criteria (not yet satisfied):
            {question.resolution_criteria}

            {question.fine_print}

            Research:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering write:
            (a) Time left until resolution
            (b) Status quo outcome if nothing changed
            (c) A brief No scenario
            (d) A brief Yes scenario

            Weight the status quo heavily unless there is strong evidence of change.
            {self._get_conditional_disclaimer_if_necessary(question)}

            End with: "Probability: ZZ%" (0-100)
            """
        )
        return await self._binary_prompt_to_forecast(question, prompt)

    async def _binary_prompt_to_forecast(
        self,
        question: BinaryQuestion,
        prompt: str,
    ) -> ReasonedPrediction[float]:
        reasoning = await self._llm_invoke("default", prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning,
            BinaryPrediction,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
        )
        decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster.

            Question:
            {question.question_text}

            Options: {question.options}

            Background:
            {question.background_info}

            Resolution criteria:
            {question.resolution_criteria}

            {question.fine_print}

            Research:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering write:
            (a) Time left until resolution
            (b) Status quo outcome if nothing changed
            (c) A plausible unexpected outcome

            {self._get_conditional_disclaimer_if_necessary(question)}
            Put extra weight on status quo, but leave some probability mass for surprises.

            End with probabilities in this exact order {question.options}:
            Option_A: Probability_A
            ...
            """
        )
        return await self._multiple_choice_prompt_to_forecast(question, prompt)

    async def _multiple_choice_prompt_to_forecast(
        self,
        question: MultipleChoiceQuestion,
        prompt: str,
    ) -> ReasonedPrediction[PredictedOptionList]:
        parsing_instructions = clean_indents(
            f"""
            Option names must match one of:
            {question.options}
            Do not drop any option, even if 0%.
            """
        )
        reasoning = await self._llm_invoke("default", prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=reasoning
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster.

            Question:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units: {question.unit_of_measure if question.unit_of_measure else "Not stated (infer)"}

            Research:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting:
            - No scientific notation
            - Percentiles must be strictly increasing

            Before answering write:
            (a) Time left
            (b) Outcome if nothing changed
            (c) Outcome if trend continued
            (d) Expert/market expectations (if any)
            (e) Low unexpected scenario
            (f) High unexpected scenario

            {self._get_conditional_disclaimer_if_necessary(question)}
            Use wide 90/10 intervals.

            End with:
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            """
        )
        return await self._numeric_prompt_to_forecast(question, prompt)

    async def _numeric_prompt_to_forecast(
        self,
        question: NumericQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self._llm_invoke("default", prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            Parse a numeric percentile forecast for: "{question.question_text}"
            Units: {question.unit_of_measure}
            Convert units if needed.
            If percentiles are missing, indicate not explicitly given.
            """
        )
        percentile_list: list[Percentile] = await structure_output(
            reasoning,
            list[Percentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster.

            Question:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Research:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting:
            - Dates must be YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ
            - Percentiles must be chronological and increasing

            Before answering write:
            (a) Time left
            (b) Outcome if nothing changed
            (c) Outcome if trend continued
            (d) Expert/market expectations (if any)
            (e) Early unexpected scenario
            (f) Late unexpected scenario

            {self._get_conditional_disclaimer_if_necessary(question)}
            Use wide 90/10 intervals.

            End with:
            Percentile 10: YYYY-MM-DD
            Percentile 20: YYYY-MM-DD
            Percentile 40: YYYY-MM-DD
            Percentile 60: YYYY-MM-DD
            Percentile 80: YYYY-MM-DD
            Percentile 90: YYYY-MM-DD
            """
        )
        return await self._date_prompt_to_forecast(question, prompt)

    async def _date_prompt_to_forecast(
        self,
        question: DateQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self._llm_invoke("default", prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            Parse a date percentile forecast for: "{question.question_text}"
            If a percentile has no time, assume midnight UTC.
            If percentiles are missing, indicate not explicitly given.
            """
        )
        date_percentile_list: list[DatePercentile] = await structure_output(
            reasoning,
            list[DatePercentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )

        percentile_list = [
            Percentile(
                percentile=percentile.percentile,
                value=percentile.value.timestamp(),
            )
            for percentile in date_percentile_list
        ]
        prediction = NumericDistribution.from_question(percentile_list, question)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion | DateQuestion
    ) -> tuple[str, str]:
        if isinstance(question, NumericQuestion):
            upper_bound_number = (
                question.nominal_upper_bound
                if question.nominal_upper_bound is not None
                else question.upper_bound
            )
            lower_bound_number = (
                question.nominal_lower_bound
                if question.nominal_lower_bound is not None
                else question.lower_bound
            )
            unit_of_measure = question.unit_of_measure
        elif isinstance(question, DateQuestion):
            upper_bound_number = question.upper_bound.date().isoformat()
            lower_bound_number = question.lower_bound.date().isoformat()
            unit_of_measure = ""
        else:
            raise ValueError()

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number} {unit_of_measure}."
        else:
            upper_bound_message = f"The outcome can not be higher than {upper_bound_number} {unit_of_measure}."

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number} {unit_of_measure}."
        else:
            lower_bound_message = f"The outcome can not be lower than {lower_bound_number} {unit_of_measure}."
        return upper_bound_message, lower_bound_message

    async def _run_forecast_on_conditional(
        self, question: ConditionalQuestion, research: str
    ) -> ReasonedPrediction[ConditionalPrediction]:
        parent_info, full_research = await self._get_question_prediction_info(
            question.parent, research, "parent"
        )
        child_info, full_research = await self._get_question_prediction_info(
            question.child, full_research, "child"
        )
        yes_info, full_research = await self._get_question_prediction_info(
            question.question_yes, full_research, "yes"
        )
        no_info, full_research = await self._get_question_prediction_info(
            question.question_no, full_research, "no"
        )
        full_reasoning = clean_indents(
            f"""
            ## Parent Question Reasoning
            {parent_info.reasoning}
            ## Child Question Reasoning
            {child_info.reasoning}
            ## Yes Question Reasoning
            {yes_info.reasoning}
            ## No Question Reasoning
            {no_info.reasoning}
            """
        ).strip()
        full_prediction = ConditionalPrediction(
            parent=parent_info.prediction_value,  # type: ignore
            child=child_info.prediction_value,  # type: ignore
            prediction_yes=yes_info.prediction_value,  # type: ignore
            prediction_no=no_info.prediction_value,  # type: ignore
        )
        return ReasonedPrediction(
            reasoning=full_reasoning, prediction_value=full_prediction
        )

    async def _get_question_prediction_info(
        self, question: MetaculusQuestion, research: str, question_type: str
    ) -> tuple[ReasonedPrediction[PredictionTypes | PredictionAffirmed], str]:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        previous_forecasts = question.previous_forecasts
        if (
            question_type in ["parent", "child"]
            and previous_forecasts
            and question_type not in self.force_reforecast_in_conditional
        ):
            previous_forecast = previous_forecasts[-1]
            current_utc_time = datetime.now(timezone.utc)
            if (
                previous_forecast.timestamp_end is None
                or previous_forecast.timestamp_end > current_utc_time
            ):
                pretty_value = DataOrganizer.get_readable_prediction(previous_forecast)  # type: ignore
                prediction = ReasonedPrediction(
                    prediction_value=PredictionAffirmed(),
                    reasoning=f"Already existing forecast reaffirmed at {pretty_value}.",
                )
                return (prediction, research)  # type: ignore
        info = await self._make_prediction(question, research)
        full_research = self._add_reasoning_to_research(research, info, question_type)
        return info, full_research  # type: ignore

    def _add_reasoning_to_research(
        self,
        research: str,
        reasoning: ReasonedPrediction[PredictionTypes],
        question_type: str,
    ) -> str:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        question_type = question_type.title()
        return clean_indents(
            f"""
            {research}
            ---
            ## {question_type} Question Information
            You have previously forecasted the {question_type} Question to the value: {DataOrganizer.get_readable_prediction(reasoning.prediction_value)}
            This is relevant information for your current forecast, but it is NOT your current forecast.
            The reasoning for the {question_type} Question was:
            ```
            {reasoning.reasoning}
            ```
            Do NOT use this reasoning to re-forecast the {question_type} question.
            """
        ).strip()

    def _get_conditional_disclaimer_if_necessary(
        self, question: MetaculusQuestion
    ) -> str:
        if question.conditional_type not in ["yes", "no"]:
            return ""
        return clean_indents(
            """
            You are given a conditional question with a parent and child.
            Forecast ONLY the CHILD question given the parent’s resolution.
            Do not re-forecast the parent.
            """
        ).strip()

    def _extremize_report_if_binary(self, report: Any) -> None:
        try:
            pv = getattr(report, "prediction_value", None)
            if isinstance(pv, float):
                setattr(report, "prediction_value", extremize_probability(pv, self._ext_cfg))
            pred = getattr(report, "prediction", None)
            if isinstance(pred, float):
                setattr(pred, "prediction", extremize_probability(pred, self._ext_cfg))
        except Exception:
            return

    def _extremize_reports(self, forecast_reports: list[Any]) -> list[Any]:
        for r in forecast_reports:
            self._extremize_report_if_binary(r)
        return forecast_reports

    async def forecast_on_tournament(self, *args, **kwargs):
        reports = await super().forecast_on_tournament(*args, **kwargs)
        if isinstance(reports, list):
            return self._extremize_reports(reports)
        return reports

    async def forecast_questions(self, *args, **kwargs):
        reports = await super().forecast_questions(*args, **kwargs)
        if isinstance(reports, list):
            return self._extremize_reports(reports)
        return reports


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(description="Run the TemplateBot forecasting system")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode
    assert run_mode in ["tournament", "metaculus_cup", "test_questions"]

    template_bot = SpringTemplateBot2026(
        research_reports_per_question=1,
        predictions_per_research_report=3,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
    )

    client = MetaculusClient()
    if run_mode == "tournament":
        seasonal_tournament_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
        minibench_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
        )
        market_pulse_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                "market-pulse-26q1", return_exceptions=True
            )
        )
        forecast_reports = seasonal_tournament_reports + minibench_reports + market_pulse_reports
    elif run_mode == "metaculus_cup":
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",
            "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",
        ]
        template_bot.skip_previously_forecasted_questions = False
        questions = [client.get_question_by_url(question_url) for question_url in EXAMPLE_QUESTIONS]
        forecast_reports = asyncio.run(template_bot.forecast_questions(questions, return_exceptions=True))

    template_bot.log_report_summary(forecast_reports)
