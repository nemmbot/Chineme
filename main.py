# main.py

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
from urllib.request import Request, urlopen

import numpy as np
import dotenv
from forecasting_tools import (
    BinaryQuestion,
    BinaryPrediction,
    ConditionalPrediction,
    ConditionalQuestion,
    DatePercentile,
    DateQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
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

# -----------------------------
# Environment & API Keys
# -----------------------------
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
VULTR_SERVERLESS_INFERENCE_API_KEY = os.getenv(
    "VULTR_SERVERLESS_INFERENCE_API_KEY", ""
).strip()
VULTR_API_BASE = "https://api.vultrinference.com/v1"

# -----------------------------
# Models (all via Vultr Serverless Inference)
# See https://api.vultrinference.com/v1/models for the full catalog.
# -----------------------------
_MODEL_PRIMARY = "deepseek-r1"
_MODEL_PARSER = "qwen2.5-32b-instruct"

# Committee: each question is forecast by all three; median is taken.
# Use architecturally diverse models to reduce correlated errors.
_COMMITTEE_MODELS = [
    "deepseek-r1",
    "llama-3.3-70b-instruct-fp8",
    "qwen2.5-32b-instruct",
]


def _make_vultr_llm(model_id: str, **kwargs) -> GeneralLlm:
    """Create a GeneralLlm routed through Vultr's OpenAI-compatible API."""
    return GeneralLlm(
        model=f"openai/{model_id}",
        api_key=VULTR_SERVERLESS_INFERENCE_API_KEY,
        base_url=VULTR_API_BASE,
        # GeneralLlm strips the openai/ prefix before calling litellm; without an
        # explicit provider, litellm cannot route custom-base-url models.
        custom_llm_provider="openai",
        **kwargs,
    )

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ConservativeHybridBot")
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


# ==============================
# Extremization helpers
# ==============================

@dataclass
class ExtremizationConfig:
    enabled: bool = True
    factor: float = 1.45
    floor: float  = 0.02
    ceil: float   = 0.98


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
    return max(cfg.floor, min(cfg.ceil, _sigmoid(x)))


# ==============================
# Tavily helper
# ==============================

class TavilySearcher:
    def __init__(self, api_key: str, max_results: int = 6, timeout_s: int = 25):
        self.api_key    = api_key
        self.max_results = max_results
        self.timeout_s  = timeout_s

    def _post_json(self, url: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req  = Request(url, data=data,
                       headers={"Content-Type": "application/json",
                                "Accept": "application/json"},
                       method="POST")
        with urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))

    async def search(self, query: str) -> dict:
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": self.max_results,
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
        }
        return await asyncio.to_thread(
            self._post_json, "https://api.tavily.com/search", payload
        )


# ==============================
# Main bot
# ==============================

class ConservativeHybridBot(ForecastBot):
    """
    Conservative forecasting bot.

    Research    : Tavily web search + Vultr LLM summarization
    Committee   : 3-model Vultr vote → median aggregation
    Question types: Binary, MultipleChoice, Numeric, Date, Conditional
    Extras      : Superforecasting preamble, extremization (logit factor 1.45)
    """

    _max_concurrent_questions = 1
    _concurrency_limiter      = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 2

    _min_seconds_between_search_calls = 1.2
    _min_seconds_between_llm_calls    = 0.35
    _last_search_call_ts = 0.0
    _last_llm_call_ts    = 0.0

    # Sub-questions inside a ConditionalQuestion that should NOT be re-forecast
    # if we already have a previous forecast on them (mirrors Chineme's behaviour)
    force_reforecast_in_conditional: list[str] = []

    def __init__(self, *args, **kwargs):
        llms = kwargs.pop("llms", None)
        if llms is None:
            primary_llm = _make_vultr_llm(
                _MODEL_PRIMARY, temperature=0.15, timeout=120, allowed_tries=2
            )
            parser_llm = _make_vultr_llm(
                _MODEL_PARSER, temperature=0.15, timeout=60, allowed_tries=2
            )
            llms = {
                "default":    primary_llm,
                "summarizer": primary_llm,
                "researcher": parser_llm,
                "parser":     parser_llm,
            }
        super().__init__(*args, llms=llms, **kwargs)

        self._research_cache: dict[str, str] = {}
        self._tavily = TavilySearcher(api_key=TAVILY_API_KEY,
                                      max_results=6, timeout_s=25)
        self._ext_cfg = ExtremizationConfig(
            enabled=os.getenv("EXTREMIZE_ENABLED", "true").lower()
                    in ["1", "true", "yes", "y"],
            factor=float(os.getenv("EXTREMIZE_FACTOR", "1.45")),
            floor=float(os.getenv("EXTREMIZE_FLOOR",   "0.02")),
            ceil=float(os.getenv("EXTREMIZE_CEIL",     "0.98")),
        )

    # ------------------------------------------------------------------
    # Throttling
    # ------------------------------------------------------------------

    async def _throttle_search(self) -> None:
        now  = time.time()
        wait = (self._last_search_call_ts + self._min_seconds_between_search_calls) - now
        if wait > 0:
            await asyncio.sleep(wait + random.random() * 0.15)
        self._last_search_call_ts = time.time()

    async def _throttle_llm(self) -> None:
        now  = time.time()
        wait = (self._last_llm_call_ts + self._min_seconds_between_llm_calls) - now
        if wait > 0:
            await asyncio.sleep(wait + random.random() * 0.10)
        self._last_llm_call_ts = time.time()

    async def _llm_invoke(self, model_key: str, prompt: str) -> str:
        await self._throttle_llm()
        return await self.get_llm(model_key, "llm").invoke(prompt)

    # ------------------------------------------------------------------
    # Superforecasting preamble
    # ------------------------------------------------------------------

    @staticmethod
    def _superforecasting_preamble() -> str:
        return clean_indents("""
            ## Superforecasting Protocol — follow every step before giving a number

            **1. Reference class first (outside view)**
            Identify the broadest reference class. What fraction of similar past questions
            resolved YES (or at the predicted value)? Anchor your initial estimate there.

            **2. Inside view — case-specific evidence**
            What makes THIS case different?
            - Causal drivers toward YES / a higher value
            - Causal drivers toward NO / a lower value
            - Key uncertainties that could flip the outcome

            **3. Adjust for scope and time horizon**
            Longer horizons regress to base rates. Short horizons with strong status-quo
            momentum should reflect that inertia.

            **4. Check for cognitive biases**
            Availability, anchoring, conjunction fallacy, overconfidence.

            **5. Seek disconfirming evidence**
            What argues most strongly AGAINST your current lean?

            **6. Synthesise: blend outside + inside view**
            Start from the base rate, then adjust — usually by less than feels natural.

            **7. Express calibrated confidence**
            Near 50%: genuine uncertainty. Near 5%/95%: overwhelming evidence only.
            Avoid round numbers unless the evidence truly warrants them.
        """).strip()

    # ------------------------------------------------------------------
    # Research: query decomposition
    # ------------------------------------------------------------------

    async def _decompose_question(self, question: MetaculusQuestion) -> list[str]:
        prompt = clean_indents(f"""
            Return 3–5 web-search queries that would most improve a forecast for the question below.
            Queries should be short, specific, and cover: base rates, key drivers, timelines, and
            prediction markets if relevant.
            Output ONLY a JSON array of strings.

            Question: {question.question_text}
            Resolution criteria: {question.resolution_criteria}
            Fine print: {question.fine_print}
        """)
        try:
            raw   = (await self._llm_invoke("researcher", prompt)).strip()
            start, end = raw.find("["), raw.rfind("]")
            if start != -1 and end > start:
                raw = raw[start:end + 1]
            queries = json.loads(raw)
            if isinstance(queries, list):
                return [q.strip() for q in queries
                        if isinstance(q, str) and q.strip()][:5]
        except Exception:
            pass
        return [
            f"{question.question_text} latest updates",
            f"{question.question_text} base rate historical frequency",
            f"{question.question_text} prediction market probability",
        ]

    # ------------------------------------------------------------------
    # Research: individual sources
    # ------------------------------------------------------------------

    def _format_tavily_results(self, query: str, results: dict) -> str:
        items = results.get("results", []) or []
        lines = [f"Query: {query}"]
        for r in items[:self._tavily.max_results]:
            title   = (r.get("title")   or "").strip()
            url     = (r.get("url")     or "").strip()
            snippet = (r.get("content") or "").strip()
            if title or url or snippet:
                if title:   lines.append(f"- {title}")
                if url:     lines.append(f"  URL: {url}")
                if snippet: lines.append(f"  Notes: {snippet}")
        return "\n".join(lines).strip()

    async def _tavily_research_bundle(self, question: MetaculusQuestion) -> str:
        if not TAVILY_API_KEY:
            return ""
        queries = await self._decompose_question(question)
        market_queries = [
            f"metaforecast {question.question_text}",
            f"prediction market odds {question.question_text}",
        ]
        merged: list[str] = []
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
        return "\n\n".join(b for b in blocks if b.strip()).strip()

    # ------------------------------------------------------------------
    # Research: orchestrator (Tavily + summarize)
    # ------------------------------------------------------------------

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            if question.page_url in self._research_cache:
                return self._research_cache[question.page_url]

            base = clean_indents(f"""
                Question: {question.question_text}
                Resolution criteria: {question.resolution_criteria}
                Fine print: {question.fine_print}
            """).strip()

            try:
                tavily_result = await self._tavily_research_bundle(question)
            except Exception as exc:
                tavily_result = f"Tavily error: {type(exc).__name__}: {exc}"

            research_blocks: list[str] = []
            if tavily_result.strip():
                research_blocks.append(f"--- Tavily web research ---\n{tavily_result.strip()}")

            if not research_blocks:
                self._research_cache[question.page_url] = base
                return base

            blocks_joined = "\n\n".join(research_blocks)
            summarize_prompt = clean_indents(f"""
                You are an assistant to a superforecaster.
                Summarize the most relevant evidence for the question below.
                Include: status quo, key drivers, base rates, timelines/milestones,
                and any market probabilities found. Be concise but information-dense.

                {base}

                {blocks_joined}
            """)
            try:
                summary = await self._llm_invoke("summarizer", summarize_prompt)
                final = clean_indents(f"""
                    {base}

                    --- RESEARCH SUMMARY ---
                    {summary}

                    --- RAW RESEARCH SNIPPETS ---
                    {blocks_joined}
                """).strip()
            except Exception:
                final = f"{base}\n\n{blocks_joined}"

            self._research_cache[question.page_url] = final
            logger.info(f"Research cached for {question.page_url}")
            return final

    # ------------------------------------------------------------------
    # Bound helpers (numeric + date)
    # ------------------------------------------------------------------

    def _create_bound_messages(
        self, question: NumericQuestion | DateQuestion
    ) -> tuple[str, str]:
        if isinstance(question, NumericQuestion):
            upper = (question.nominal_upper_bound
                     if question.nominal_upper_bound is not None
                     else question.upper_bound)
            lower = (question.nominal_lower_bound
                     if question.nominal_lower_bound is not None
                     else question.lower_bound)
            unit  = question.unit_of_measure or ""
        else:  # DateQuestion
            upper = question.upper_bound.date().isoformat()
            lower = question.lower_bound.date().isoformat()
            unit  = ""

        upper_msg = (
            f"The question creator thinks the value is likely not higher than {upper} {unit}."
            if question.open_upper_bound
            else f"The outcome cannot be higher than {upper} {unit}."
        )
        lower_msg = (
            f"The question creator thinks the value is likely not lower than {lower} {unit}."
            if question.open_lower_bound
            else f"The outcome cannot be lower than {lower} {unit}."
        )
        return upper_msg, lower_msg

    # ------------------------------------------------------------------
    # Conditional disclaimer helper
    # ------------------------------------------------------------------

    @staticmethod
    def _conditional_disclaimer(question: MetaculusQuestion) -> str:
        if getattr(question, "conditional_type", None) not in ["yes", "no"]:
            return ""
        return clean_indents("""
            You are forecasting a CONDITIONAL question. A parent question has already resolved.
            Forecast ONLY the CHILD question given that parent resolution.
            Do NOT re-forecast or discuss the parent question.
        """).strip()

    # ------------------------------------------------------------------
    # Single-model forecast (one committee member)
    # ------------------------------------------------------------------

    async def _single_forecast(
        self,
        question: MetaculusQuestion,
        research: str,
        model_override: str,
    ):
        """Run one forecast with the given model; returns (prediction, reasoning)."""
        original_default = self._llms.get("default")
        self._llms["default"] = _make_vultr_llm(
            model_override, temperature=0.15, timeout=120, allowed_tries=2
        )

        try:
            if isinstance(question, BinaryQuestion):
                result, reasoning = await self._single_binary(question, research)

            elif isinstance(question, MultipleChoiceQuestion):
                result, reasoning = await self._single_multiple_choice(question, research)

            elif isinstance(question, NumericQuestion):
                result, reasoning = await self._single_numeric(question, research)

            elif isinstance(question, DateQuestion):
                result, reasoning = await self._single_date(question, research)

            else:
                raise ValueError(f"Unsupported question type in committee: {type(question)}")

        finally:
            if original_default is not None:
                self._llms["default"] = original_default

        return result, reasoning

    # ---- per-type single-forecast helpers ----

    async def _single_binary(
        self, question: BinaryQuestion, research: str
    ) -> tuple[float, str]:
        prompt = clean_indents(f"""
            You are a professional superforecaster with a conservative, well-calibrated style.

            {self._superforecasting_preamble()}

            ---
            CRITICAL: This question is NOT YET RESOLVED. Do not forecast near 99% unless
            the resolution criteria are virtually certain to be satisfied.

            Question: {question.question_text}
            Background: {question.background_info}
            Resolution criteria: {question.resolution_criteria}
            Fine print: {question.fine_print}
            Research: {research}
            Today: {datetime.now().strftime("%Y-%m-%d")}

            {self._conditional_disclaimer(question)}

            Reason step-by-step:
            (a) Reference class and base rate
            (b) Time remaining until resolution
            (c) Status quo (outside-view anchor)
            (d) Key YES drivers
            (e) Key NO drivers
            (f) Bias check
            (g) Final synthesis

            Weight the status quo heavily unless strong, specific evidence suggests change.
            End with: "Probability: ZZ%"
        """)
        reasoning = await self._llm_invoke("default", prompt)
        pred: BinaryPrediction = await structure_output(
            reasoning, BinaryPrediction,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
        )
        return max(0.01, min(0.99, pred.prediction_in_decimal)), reasoning

    async def _single_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> tuple[PredictedOptionList, str]:
        prompt = clean_indents(f"""
            You are a professional superforecaster with a conservative, well-calibrated style.

            {self._superforecasting_preamble()}

            ---
            Question: {question.question_text}
            Options: {question.options}
            Background: {question.background_info}
            Resolution criteria: {question.resolution_criteria}
            Fine print: {question.fine_print}
            Research: {research}
            Today: {datetime.now().strftime("%Y-%m-%d")}

            {self._conditional_disclaimer(question)}

            Assign probabilities to every option. Do not assign 0% unless logically impossible.
            Weight the status quo option, but preserve meaningful probability mass for surprises.

            End with probabilities in this exact order {question.options}:
            Option_A: Probability_A
            ...
        """)
        reasoning = await self._llm_invoke("default", prompt)
        result: PredictedOptionList = await structure_output(
            reasoning, PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=f"Options must be exactly: {question.options}",
        )
        return result, reasoning

    async def _single_numeric(
        self, question: NumericQuestion, research: str
    ) -> tuple[NumericDistribution, str]:
        upper_msg, lower_msg = self._create_bound_messages(question)
        prompt = clean_indents(f"""
            You are a professional superforecaster with a conservative, well-calibrated style.

            {self._superforecasting_preamble()}

            ---
            Question: {question.question_text}
            Units: {question.unit_of_measure or "Infer from context"}
            Background: {question.background_info}
            Resolution criteria: {question.resolution_criteria}
            Fine print: {question.fine_print}
            {lower_msg}
            {upper_msg}
            Research: {research}
            Today: {datetime.now().strftime("%Y-%m-%d")}

            {self._conditional_disclaimer(question)}

            Reason step-by-step:
            (a) Reference class and historical base rate
            (b) Status quo / trend-continuation anchor
            (c) Factors pushing the value higher than trend
            (d) Factors pushing the value lower than trend
            (e) Expert or market expectations
            (f) Tail scenarios — calibrate the 10th and 90th percentiles
            (g) Bias check — are my intervals too narrow?

            Use WIDE 90/10 intervals to reflect genuine uncertainty.
            No scientific notation. Percentiles must be strictly increasing.

            End with:
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
        """)
        reasoning = await self._llm_invoke("default", prompt)
        percentile_list: list[Percentile] = await structure_output(
            reasoning, list[Percentile],
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
        )
        return NumericDistribution.from_question(percentile_list, question), reasoning

    async def _single_date(
        self, question: DateQuestion, research: str
    ) -> tuple[NumericDistribution, str]:
        upper_msg, lower_msg = self._create_bound_messages(question)
        prompt = clean_indents(f"""
            You are a professional superforecaster with a conservative, well-calibrated style.

            {self._superforecasting_preamble()}

            ---
            Question: {question.question_text}
            Background: {question.background_info}
            Resolution criteria: {question.resolution_criteria}
            Fine print: {question.fine_print}
            {lower_msg}
            {upper_msg}
            Research: {research}
            Today: {datetime.now().strftime("%Y-%m-%d")}

            {self._conditional_disclaimer(question)}

            Reason step-by-step:
            (a) Reference class: how long do similar events/processes historically take?
            (b) Time already elapsed and current pace (outside-view anchor)
            (c) Status quo / trend-continuation scenario
            (d) Factors that could accelerate the timeline
            (e) Factors that could delay the timeline
            (f) Expert or market expectations on timing
            (g) Tail scenarios — unusually early and unusually late dates
            (h) Bias check — am I anchoring too tightly to the most salient date?

            Use WIDE 90/10 intervals to reflect genuine timing uncertainty.
            Dates must be YYYY-MM-DD. Percentiles must be strictly increasing (chronological).

            End with:
            Percentile 10: YYYY-MM-DD
            Percentile 20: YYYY-MM-DD
            Percentile 40: YYYY-MM-DD
            Percentile 60: YYYY-MM-DD
            Percentile 80: YYYY-MM-DD
            Percentile 90: YYYY-MM-DD
        """)
        reasoning = await self._llm_invoke("default", prompt)
        parsing_instructions = clean_indents(f"""
            Parse a date percentile forecast for: "{question.question_text}"
            If a percentile has no time component, assume midnight UTC.
            If any percentile is missing, indicate it was not explicitly given.
        """)
        date_percentile_list: list[DatePercentile] = await structure_output(
            reasoning, list[DatePercentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )
        # Convert DatePercentile → Percentile (timestamp float) for NumericDistribution
        percentile_list = [
            Percentile(percentile=dp.percentile, value=dp.value.timestamp())
            for dp in date_percentile_list
        ]
        return NumericDistribution.from_question(percentile_list, question), reasoning

    # ------------------------------------------------------------------
    # Committee forecasters → median aggregation
    # ------------------------------------------------------------------

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        forecasts, reasonings = [], []
        for model in _COMMITTEE_MODELS:
            pred, reason = await self._single_forecast(question, research, model)
            forecasts.append(pred)
            reasonings.append(f"[{model}]\n{reason}")

        median_pred = float(np.median(forecasts))
        extremized  = extremize_probability(median_pred, self._ext_cfg)
        logger.info(
            f"Binary  median={median_pred:.3f} → extremized={extremized:.3f} "
            f"| {question.question_text[:60]}"
        )
        return ReasonedPrediction(
            prediction_value=extremized,
            reasoning=" | ".join(reasonings),
        )

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        forecasts, reasonings = [], []
        for model in _COMMITTEE_MODELS:
            pred, reason = await self._single_forecast(question, research, model)
            forecasts.append(pred)
            reasonings.append(f"[{model}]\n{reason}")

        all_probs    = np.array([
            [opt["probability"] for opt in f.predicted_options] for f in forecasts
        ])
        median_probs = np.median(all_probs, axis=0)
        total        = median_probs.sum()
        median_probs = median_probs / total if total > 0 else np.full_like(
            median_probs, 1.0 / len(median_probs)
        )
        options = forecasts[0].predicted_options
        return ReasonedPrediction(
            prediction_value=PredictedOptionList([
                {"option": opt["option"], "probability": float(p)}
                for opt, p in zip(options, median_probs)
            ]),
            reasoning=" | ".join(reasonings),
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        forecasts, reasonings = [], []
        for model in _COMMITTEE_MODELS:
            pred, reason = await self._single_forecast(question, research, model)
            forecasts.append(pred)
            reasonings.append(f"[{model}]\n{reason}")

        target_percentiles = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
        aggregated: list[Percentile] = []
        for p in target_percentiles:
            values = []
            for f in forecasts:
                for item in f.declared_percentiles:
                    if abs(item.percentile - p) < 0.01:
                        values.append(item.value)
                        break
                else:
                    values.append(0.0)
            aggregated.append(Percentile(percentile=p, value=float(np.median(values))))

        return ReasonedPrediction(
            prediction_value=NumericDistribution.from_question(aggregated, question),
            reasoning=" | ".join(reasonings),
        )

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        """
        Date committee: run all three models, median each percentile timestamp,
        then reconstruct a NumericDistribution.
        """
        forecasts, reasonings = [], []
        for model in _COMMITTEE_MODELS:
            pred, reason = await self._single_forecast(question, research, model)
            forecasts.append(pred)
            reasonings.append(f"[{model}]\n{reason}")

        target_percentiles = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
        aggregated: list[Percentile] = []
        for p in target_percentiles:
            values = []
            for f in forecasts:
                for item in f.declared_percentiles:
                    if abs(item.percentile - p) < 0.01:
                        values.append(item.value)
                        break
                else:
                    values.append(0.0)
            aggregated.append(Percentile(percentile=p, value=float(np.median(values))))

        logger.info(
            f"Date    committee complete | {question.question_text[:60]}"
        )
        return ReasonedPrediction(
            prediction_value=NumericDistribution.from_question(aggregated, question),
            reasoning=" | ".join(reasonings),
        )

    # ------------------------------------------------------------------
    # Conditional question support
    # ------------------------------------------------------------------

    async def _run_forecast_on_conditional(
        self, question: ConditionalQuestion, research: str
    ) -> ReasonedPrediction[ConditionalPrediction]:
        """
        Forecast each sub-question of a ConditionalQuestion independently,
        reusing existing forecasts for parent/child when available.
        """
        parent_info, research = await self._forecast_sub_question(
            question.parent, research, "parent"
        )
        child_info, research = await self._forecast_sub_question(
            question.child, research, "child"
        )
        yes_info, research = await self._forecast_sub_question(
            question.question_yes, research, "yes"
        )
        no_info, research = await self._forecast_sub_question(
            question.question_no, research, "no"
        )

        # Extremize any binary sub-predictions
        for info in [parent_info, child_info, yes_info, no_info]:
            pv = getattr(info, "prediction_value", None)
            if isinstance(pv, float):
                info.prediction_value = extremize_probability(pv, self._ext_cfg)  # type: ignore[attr-defined]

        full_reasoning = clean_indents(f"""
            ## Parent Question Reasoning
            {parent_info.reasoning}

            ## Child Question Reasoning
            {child_info.reasoning}

            ## YES Branch Reasoning
            {yes_info.reasoning}

            ## NO Branch Reasoning
            {no_info.reasoning}
        """).strip()

        return ReasonedPrediction(
            prediction_value=ConditionalPrediction(
                parent=parent_info.prediction_value,          # type: ignore
                child=child_info.prediction_value,            # type: ignore
                prediction_yes=yes_info.prediction_value,     # type: ignore
                prediction_no=no_info.prediction_value,       # type: ignore
            ),
            reasoning=full_reasoning,
        )

    async def _forecast_sub_question(
        self,
        question: MetaculusQuestion,
        research: str,
        role: str,  # "parent" | "child" | "yes" | "no"
    ) -> tuple[ReasonedPrediction, str]:
        """
        Forecast a sub-question, reusing the most recent valid existing forecast
        for parent/child roles when available (same logic as Chineme).
        """
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        previous_forecasts = getattr(question, "previous_forecasts", None)
        if (
            role in ("parent", "child")
            and role not in self.force_reforecast_in_conditional
            and previous_forecasts
        ):
            prev = previous_forecasts[-1]
            now  = datetime.now(timezone.utc)
            if prev.timestamp_end is None or prev.timestamp_end > now:
                pretty = DataOrganizer.get_readable_prediction(prev)
                info   = ReasonedPrediction(
                    prediction_value=PredictionAffirmed(),
                    reasoning=f"Existing forecast reaffirmed at {pretty}.",
                )
                return info, research  # type: ignore

        info        = await self._make_prediction(question, research)
        new_research = self._append_sub_reasoning(research, info, role)
        return info, new_research  # type: ignore

    def _append_sub_reasoning(
        self,
        research: str,
        info: ReasonedPrediction,
        role: str,
    ) -> str:
        """Append a completed sub-question's prediction into the running research context."""
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        role_title = role.title()
        pretty     = DataOrganizer.get_readable_prediction(info.prediction_value)
        return clean_indents(f"""
            {research}
            ---
            ## {role_title} Sub-question — already forecasted
            Prediction: {pretty}
            Reasoning:
            ```
            {info.reasoning}
            ```
            Do NOT re-forecast this sub-question. Use it only as context.
        """).strip()


# ==============================
# Entrypoint — Tournament Only
# ==============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Conservative Hybrid Bot.")
    parser.add_argument(
        "--tournament-ids",
        nargs="+",
        type=str,
        default=[
            "33022",
            "market-pulse-26q3",
            MetaculusApi.CURRENT_MINIBENCH_ID,
        ],
    )
    args = parser.parse_args()

    bot = ConservativeHybridBot(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
    )

    try:
        all_reports = []
        for tid in args.tournament_ids:
            logger.info(f"Forecasting on tournament: {tid}")
            reports = asyncio.run(
                bot.forecast_on_tournament(tid, return_exceptions=True)
            )
            all_reports.extend(reports)
        bot.log_report_summary(all_reports)
        logger.info("Run completed successfully.")
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
