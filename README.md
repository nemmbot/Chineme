## Example usage of /news and /deepnews:
If you are using AskNews, here is some useful example code.
```python
from asknews_sdk import AsyncAskNewsSDK
import asyncio

"""
More information available here:
https://docs.asknews.app/en/news
https://docs.asknews.app/en/deepnews

Installation:
pip install asknews
"""

client_id = ""
client_secret = ""

ask = AsyncAskNewsSDK(
    client_id=client_id,
    client_secret=client_secret,
    scopes=["chat", "news", "stories", "analytics"],
)

# /news endpoint example
async def search_news(query):

  hot_response = await ask.news.search_news(
      query=query, # your natural language query
      n_articles=5, # control the number of articles to include in the context
      return_type="both",
      strategy="latest news" # enforces looking at the latest news only
  )

  print(hot_response.as_string)

  # get context from the "historical" database that contains a news archive going back to 2023
  historical_response = await ask.news.search_news(
      query=query,
      n_articles=10,
      return_type="both",
      strategy="news knowledge" # looks for relevant news within the past 60 days
  )

  print(historical_response.as_string)

# /deepnews endpoint example:
async def deep_research(
    query, sources, model, search_depth=2, max_depth=2
):

    response = await ask.chat.get_deep_news(
        messages=[{"role": "user", "content": query}],
        search_depth=search_depth,
        max_depth=max_depth,
        sources=sources,
        stream=False,
        return_sources=False,
        model=model,
        inline_citations="numbered"
    )

    print(response)


if __name__ == "__main__":
    query = "What is the TAM of the global market for electric vehicles in 2025? With your final report, please report the TAM in USD using the tags <TAM> ... </TAM>"

    sources = ["asknews"]
    model = "deepseek-basic"
    search_depth = 2
    max_depth = 2
    asyncio.run(
        deep_research(
            query, sources, model, search_depth, max_depth
        )
    )

    asyncio.run(search_news(query))
```

Some tips for DeepNews:

You will get tags in your response, including:

<think> </think>
<asknews_search> </asknews_search>
<final_response> </final_response>

These tags are likely useful for extracting the pieces that you need for your pipeline. For example, if you don't want to include all the thinking/searching, you could just extract <final_response> </final_response>


## Ideas for bot improvements
Below are some ideas for making a novel bot.
- Finetuned LLM on Metaculus Data: Create an optimized prompt (using DSPY or a similar toolset) and/or a fine-tuned LLM using all past Metaculus data. The thought is that this will train the LLM to be well-calibrated on real-life questions. Consider knowledge cutoffs and data leakage from search providers.
- Dataset explorer: Create a tool that can find if there are datasets or graphs related to a question online, download them if they exist, and then run data science on them to answer a question.
- Question decomposer: A tool that takes a complex question and breaks it down into simpler questions to answer those instead
- Meta-Forecast Researcher: A tool that searches all major prediction markets, prediction aggregators, and possibly thought leaders to find relevant forecasts, and then combines them into an assessment for the current question (see [Metaforecast](https://metaforecast.org/)).
- Base rate researcher: Create a tool to find accurate base rates. There is an experimental version [here](https://forecasting-tools.streamlit.app/base-rate-generator) in [forecasting-tools](https://github.com/Metaculus/forecasting-tools) that works 50% of the time.
- Key factors researcher: Improve our experimental [key factors researcher](https://forecasting-tools.streamlit.app/key-factors) to find higher significance key factors for a given question.
- Monte Carlo Simulations: Experiment with combining some tools to run effective Monte Carlo simulations. This could include experimenting with combining Squiggle with the question decomposer.
- Adding personality diversity, LLM diversity, and other variations: Have GPT come up with a number of different ‘expert personalities’ or 'world-models' that it runs the forecasting bot with and then aggregates the median. Additionally, run the bot on different LLMs and see if the median of different LLMs improves the forecast. Finally, try simulating up to hundreds of personalities/LLM combinations to create large, diverse crowds. Each individual could have a backstory, thinking process, biases they are resistant to, etc. This will ideally improve accuracy and give more useful bot reasoning outputs to help humans reading the output consider things from multiple angles.
- Worldbuilding: Have GPT world build different future scenarios and then forecast all the different parts of those scenarios. It would then choose the most likely future world. In addition to a forecast, descriptions of future ‘worlds’ are created. This can take inspiration from Feinman paths.
- Consistency Forecasting: Forecast many tangential questions all at once (in a single prompt) and prompts for consistency rules.
- Extremize & Calibrate Predictions: Using the historical performance of a bot, adjust forecasts to be better calibrated. For instance, if predictions of 30% from the bot actually happen 40% of the time, then transform predictions of 30% to 40%.
- Assigning points to evidence: Starting with some ideas from a [blog post from Ozzie Gooen](https://forum.effectivealtruism.org/posts/mrAZFnEjsQAQPJvLh/using-points-to-rate-different-kinds-of-evidence), you could experiment with assigning ‘points’ to major types of evidence and having GPT categorize the evidence it finds related to a forecast so that the ‘total points’ can be calculated. This can then be turned into a forecast, and potentially optimized using machine learning on past Metaculus data.
- Search provider benchmark: Run bots using different combinations of search providers (e.g. Google, Bing, Exa.ai, Tavily, AskNews, Perplexity, etc) and search filters (e.g. only recent data, sites with a certain search rank, etc) and see if any specific one is better than others, or if using multiple of them makes a difference.
- Timeline researcher: Make a tool that can take a niche topic and make a timeline for all major and minor events relevant to that topic.
- Research Tools: Utilize the ComputerUse and DataAnalyzer tool from forecasting-tools for advanced analysis and to find/analyze datasets.
