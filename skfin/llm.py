import os
import ssl
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import pandas as pd
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import httpx

ssl._create_default_https_context = ssl._create_unverified_context


@retry(wait=wait_exponential(multiplier=1, min=6, max=10), stop=stop_after_attempt(5))
def apply_prompt_single_text(
    text,
    content_function,
    response_format,
    nested_response=False,
    model="gpt-4o-mini",
    max_new_tokens=2048,
    temperature=0,
    api_key=None,
):
    """Apply a specified prompt to a single text input using an OpenAI GPT model, and format the response.

    This function retries on failure with an exponential backoff strategy up to 5 attempts.

    Args:
        text (str): The input text to be processed by the prompt.
        content_func (callable): A function that takes a single string argument (text) and returns a formatted string.
        response_format (ResponseFormat): An instance specifying the desired format of the response.
        model (str, optional): The identifier for the model to use. Defaults to "gpt-4o-mini".
        max_new_tokens (int, optional): The maximum number of new tokens to generate in the response. Defaults to 2048.
        temperature (float, optional): Sampling temperature to use for the response generation. A value closer to 0 makes the output more deterministic. Defaults to 0.
        api_key (str, optional): OpenAi key. Defaults to None, in which case, it loaded in the local folder.

    Returns:
        dict: A dictionary where keys correspond to the properties specified in the response format's JSON schema,
              and values are the parsed responses from the model. Returns 'NA' if no output is generated.

    Raises:
        openai.error.OpenAIError: If fails to get a response after the specified retry attempts.
    """

    messages = [
        {
            "role": "system",
            "content": "Assume the role of a financial analyst.",
        },
        {
            "role": "user",
            "content": content_function(text),
        },
    ]
    if api_key is None:
        with open("key", "r") as f:
            api_key = f.read()
    client = OpenAI(api_key=api_key, 
                    http_client=httpx.Client(verify= os.environ.get("REQUESTS_CA_BUNDLE")))
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=response_format,
        max_tokens=max_new_tokens,
        temperature=0,
        seed=42,
        logprobs=True,
        top_logprobs=10,
        top_p=1,
    )
    message = completion.choices[0].message
    if message.parsed:
        keys = list(response_format.model_json_schema()["properties"].keys())
        output = {k: message.parsed.__dict__[k] for k in keys}
        if nested_response:
            assert len(output) == 1
            output = output[list(output.keys())[0]]
            assert isinstance(output, list)
            k, v = output[0].model_json_schema()["properties"].keys()
            output = pd.Series({c.__dict__[k]: c.__dict__[v] for c in output})
    else:
        print("No output")
        output = "NA"

    return output


def apply_prompt(
    dataframe,
    content_function,
    response_format,
    nested_response=False,
    model="gpt-4o-mini",
    text_column="text",
    max_workers=25,
    sequential=False,
    return_dataframe=True,
    stacked_dataframe=False, 
    api_key=None,
):
    """
    Apply a specified prompt to a DataFrame containing text data using an OpenAI GPT model.

    This function can process the text data either sequentially or concurrently using a thread pool.

    Args:
        dataframe (pd.DataFrame): The input DataFrame containing the text data to be processed.
        content_function (callable): A function that takes a single string argument (text) and returns a formatted string.
        response_format (ResponseFormat): An instance specifying the desired format of the response.
        text_columns (str, optional): The column name in the DataFrame that contains the text data. Defaults to 'text'.
        max_workers (int, optional): The maximum number of worker threads to use for concurrent execution. Defaults to 25.
        sequential (bool, optional): If True, process the text data sequentially. If False, process the data concurrently. Defaults to False.
        return_dataframe (bool, optional): If True, return the results as a new DataFrame. If False, return the results as a list. Defaults to True.
        api_key (str, optional): OpenAi key. Defaults to None, in which case, it loaded in the local folder.

    Returns:
        pd.DataFrame or list: The processed output. A DataFrame if `return_dataframe` is True, otherwise a list.
    """

    texts = dataframe[text_column].values

    if sequential:
        results = [
            apply_prompt_single_text(
                text,
                content_function=content_function,
                response_format=response_format,
                nested_response=nested_response,
                model=model,
                api_key=api_key,
            )
            for text in texts
        ]
    else:
        apply_prompt_single_text_partial = partial(
            apply_prompt_single_text,
            content_function=content_function,
            response_format=response_format,
            nested_response=nested_response,
            model=model,
            api_key=api_key,
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(apply_prompt_single_text_partial, texts))

    if return_dataframe:
        if stacked_dataframe:
            return pd.concat({dataframe.index[i]: pd.DataFrame(d) for i, d in enumerate(results)})
        else:
            return pd.DataFrame(results, index=dataframe.index)
    else:
        return results
