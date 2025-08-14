from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from overrides import override
import ast
import json


class SmolLM3Handler(OSSHandler):
    def __init__(self, model_name, temperature, revision=None) -> None:
        super().__init__(model_name, temperature, revision)
        self.is_fc_model = True
        self.model_name_huggingface = model_name.replace("-FC", "")

    @override
    def _format_prompt(self, messages, function):
        """
        "chat_template":
        {# ───── defaults ───── #}
        {%- if enable_thinking is not defined -%}
        {%- set enable_thinking = true -%}
        {%- endif -%}
        {# ───── reasoning mode ───── #}
        {%- if enable_thinking -%}
        {%- set reasoning_mode = "/think" -%}
        {%- else -%}
        {%- set reasoning_mode = "/no_think" -%}
        {%- endif -%}
        {# ───── header (system message) ───── #}
        {{- "<|im_start|>system\n" -}}
        {%- if messages[0].role == "system" -%}
        {%- set system_message = messages[0].content -%}
        {%- if "/no_think" in system_message -%}
            {%- set reasoning_mode = "/no_think" -%}
        {%- elif "/think" in system_message -%}
            {%- set reasoning_mode = "/think" -%}
        {%- endif -%}
        {%- set custom_instructions = system_message.replace("/no_think", "").replace("/think", "").rstrip() -%}
        {%- endif -%}
        {%- if "/system_override" in system_message -%}
        {{- custom_instructions.replace("/system_override", "").rstrip() -}}
        {{- "<|im_end|>\n" -}}
        {%- else -%}
        {{- "## Metadata\n\n" -}}
        {{- "Knowledge Cutoff Date: June 2025\n" -}}
        {%- set today = strftime_now("%d %B %Y") -%}
        {{- "Today Date: " ~ today ~ "\n" -}}
        {{- "Reasoning Mode: " + reasoning_mode + "\n\n" -}}

        {{- "## Custom Instructions\n\n" -}}
        {%- if custom_instructions -%}
            {{- custom_instructions + "\n\n" -}}
        {%- elif reasoning_mode == "/think" -%}
            {{- "You are a helpful AI assistant named SmolLM, trained by Hugging Face. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracking, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> Thought section </think> Solution section. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion.\n\n" -}}
        {%- else -%}
            {{- "You are a helpful AI assistant named SmolLM, trained by Hugging Face.\n\n" -}}
        {%- endif -%}

        {{- "## Tools\n\n" -}}
        {{- "### XML Tools\n\n" -}}
        {%- if tools -%}
            {%- set ns = namespace(xml_tool_string="You may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n\n<tools>\n") -%}
            {%- for tool in tools -%}
            {%- set ns.xml_tool_string = ns.xml_tool_string ~ (tool | tojson) ~ "\n" -%}
            {%- endfor -%}
            {%- set xml_tools = ns.xml_tool_string + "</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags." -%}
        {%- endif -%}
        {%- if xml_tools -%}
            {{- xml_tools -}}
        {%- else -%}
            {{- "None"  -}}
        {%- endif -%}
        {{- "\n\n" -}}
        {{- "### Python Tools\n\n" -}}
        {%- if python_tools -%}
            {{- python_tools -}}
        {%- else -%}
            {{- "None"  -}}
        {%- endif -%}
        {{- "\n\n" -}}
        {{- "<|im_end|>\n" -}}
        {%- endif -%}
        {# ───── main loop ───── #}
        {%- for message in messages -%}
            {%- set content = message.content if message.content is string else "" -%}
            {%- if message.role == "user" -%}
                {{ "<|im_start|>" + message.role + "\n"  + content + "<|im_end|>\n" }}
            {%- elif message.role == "assistant" -%}
                {% generation %}
                {%- if reasoning_mode == "/think" -%}
                    {{ "<|im_start|>assistant\n" + content.lstrip("\n") + "<|im_end|>\n" }}
                {%- else -%}
                    {{ "<|im_start|>assistant\n" + "<think>\n\n</think>\n" + content.lstrip("\n") + "<|im_end|>\n" }}
                {%- endif -%}
                {% endgeneration %}
            {%- elif message.role == "tool" -%}
            {{ "<|im_start|>" + "user\n"  + content + "<|im_end|>\n" }}
            {%- endif -%}
        {%- endfor -%}
        {# ───── generation prompt ───── #}
        {%- if add_generation_prompt -%}
            {%- if reasoning_mode == "/think" -%}
                {{ "<|im_start|>assistant\n" }}
            {%- else -%}
                {{ "<|im_start|>assistant\n" + "<think>\n\n</think>\n"  }}
            {%- endif -%}
        {%- endif -%}
        """
        formatted_prompt = ""
        
        # Define the tool message delimiter used to split system prompts containing tool info
        tool_message = "Here is a list of functions in JSON format that you can invoke.\n"

        formatted_messages = []
        already_found_system_prompt = False
        for message in messages:
            if message["role"] == "system":
                text = message["content"]
                split_text = text.split(tool_message)
                if len(split_text) > 1:
                    assert not already_found_system_prompt, "System prompt with tool description found multiple times"
                    # If the system message contains tool information, we split it
                    already_found_system_prompt = True
                    modified_system_prompt = split_text[0]
                    modified_system_prompt  = modified_system_prompt.replace(
                        "If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]",
                        "If you decide to invoke any of the function(s), you MUST put it in the format specified below.",
                    )
                    tools = ast.literal_eval(split_text[1])
                    for tool in tools:
                        # Remove the mention of python since we expect JSON format instead.
                        tool["description"] = tool["description"].replace("Note that the provided function is in Python 3 syntax.", "")
                        tool["name"] = tool["name"].replace(".", "_")
                    formatted_messages.append({"role": "system", "content": modified_system_prompt})
            else:
                formatted_messages.append(message)

        formatted_prompt = self.tokenizer.apply_chat_template(
            formatted_messages,
            add_generation_prompt=True,
            enable_thinking=True,
            tokenize=False,
            xml_tools="\n".join([json.dumps(tool) for tool in tools]),
        )
        formatted_prompt = formatted_prompt.replace("with function name and arguments within <tool_call></tool_call> XML tags.", "with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>")
        # Write prompt to a file for debugging
        # The function is going to be called multiple times, so let's add to the file instead of overwriting
        with open("smollm3_prompt.txt", "a") as f:
            f.write(formatted_prompt + "*" * 20 + "\n\n")

        return formatted_prompt

    @override
    def decode_ast(self, result, language="Python"):
        result = (
            result.replace("<tool_call>", "")
            .replace("</tool_call>", "")
            .strip()
            .replace(": false", ": False")
            .replace(": true", ": True")
            .replace(": null", ": None")
            .replace("\n", "")
        )
        result_dict = ast.literal_eval(result)
        result_dict = {
            result_dict["name"]: (
                ast.literal_eval(result_dict["arguments"])
                if isinstance(result_dict["arguments"], str)
                else result_dict["arguments"]
            )
        }
        # Below is a generous parsing for python-like function calls.
        # except Exception as e:
        #     # Then maybe the function was given as a function
        #     # Typical input: "function_name(arg1=value1, arg2=value2)"
        #     # We need to parse the input and return the result in the format of {"function_name": {"arg1": "value1", "arg2": "value2"}}
        #     result_dict = {}
        #     if result.startswith("["):
        #         result = result[1:]
        #     if result.endswith("]"):
        #         result = result[:-1]
        #     function_name = result.split("(")[0]
        #     arguments_str = result[len(function_name):]
        #     func_arguments = eval("dict" + arguments_str)
        #     result_dict[function_name] = func_arguments
        return [result_dict]

    @override
    def _parse_query_response_prompting(self, api_response: any) -> dict:
        model_response = api_response.choices[0].text

        reasoning_content = ""
        cleaned_response = model_response
        if "</think>" in model_response:
            parts = model_response.split("</think>")
            reasoning_content = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
            cleaned_response = parts[-1].lstrip("\n")

        return {
            "model_responses": cleaned_response,
            "reasoning_content": reasoning_content,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }


if __name__ == "__main__":
    handler = SmolLM3Handler("HuggingFaceTB/SmolLM3-SFT", 0.1, revision="v27.00-step-000000172")
    messages = [
        {
            "role": "system",
            "content": "You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.\nIf none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.\nYou should only return the function calls in your response.\n\nIf you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]\nYou SHOULD NOT include any other text in the response.\n\nAt each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.\n\nHere is a list of functions in JSON format that you can invoke.\n[{'name': 'environmental_data.air_quality_index', 'description': 'Retrieves Air Quality Index (AQI) for specified location over a number of days. Note that the provided function is in Python 3 syntax.', 'parameters': {'type': 'dict', 'properties': {'location': {'type': 'string', 'description': 'Name of the city or town to retrieve air quality index for.'}, 'days': {'type': 'integer', 'description': 'Number of days for which to retrieve data. If not provided, default to today.'}}, 'required': ['location']}}]\n\n",
        },
        {"role": "user", "content": "Find air quality index in San Jose for next three days."},
    ]
    function = None  # Assuming no function is provided for this example
    formatted_prompt = handler._format_prompt(messages, function)
    print(formatted_prompt)