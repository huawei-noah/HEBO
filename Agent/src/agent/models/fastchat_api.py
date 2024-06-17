import requests

from agent.models.openai_api import OpenAIAPIEmbeddingBackend
from agent.models.openai_api import OpenAIAPILanguageBackend


class FastChatAPILanguageBackend(OpenAIAPILanguageBackend):
    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        """Counts the number of tokens in a given text according to the model's tokenizer.

        Args:
            list[dict[str, str]]: The list of messages to count tokens for.

        Returns:
            int: The number of tokens after encoding the prompt.
        """
        # URL of the API endpoint
        api_url = self.base_url[:-3] + "/api/v1/token_check"

        # TODO: very approximate:
        concat_prompt = "\n".join([m["content"] for m in messages])

        # Data to be sent to API
        data = {"prompts": [{"prompt": concat_prompt, "model": self.model_id, "max_tokens": 0}]}

        # Call the API
        response = requests.post(api_url, json=data)

        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            total_tokens = sum(item["tokenCount"] for item in result["prompts"])
            return total_tokens
        else:
            print(f"Error: API call failed with status code {response.status_code}")
            print(response.json())
            return 0


class FastChatAPIEmbeddingBackend(OpenAIAPIEmbeddingBackend):
    pass
