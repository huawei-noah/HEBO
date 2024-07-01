import openai


class LLM:

    def __init__(
        self,
        model: str,
        base_url: str,
        temperature: float,
        timeout: float,
        api_key: str,
    ) -> None:
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def __call__(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[dict(role="user", content=prompt)],
            temperature=self.temperature,
            timeout=self.timeout,
        )
        return resp.choices[0].message.content
