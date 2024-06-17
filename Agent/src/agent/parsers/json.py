import json
import re
from typing import Dict, List


class JSONOutputParser:
    def __init__(self, expected_keys: List[str]):
        self.expected_keys = expected_keys

    def formatting_instructions(self) -> str:
        base = (
            "Please provide the answer as a JSON output enclosed between lines containing ```json and ``` markers. "
            "For example, respond with:\n```json\n{ "
        )
        for key in self.expected_keys:
            base += f"\n    {key}: <{key}>,"
        base += "\n}\n```"
        return base

    def parse_raw_output(self, raw_output: str) -> Dict[str, str]:
        # Extract JSON string enclosed within ```json and ```
        pattern = r"```json(.*?)```"
        matches = re.search(pattern, raw_output, flags=re.DOTALL)
        if not matches:
            raise ValueError("JSON format not found or incorrectly provided.")

        json_str = matches.group(1).strip()
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format.")

        results = {}
        for key in self.expected_keys:
            normalized_key = key.lower()  # Normalize key for consistent casing
            if normalized_key in data:
                # Convert value to string, maintaining JSON structure if necessary
                results[key] = json.dumps(data[normalized_key], ensure_ascii=False)
            else:
                results[key] = "Key not found"

        return results


# Example usage
if __name__ == "__main__":
    expected_keys = ["Name", "Age", "Location"]
    parser = JSONOutputParser(expected_keys)

    raw_output = """
    ```json
    {
        "Name": "John Doe",
        "Age": 30,
        "Location": "New York"
    }
    ```
    """
    print(parser.formatting_instructions())
    parsed_output = parser.parse_raw_output(raw_output)
    # Ensure the output is in JSON format
    print(json.dumps(parsed_output, indent=2, ensure_ascii=False))
