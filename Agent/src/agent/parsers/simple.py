import re

from agent.parsers import OutputParser


class SimpleOutputParser(OutputParser):
    def formatting_instructions(self) -> str:
        base = "Strictly answer in the format:"
        for key in self.expected_keys:
            base += f"\n{key}: <{key.lower()}>"
        return base

    def parse_raw_output(self, raw_output) -> dict[str, str]:
        pattern = rf"({'|'.join(self.expected_keys)}): (.*?)(?=\n(?:{'|'.join(self.expected_keys)}): |\Z)"
        matches = re.findall(pattern, raw_output, flags=re.DOTALL | re.IGNORECASE)

        results = {}
        for key, value in matches:
            normalized_key = key.rstrip(": ").lower()  # Normalize and capitalize key for consistent casing
            if normalized_key in results:
                results[normalized_key] += "\n" + value.strip()
            else:
                results[normalized_key] = value.strip()

        return results
