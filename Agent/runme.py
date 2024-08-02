import os
from agent.tasks.rosllm import ROSLLM
from minillm.llm import LLM
from minillm import config_path


def create_prompt(obs):
    prompt_path = "/home/c84310777/Documents/rsl-demo/HEBO/Agent/src/agent/prompts/templates/rsl/external_action.txt"
    with open(prompt_path, "r") as f:
        prompt = f.read()
    return prompt + obs


def load_llm():
    path = os.path.join(config_path, "deepseek.yaml")
    return LLM.load(path)


def main():
    env = ROSLLM()
    llm = load_llm()
    done = False
    obs = env.reset()
    while not done:
        prompt = create_prompt(obs)
        response = llm(prompt)
        print(f"======== Step {env.step_index+1} ========")
        print("--FULL PROMPT START--")
        print(prompt)
        print("--FULL PROMPT END--")
        print("--FULL RESPONSE START--")
        print(response)
        print("--FULL RESPONSE END--")
        obs, _, done = env.step(response)


if __name__ == "__main__":
    main()
