import os
import re


class PythonInterpreter:
    def __init__(
        self,
        code_path="code.py",
        code_output_path="code_output.txt",
        code_error_path="code_error.txt",
    ):
        self.code_path = code_path
        self.code_output_path = code_output_path
        self.code_error_path = code_error_path

    def __call__(self, agent_input):
        print("============ Python Interpreter ============")
        # parse the agent input to get the python code
        pattern = r"```python(.*?)```"
        code_blocks = re.findall(pattern, agent_input, re.DOTALL)
        # check if the LLM output does not follow the required format
        if len(code_blocks) == 0:
            self.code = ""
            self.code_output = ""
            self.code_error = "The agent does not generate code in the correct format. Please retry!"
            print(self.code_error)
        else:
            self.code = code_blocks[-1]
            with open(self.code_path, "w") as f:
                f.write(self.code)
            # run the code
            os.system(f"python {self.code_path} 2> {self.code_error_path} > {self.code_output_path}")
            with open(self.code_error_path) as f:
                self.code_error = f.read()
            with open(self.code_output_path) as f:
                self.code_output = f.read()
            os.remove(self.code_path)
            os.remove(self.code_error_path)
            os.remove(self.code_output_path)

            print("The code below is run in a python interpreter: ")
            print(self.code)
            if self.code_error == "":
                print("The code runs successfully! Here is the result: ")
                print(self.code_output)
            else:
                print("The code does not run successfully. Here is the error messages: ")
                print(self.code_error)
        print("============================================")

        return {
            "code": self.code,
            "code_output": self.code_output,
            "code_error": self.code_error,
        }
