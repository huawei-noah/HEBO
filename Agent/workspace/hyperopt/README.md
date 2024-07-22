# How to add a task

This is a guide to add tasks for the Hyperopt experiments showcasing that the Agent is useful in tuning the
hyperparameters of open-source code from Datascience competitions on Kaggle.

## Get code from Kaggle

From Kaggle, pick a competition and take a submission with open code.
Make sure the license is open to at least non-commercial use.

## Create a workspace

For the competition named `task_id`, create the following file structure

```
 workspace/
 | hyperopt/
 | | task_id/
 | | | code/
 | | | | code.py      
 | | | data/
 | | | | train.csv, test.csv, ...
``` 

Subscribe to the competition and download its data into `workspace/hyperopt/task_id/data/`

Download the (open-source) notebook from Kaggle and convert it into a script
in `workspace/hyperopt/task_id/data/code.py`. For that you can
use `jupyter nbconvert --to script --no-prompt NOTEBOOK-NAME.ipynb`

Finally, rename this newly created script to `code.py` and edit it in order to make sure it reads the data
from `workspace/hyperopt/task_id/data/` and *importantly* that it computes a variable called `score` at the end that is
the number that the Hyperopt will try to **minimize**.

## Run the task

Now you can run the task. Make sure to set the `task_id`:

```bash
task_id=YOUR-TASK-ID
python -u src/agent/start.py task=hyperopt method=direct-hyperopt llm@agent.llm=openai/gpt3.5 task.task_id=$task_id
```

#### Note

If you want to see the prompts and responses of the Agent, you can use `HUMAN_SUPERVISE=1` at the beginning of the
command. But note that this will prevent from running automatically, and you will have to press `Continue` after each
interaction.