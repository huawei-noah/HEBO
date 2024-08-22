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

## Modify the Original Code (just a little bit)

Make sure the script reads the data from your local workspace by adding `FILEPATH` at the beginning, e.g.

```python
FILE_PATH = "./workspace/hyperopt/abalone/data/"
...
df_train = pd.read_csv(FILE_PATH + "train.csv", index_col="id")
df_test = pd.read_csv(FILE_PATH + "test.csv", index_col="id")
...
```

## Run the task

Now you can run the task. Make sure to set the `task_id`:

```bash
task_id=YOUR-TASK-ID
python -u src/agent/start.py task=hyperopt method=direct-hyperopt llm@agent.llm=openai/gpt3.5 task.task_id=$task_id
```

## Run All Tasks

There is python script that will launch multiple subprocesses in parallel for running all the specified tasks and seeds.
This script assumes all workspaces are ready to be run. To launch it, there is a shell script as well that you can run
as follows from the `Agent` root directory:

```shell
bash -x scripts/run-simple.sh 
```

## Add a Reflection Strategy

It's easy to add a different way to do reflection. Simply add a new
prompt `src/agent/prompts/templates/hyperopt/reflection_strategy/new_reflection.jinja` and specify its name in the
configuration `configs/method/direct-hyperot.yaml`. Finally, run the shell script as

```shell
bash -x scripts/run-simple.sh new_reflection
```

#### Note

If you want to see the prompts and responses of the Agent, you can use `HUMAN_SUPERVISE=1` at the beginning of the
command. But note that this will prevent from running automatically, and you will have to press `Continue` after each
interaction.