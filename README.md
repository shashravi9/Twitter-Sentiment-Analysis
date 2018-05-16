Instruction to run the following commands:
```
python collect.py
python cluster.py
python classify.py
python summarize.py
```

Here is what each script do:

- `collect.py`: This collects data used in your analysis. This may mean submitting queries to Twitter or Facebook API, or scraping webpages. The data should be raw and come directly from the original source -- that is, you may not use data that others have already collected and processed for you (e.g., you may not use [SNAP](http://snap.stanford.edu/data/index.html) datasets). Running this script should create a file or files containing the data that you need for the subsequent phases of analysis.
- `cluster.py`: This reads the data collected in the previous steps and use any community detection algorithm to cluster users into communities. This writes files that need in next phases. So I have saved the results.
- `classify.py`: This classifies your data along any dimension of your choosing sentiment. Results are saved in a file for further steps.
- `summarize.py`: This should read the output of the previous methods to write a textfile called `summary.txt` containing the following entries:
  - Number of users collected:
  - Number of messages collected:
  - Number of communities discovered:
  - Average number of users per community:
  - Number of instances per class found:
  - One example from each class:

Additionally, I created a plain text file called 'description.txt' that contains a brief summary of what my code does and conclusions of what I have made from the analysis.
