# FairNeuron

## TL;DR

A model fairness fixing frameworks which avoids training an adversary model and does not require modifying model training protocol or architecture.

## Repo Structure

```
- data/
- FN/
    - utils/
    - EA.py
    - Evaluate.py
    - FairNeuron.py
    - models.py
    - run.py
- README.md
- requirements.txt
```

## Demo

```shell
$ pip install -r requirements.txt
$ cd ./FN/
$ python run.py 
```

## Results
### Effectiveness
![Effectiveness](https://github.com/Antimony5292/MyFigs/blob/main/ICSE22/effect1.png)

To evaluate the effectiveness of FairNeuron, we test the following models: the naive baseline model, models fixed by FAD, by Ethical Adversaries, and by FairNeuron. Due to the randomness in these experiments, we ran the training 10 times to ensure the reliability of results and enforced these fixing algorithms to share the same original training dataset. To measure the effectiveness of FairNeuron, we compare the performance between FairNeuron and the other algorithms in terms of both utility and fairness. To demonstrate the effectiveness of the three components of FairNeuron (i.e. neural network slicing, sample clustering and selective training), we conducted a detailed comparison between our algorithm and other popular works.

This table shows the average DPR improvement of naive models, which are 198.47%, 257.23%, and 3995.23% of Census, Credit and COMPAS, respectively. FairNeuron mitigates the naive model by 69.69%, 21.12% and 38.95% in terms of EO, and 74.68%, 2.08%, 96.19% in terms of DP. Compared with the other algorithms, FairNeuron achieves the best fairness performance on Census and COMPAS. 

### Efficiency
![Efficiency](https://github.com/Antimony5292/MyFigs/blob/main/ICSE22/effi1.png)

To evaluate the efficiency of FairNeuron, we measured the time usage of ordinary training, Ethical Adversaries and FairNeuron training on all three datasets. We conducted the experiments 10 times and computed the average overhead. For ordinary training, the runtime overhead all comes from the training procedure, but for FairNeuron, the hyperparameters tuning accounts for a larger ratio of the total time usage, as shown in above figure. FairNeuron takes only less than twice of the time usage of ordinary training on large datasets like Census, but on small datasets like the German Credit dataset, it takes relatively a long time. If FairNeuron tries more times, the average time will be reduced because the hyperparameters tuning is only conducted once. Overall, FairNeuron is more efficient than Ethical Adversaries in fixing models, with an average speedup of 180%.