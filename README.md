# FairNeuron

### [Pre-print](https://arxiv.org/abs/2204.02567) | [Proceeding](https://ieeexplore.ieee.org/document/9793993) | [Video](https://youtu.be/pAe4os3JjA0) | Tutorial . 

## TL;DR

FairNeuron is a deep neural network (DNN) model automatic repairing tool, which avoids training an adversary model and does not require modifying model training protocol or architecture.

## Repo Structure

1. data: It contains three datasets we used in this paper, and original results we got.

2. FN: It contains the source codes of this work and a demo. The way to run the demo cases has been shown [here](#setup).

4. misc.: The `FairNeuron.ipynb` is the original code of this paper. The `README.md` shows how to use the our demos, the repo structure, the way to reproduce our experiments and our experiment results. And the `requirement.txt` shows all the dependencies of this work.

```
- data/
- FN/
    - utils/
    - EA.py
    - Evaluate.py
    - FairNeuron.py
    - models.py
    - run.py
- FairNeuron.ipynb
- README.md
- requirements.txt
```


## Results
### Effectiveness
![Rebuttal](./appendix.png)

To evaluate the effectiveness of FairNeuron, we test the following models: the naive baseline model, models fixed by FAD, by Ethical Adversaries, and by FairNeuron. We also compared FairNeuron with two popular pre-/post-processing method, reweighing and reject option classification (ROC). Due to the randomness in these experiments, we ran the training 10 times to ensure the reliability of results and enforced these fixing algorithms to share the same original training dataset. To measure the effectiveness of FairNeuron, we compare the performance between FairNeuron and the other algorithms in terms of both utility and fairness. 

This table shows the average DPR improvement of naive models, which are 198.47%, 257.23%, and 3995.23% of Census, Credit and COMPAS, respectively. FairNeuron mitigates the naive model by 69.69%, 21.12% and 38.95% in terms of EO, and 74.68%, 2.08%, 96.19% in terms of DP. Compared with the other algorithms, FairNeuron achieves the best fairness performance on Census and COMPAS. 

### Efficiency
![Efficiency](./effi1.png)

To evaluate the efficiency of FairNeuron, we measured the time usage of ordinary training, Ethical Adversaries and FairNeuron training on all three datasets. We conducted the experiments 10 times and computed the average overhead. For ordinary training, the runtime overhead all comes from the training procedure, but for FairNeuron, the hyperparameters tuning accounts for a larger ratio of the total time usage, as shown in above figure. FairNeuron takes only less than twice of the time usage of ordinary training on large datasets like Census, but on small datasets like the German Credit dataset, it takes relatively a long time. If FairNeuron tries more times, the average time will be reduced because the hyperparameters tuning is only conducted once. Overall, FairNeuron is more efficient than Ethical Adversaries in fixing models, with an average speedup of 180%.

## <span id="setup">Setup</span>
### (Recommended) Create a virtual environment
FairNeuron requires specific versions of some Python packages which may conflict with other projects on your system. A virtual environment is strongly recommended to ensure the requirements may be installed safely.

### Install with `pip`
To install the requirements, run:

`pip install -r ./requirements.txt`

Note: The version of Pytorch in this requirements is CPU version. If you need GPU version, please check your CUDA version and [install Pytorch manually](https://pytorch.org/).

### Run demo

```shell
cd ./FN
python run.py
```

The optional Args are:

| Argument     | Help                                                    | Default                                                      |
| ------------ | ------------------------------------------------------- | ------------------------------------------------------------ |
| --dataset    | Choose dataset. Include "compas", "census" and "credit" | compas                                                       |
| --epoch      | Training epoch for network                              | 10                                                           |
| --batch-size | Batch size of dataloader                                | 128                                                          |
| --input-path | Path of dataset                                         | ../data/COMPAS/compas_recidive_two_years_sanitize_age_category_jail_time_decile_score.csv |
| --save-dir   | Path to save results                                    | ./results                                                    |
| --rand       | Determine whether dropout randomly                      | False                                                        |



# Citation

If you find our work useful in your research, please consider citing:

```
@INPROCEEDINGS{9793993,
  author={Gao, Xuanqi and Zhai, Juan and Ma, Shiqing and Shen, Chao and Chen, Yufei and Wang, Qian},
  booktitle={2022 IEEE/ACM 44th International Conference on Software Engineering (ICSE)}, 
  title={Fairneuron: Improving Deep Neural Network Fairness with Adversary Games on Selective Neurons}, 
  year={2022},
  pages={921-933},
  doi={10.1145/3510003.3510087}}
```



# Misc

Please open an issue if there is any question.

