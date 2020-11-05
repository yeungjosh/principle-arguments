# NLP Project


## [Paper](https://github.com/yeungjosh/principle-arguments/blob/main/NLP_Final_Report.pdf)


Datasets

The claim stance dataset includes stance annotations for claims
http://www.research.ibm.com/haifa/dept/vst/debating_data.shtml#Claim%20Stance

Classes of Principled Arguments
http://www.research.ibm.com/haifa/dept/vst/debating_data.shtml#Classes%20of%20Principled%20Arguments

Reason Identification and Classification Dataset
http://www.hlt.utdallas.edu/~saidul/stance/reason.html


## Running the project

`python3 main.py`

The program does not accept any arguments. 
It will run k-folds cross validation on all the models, which are defined starting on line 82.

To add or subtract models for running, comment/uncomment the corresponding lines in `main.py`.

To change the number of folds, change line 139:

```kf = KFold(5)```
