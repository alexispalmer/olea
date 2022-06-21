# OLEAT (Offensive Language Error Analysis Tool)

## About OLEAT
Hate speech and offensive language detection models can benefit from in-depth error analysis, more than just an F1 score, but many systems lack any extensive error analysis. To address this issue, we present OLEAT, an extensible tool that provides researchers further insight into the performance of their offensive language detection model on different datasets. 

The datasets currently available with OLEAT:

- **COLD (Complex Offensive Language Dataset )** - The COLD data set is intended for researchers to diagnose and assess their automatic hate speech detection systems. The corpus highlights 4 different types of complex offensive language: slurs, reclaimed slurs, adjective nominalization, distancing, and also non-offensive texts. (Palmer et al., 2020)
- (Coming Soon: Any dataset)

## Local Installation
### Requirements
(Insert requirements for installation)

### Install
(Insert code for installation)
```sh

```


## Getting Started
The user provides a pre-trained hate speech detection model and predicts it on an OLEAT-supported dataset. The user can then apply different analyses to their predictions to gain insight into what cases their model fails on. 

(Insert code for generating a report)
```sh

```
1. Create an instance of the Dataset class?
2. Predict on a dataset
3. Choose an analysis for a subset of data


## Analysis
OLEAT provides generic analysis that can be applied to any NLP classification task, by evaluating performance based on a subset of the data. This can be applied to text length, and text containing certain strings (coming soon:  regex expressions). OLEAT also provides analysis specific to COLD, showing model performance on different levels of annotator agreement of offensiveness and analysis of the fine-grained categories outlined in Palmer et al. The analysis provides metrics of F1, precision, and recall for each subset of data. 


## Contact
Please contact Alexis Palmer (Alexis.Palmer@colorado.edu), Dananjay Srinivas (Dananjay.Srinivas@colorado.edu), Marie Grace (Marie.Grace@colorado.edu), or Jay Seabrum (Xajavion.Seabrum@colorado.edu)

## Resources

Palmer, A., Carr, C., Robinson, M., & Sanders, J. (2020). COLD: Annotation scheme and evaluation data set for complex oﬀensive language in English. 28.




