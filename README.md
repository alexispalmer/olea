# OLEA (Offensive Language Error Analysis)

## About OLEA
Hate speech and offensive language detection models can benefit from in-depth error analysis, more than just an F1 score, but many systems lack any extensive error analysis. To address this issue, we present OLEA, an extensible tool that provides researchers further insight into the performance of their offensive language detection model on different datasets. 

The datasets currently available with OLEA:

- **COLD (Complex Offensive Language Dataset )** - The COLD data set is intended for researchers to diagnose and assess their automatic hate speech detection systems. The corpus highlights 4 different types of complex offensive language: slurs, reclaimed slurs, adjective nominalization, distancing, and also non-offensive texts. (Palmer et al., 2020)
- **HateCheck** - HateCheck is a suite of functional tests for hate speech detection models that enable more targeted diagnostic insights. It specifies 29 model functionalities motivated by a review of previous research and a series of interviews with civil society stakeholders with test cases for each functionality. (Röttger et al., 2021)
- **HateXplain** - Each post in this dataset is annotated from three different perspectives: the basic, commonly used 3-class classification (i.e., hate, offensive or normal), the target community (i.e., the community that has been the victim of hate speech/offensive speech in the post), and the rationales, i.e., the portions of the post on which their labelling decision (as hate, offensive or normal) is based. (Matthew et al., 2022)

## Local Installation
### Requirements
```
'numpy>1.21.0' , 
'scipy>1.6.0' , 
'datasets>2.2.0' , 
'matplotlib>3.0' , 
'pandas>1.2.0' , 
'Pillow>8.0.0' , 
'scikit-learn>1.0' , 
'emoji>1.0',
'wordsegment>1.3'
```

### Install
```sh
pip install olea
```

## Getting Started
The user provides a pre-trained hate speech detection model and predicts it on an OLEA-supported dataset. The user can then apply different analyses to their predictions to gain insight into what cases their model fails on. Consider this introductory example

1. Import Statements
```sh
from olea.data.cold import COLD, COLDSubmissionObject
from olea.analysis.cold import COLDAnalysis

#import statements for downloading example model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TextClassificationPipeline
```
2. Downloading the data
```sh
#Load Dataset
cold = COLD()

#Load Model
link = "Hate-speech-CNERG/bert-base-uncased-hatexplain"
tokenizer = AutoTokenizer.from_pretrained(link)
model = AutoModelForSequenceClassification.from_pretrained(link)
```
3. Predicting on the dataset (Example model is HateXplain downloaded from HuggingFace)
```sh
#Predict on COLD
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)
preds = pd.DataFrame(pipe(list(cold.data()["Text"]))).label
```
4. Define a Mapping and Create Submission Object
```sh
hate_map = {"offensive": 'Y' , "hate speech": 'Y' , "normal":'N'}
submission = cold.submit(cold.data(), preds, map=hate_map)
```
5. Choose an analysis 
```sh
plot_info, metrics = COLDAnalysis.analyze_on(submission,'Cat',show_examples = True)
```

## Analysis
OLEA provides generic analysis that can be applied to any NLP classification task, by evaluating performance based on a subset of the data. This can be applied to text length, and text containing certain strings, and text determined to be written in AAVE (Blodgett et al., 2016). OLEA also provides analysis specific to COLD, showing model performance on different levels of annotator agreement of offensiveness and analysis of the fine-grained categories outlined in Palmer et al. The analysis provides metrics of F1, precision, and recall for each subset of data. 

Generic Analysis includes:

-   `analyze_on` for evaluating model performance on any specified
    categorical column.

-   `check_substring` for evaluating model performance on presence of a
    specified substring in text

-   `aave` for evaluating how the model predicts on instances that are
    written using African American Vernacular English. The scores are
    calculated using the TwitterAAE model
    (Blodgett et al., 2016). These scores represent an
    inference of the proportion of words in the text that come from a
    demographically-associated language/dialect.

-   `str_len_analysis` for evaluating how the model performs on
    instances of different character or word lengths using a histogram.

-   `check_anno_agreement` for evaluating model performance on
    instances with full annotator agreement on the offensiveness of a
    text (\"Y\",\"Y\",\"Y\") or (\"N\",\"N\",\"N\") vs instances with
    partial agreement. This should indicate \"easy\" (full) vs
    \"difficult\" (partial) cases.

The COLD-specific analysis includes:

-   `analyze_on` for evaluating model performance on the COLD specific
    categories outlined in (Palmer et al., 2020). These categories are
    constructed from offensiveness, presence of adjectival
    nomanilization, presence of slur, and presence of linguistic
    distancing.

The HateCheck-specifc analysis includes:

-   `analyze_on` for evaluating model performance on the HateCheck
    specific categories outlined in (Röttger et al., 2021). Some
    categories included are negation, counter, derogation, and
    profanity.



## Contact
Marie Grace, Jay Seabrum, Dananjay Srinivas, and Alexis Palmer all contributed to this library. 
Please contact olea.ask@gmail.com for further inquiries. 

## Resources

Blodgett, S. L., Green, L., & O’Connor, B. (2016). Demographic Dialectal Variation in Social Media: A Case Study of African-American English. Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, 1119–1130. https://doi.org/10.18653/v1/D16-1120

Mathew, B., Saha, P., Yimam, S. M., Biemann, C., Goyal, P., & Mukherjee, A. (2022). HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection (arXiv:2012.10289). arXiv. http://arxiv.org/abs/2012.10289

Palmer, A., Carr, C., Robinson, M., & Sanders, J. (2020). COLD: Annotation scheme and evaluation data set for complex oﬀensive language in English. 28.

Röttger, P., Vidgen, B., Nguyen, D., Waseem, Z., Margetts, H., & Pierrehumbert, J. B. (2021). HateCheck: Functional Tests for Hate Speech Detection Models. Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), 41–58. https://doi.org/10.18653/v1/2021.acl-long.4
