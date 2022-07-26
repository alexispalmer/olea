from olea.data.dso import DatasetSubmissionObject



def check_substring(substring:str, submission:DatasetSubmissionObject) : 

    data, groundtruth, prediction = submission.filter_submission('Text', lambda x : substring in x)
    return data, groundtruth, prediction

    

