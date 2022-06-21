from src.data.dso import DatasetSubmissionObject


class Generic(object) : 

    @staticmethod
    def check_substring(substring:str, submission:DatasetSubmissionObject) : 
        data, groundtruth, prediction = submission.filter_submission('Text', lambda x : substring in x)
        return data, groundtruth, prediction

    @staticmethod
    def str_len_analysis(submission:DatasetSubmissionObject) : 
        pass


