from src.data.dso import DatasetSubmissionObject

from src.metrics.metrics import Metrics 


class COLDAnalysis(object) : 

    @staticmethod
    def coarse_analysis(cold_submission:DatasetSubmissionObject) : 


        groundtruth, predictions = cold_submission.submission['Off'] , cold_submission.submission['preds']

        m = Metrics(groundtruth, predictions)
        return m.get_metrics_dictionary()


    @staticmethod
    def categorical_analysis(cold_submission:DatasetSubmissionObject, category:str) : 

        groundtruth, predictions, category_labels = cold_submission.submission['Off'][:-1] , cold_submission.submission['preds'][:-1] , cold_submission.submission[category][:-1]

        gt_dict = {}
        pt_dict = {}
        
        for gt, pt, ct in zip(groundtruth, predictions, category_labels) : 

            if ct in gt_dict : 
                gt_dict[ct].append(gt)
                pt_dict[ct].append(pt)
            else : 
                gt_dict[ct] = [gt]
                pt_dict[ct] = [pt]

        result_dict = {}
            
        for ct in gt_dict.keys() :
            m = Metrics(gt_dict[ct] , pt_dict[ct])  
            result_dict[ct] = m.get_metrics_dictionary()

        return result_dict

if __name__ == '__main__' : 

    from src.data.cold import COLD, COLDSubmissionObject
    from src.analysis.cold import COLDAnalysis
    import numpy as np

    cold = COLD()

    dataset = cold.data()

    num_preds = dataset.shape[0]
    yn_preds = np.random.choice(['Y' , 'N'], size=num_preds)
    bool_preds = np.random.choice([True, False], size=num_preds)

    map = {True : 'Y' , False:'N'}

    print('Yes-No Preds')

    submission = cold.submit(dataset, bool_preds, map=map)

    # print(COLDAnalysis.coarse_analysis(submission))

    print(COLDAnalysis.categorical_analysis(submission, category='Nom'))













