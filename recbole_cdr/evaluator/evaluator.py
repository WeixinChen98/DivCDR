
"""
recbole.evaluator.evaluator
#####################################
"""

from recbole_cdr.evaluator.register import metrics_dict
from recbole_cdr.evaluator.collector import DataStruct
from collections import OrderedDict

from logging import getLogger


class Evaluator(object):
    """Evaluator is used to check parameter correctness, and summarize the results of all metrics.
    """

    def __init__(self, config):
        self.config = config
        self.metrics = [metric.lower() for metric in self.config['metrics']]
        self.metric_class = {}

        for metric in self.metrics:
            self.metric_class[metric] = metrics_dict[metric](self.config)

        self.logger = getLogger()
        self.decimal_place = config['metric_decimal_place']


    def evaluate(self, dataobject: DataStruct):
        """Calculate all the metrics. It is called at the end of each epoch.

        Args:
            dataobject (DataStruct): It contains all the information needed for metrics.

        Returns:
            collections.OrderedDict: Contains overall, group1, and group2 evaluation results.
        """
        # Overall evaluation
        result_dict = OrderedDict()
        for metric in self.metrics:
            metric_val = self.metric_class[metric].calculate_metric(dataobject)
            result_dict.update(metric_val)

        
        # Group1 users evaluation
        group1_result_dict = OrderedDict()
        for metric in self.metrics:
            metric_val = self.metric_class[metric].calculate_metric(dataobject, group = "group1")
            group1_result_dict.update(metric_val)

        # Group2 users evaluation
        group2_result_dict = OrderedDict()
        for metric in self.metrics:
            metric_val = self.metric_class[metric].calculate_metric(dataobject, group = "group2")
            group2_result_dict.update(metric_val)

        # UGF evaluation
        ugf_result_dict = OrderedDict()
        for key in group1_result_dict.keys():
            # if key in group2_result_dict:  # Ensure the key exists in both dictionaries
            ugf = abs(group1_result_dict[key] - group2_result_dict[key])
            ugf_result_dict[key] = round(ugf, self.decimal_place)


        return {
            'overall': result_dict,
            'group1': group1_result_dict,
            'group2': group2_result_dict, 
            'ugf': ugf_result_dict
        }