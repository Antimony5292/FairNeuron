import sys
sys.path.insert(1, "../")  
import numpy as np
# np.random.seed(0)

from aif360.datasets import GermanDataset,AdultDataset,CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing

from IPython.display import Markdown, display

# dataset_orig = GermanDataset(
#     protected_attribute_names=['sex'],           # this dataset also contains protected
#                                                  # attribute for "sex" which we do not
#                                                  # consider in this evaluation
#     # privileged_classes=[lambda x: x >= 25],      # age >=25 is considered privileged
#     # features_to_drop=['personal_status', 'age'] # ignore sex-related attributes
# )

# dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

# privileged_groups = [{'sex': 1}]
# unprivileged_groups = [{'sex': 0}]

# metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train, 
#                                              unprivileged_groups=unprivileged_groups,
#                                              privileged_groups=privileged_groups)


# display(Markdown("#### Original training dataset"))
# print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())

# RW = Reweighing(unprivileged_groups=unprivileged_groups,
#                 privileged_groups=privileged_groups)
# dataset_transf_train = RW.fit_transform(dataset_orig_train)
# df,_ = dataset_transf_train.convert_to_dataframe(de_dummy_code=True,set_category=False)
# test_df,_=dataset_orig_test.convert_to_dataframe(de_dummy_code=True,set_category=False)
# df.to_csv('./reweighting_train',index=False)
# test_df.to_csv('./reweighting_test',index=False)
# metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train, 
#                                                unprivileged_groups=unprivileged_groups,
#                                                privileged_groups=privileged_groups)
# display(Markdown("#### Transformed training dataset"))
# print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_transf_train.mean_difference())

# dataset_orig = AdultDataset(
#     instance_weights_name='fnlwgt',
#     features_to_drop=[]           # this dataset also contains protected
#                                                  # attribute for "sex" which we do not
#                                                  # consider in this evaluation
#     # privileged_classes=[lambda x: x >= 25],      # age >=25 is considered privileged
#     # features_to_drop=['personal_status', 'age'] # ignore sex-related attributes
# )

# dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

# privileged_groups = [{'sex': 1}]
# unprivileged_groups = [{'sex': 0}]

# metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train, 
#                                              unprivileged_groups=unprivileged_groups,
#                                              privileged_groups=privileged_groups)


# display(Markdown("#### Original training dataset"))
# print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())
 
# RW = Reweighing(unprivileged_groups=unprivileged_groups,
#                 privileged_groups=privileged_groups)
# dataset_transf_train = RW.fit_transform(dataset_orig_train)
# df,_ = dataset_transf_train.convert_to_dataframe(de_dummy_code=True,set_category=False)
# test_df,_=dataset_orig_test.convert_to_dataframe(de_dummy_code=True,set_category=False)
# df.to_csv('./reweighting_train',index=False)
# test_df.to_csv('./reweighting_test',index=False)
# metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train, 
#                                                unprivileged_groups=unprivileged_groups,
#                                                privileged_groups=privileged_groups)
# display(Markdown("#### Transformed training dataset"))
# print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_transf_train.mean_difference())

dataset_orig = CompasDataset(
    features_to_drop=[]           # this dataset also contains protected
                                                 # attribute for "sex" which we do not
                                                 # consider in this evaluation
    # privileged_classes=[lambda x: x >= 25],      # age >=25 is considered privileged
    # features_to_drop=['personal_status', 'age'] # ignore sex-related attributes
)

dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

privileged_groups = [{'race': 1}]
unprivileged_groups = [{'race': 0}]

metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)


display(Markdown("#### Original training dataset"))
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())
 
RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
dataset_transf_train = RW.fit_transform(dataset_orig_train)
df,_ = dataset_transf_train.convert_to_dataframe(de_dummy_code=True,set_category=False)
test_df,_=dataset_orig_test.convert_to_dataframe(de_dummy_code=True,set_category=False)
df.to_csv('./reweighting_train',index=False)
test_df.to_csv('./reweighting_test',index=False)
metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train, 
                                               unprivileged_groups=unprivileged_groups,
                                               privileged_groups=privileged_groups)
display(Markdown("#### Transformed training dataset"))
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_transf_train.mean_difference())