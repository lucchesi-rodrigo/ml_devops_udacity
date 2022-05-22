import os
import logging as log
from churn_library import import_data
from loguru import logger
import pytest

log.basicConfig(
    filename='./logs/churn_library_tests.log',
    level = log.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

class TestPredictChurn:

	def test_import_working(self):
		""" test data import - this example is completed for you to assist with the other test functions """
		df = import_data(df_path='data/test_data.csv')
		assert df.shape[0] > 0

	def test_data_loading_exception(self):
		"""Invalid path"""
		with pytest.raises(FileNotFoundError):
			_= import_data(df_path='data/test_data_test.csv')

    # def test_data_loading(self):
    #     """Loads csv file"""
    #     mlm = MlModeling(model_name='lrc', model_algorithm=LogisticRegression(), model_version='0.1')
    #     mlm.data_loading('tests/data/data.csv')
    #     assert mlm.df.columns.to_list() == ['x','y']

    # def test_eda(self):
	#     '''
	#     test perform eda function
	#     '''


    # def test_encoder_helper(self):
	#     '''
	#     test encoder helper
	#     '''


    # def test_perform_feature_engineering(self):
	#     '''
	#     test perform_feature_engineering
	#     '''


    # def test_train_models(train_models):
	# '''
	#     test train_models
	# '''
