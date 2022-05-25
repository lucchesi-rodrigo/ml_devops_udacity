import os
import pandas as pd
import numpy as np
import logging as log
from churn_library import (
	import_data,create_visual_eda,create_stats_info,PlotNotAllowedError,
	CreateVisualEdaError,CreateStatsInfoError,EncoderHelperError, FeatureEngineeringError,
	perform_eda_pipeline,encoder_helper, perform_feature_engineering
	)
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

	def test_import_working_exception(self):
		"""Invalid path"""
		with pytest.raises(FileNotFoundError):
			_= import_data(df_path='data/test_data_test.csv')

	def test_create_visual_eda(self):
		"""Test create_visual_eda method"""
		df = import_data(df_path='data/test_data.csv')
		assert create_visual_eda(plot_type='histogram', df=df,col='x')

	def test_create_visual_eda_not_working_wrong_plot_type(self):
		"""Test create_visual_eda PlotNotAllowedError exception"""
		with pytest.raises(PlotNotAllowedError):
			df = import_data(df_path='data/test_data.csv')
			create_visual_eda(plot_type='line-plos-2d', df=df,col='x')

	def test_create_visual_eda_not_working_base_exception(self):
		"""Test create_visual_eda PlotNotAllowedError exception"""
		with pytest.raises(CreateVisualEdaError):
			df = import_data(df_path='data/test_data.csv')
			create_visual_eda(plot_type='histogram', df=df,col='o')

	def test_create_stats_info(self):
		"""Test create_stats_info working"""
		df = import_data(df_path='data/test_data.csv')
		assert create_stats_info(df=df,stats_calc=True)
		assert df.shape == (5, 3)

	def test_create_stats_info_not_working_base_exception(self):
		"""Test create_visual_eda PlotNotAllowedError exception"""
		with pytest.raises(CreateStatsInfoError):
			df = pd.DataFrame()
			create_stats_info(df=df, stats_calc=True)

	def test_perform_eda_pipeline_working(self):
		"""Test perform_eda_pipeline"""
		df = import_data(df_path='data/test_data.csv')
		assert perform_eda_pipeline(
			plot_type='histogram', 
			df=df,
			col='x', 
			stats_calc=True
			)

	def test_perform_eda_pipeline_not_working_base_exception(self):
		"""Test perform_eda_pipeline"""
		with pytest.raises(CreateVisualEdaError):
			df = import_data(df_path='data/test_data.csv')
			perform_eda_pipeline(
				plot_type='histogram', 
				df=df,
				col='o', 
				stats_calc=True
				)

	def test_encoder_helper_with_categoric_lst(self):
		df = pd.DataFrame(
			[
				("bird", "Falconiformes", 389.0),
				("bird", "Psittaciformes", 24.0),
				("mammal", "Carnivora", 80.2),
				("mammal", "Primates", np.nan),
				("mammal", "Carnivora", 58),
			],
			index=["falcon", "parrot", "lion", "monkey", "leopard"],
			columns=("class", "order", "max_speed"),
		)
		col_name='order'
		target_col='max_speed'

		df_new = encoder_helper(
			df=df,
			target_col=target_col,
			categoric_cols=['class','order']
			)
		
		assert sorted(df_new.columns.tolist()) == sorted(['class', 'class_max_speed', 'max_speed', 'order', 'order_max_speed'])

	def test_encoder_helper_without_categoric_lst(self):
		df = pd.DataFrame(
			[
				("bird", "Falconiformes", 389.0),
				("bird", "Psittaciformes", 24.0),
				("mammal", "Carnivora", 80.2),
				("mammal", "Primates", np.nan),
				("mammal", "Carnivora", 58),
			],
			index=["falcon", "parrot", "lion", "monkey", "leopard"],
			columns=("class", "order", "max_speed"),
		)
		col_name='order'
		target_col='max_speed'

		df_new = encoder_helper(
			df=df,
			target_col=target_col
			)
		
		assert sorted(df_new.columns.tolist()) == sorted(['class', 'class_max_speed', 'max_speed', 'order', 'order_max_speed'])

	def test_encoder_helper_without_exception(self):
		with pytest.raises(EncoderHelperError):
			target_col='max_speed'
			df = None
			_= encoder_helper(
				df=df,
				target_col=target_col
				)
    
	def test_perform_feature_engineering_working(self):
		"""Test """
		df = pd.DataFrame(
			[
				("bird", "Falconiformes", 389.0),
				("bird", "Psittaciformes", 24.0),
				("mammal", "Carnivora", 80.2),
				("mammal", "Primates", 0),
				("mammal", "Carnivora", 58),
			],
			index=["falcon", "parrot", "lion", "monkey", "leopard"],
			columns=("class", "order", "max_speed"),
		)
		x_cols=["class", "order"]
		y_cols=['max_speed']

		X_train, X_test, y_train, y_test = perform_feature_engineering(
			df=df,
			x_cols=x_cols,
			y_cols= y_cols
		)
		assert X_train.shape[0] > 1
		assert X_test.shape[0] > 1
		assert y_train.shape[0] > 1
		assert y_test.shape[0] > 1

	def test_perform_feature_engineering_exception(self):
		with pytest.raises(FeatureEngineeringError):
			df = pd.DataFrame(
				[
					("bird", "Falconiformes", 389.0),
					("bird", "Psittaciformes", 24.0),
					("mammal", "Carnivora", 80.2),
					("mammal", "Primates", 0),
					("mammal", "Carnivora", 58),
				],
				index=["falcon", "parrot", "lion", "monkey", "leopard"],
				columns=("class", "order", "max_speed"),
			)
			x_cols=["class", "order"]
			y_cols=['max_speed']

			_= perform_feature_engineering(
				df=df,
				x_cols=x_cols,
				y_cols= y_cols,
				test_size=0.9,
			)