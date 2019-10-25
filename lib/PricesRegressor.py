import pandas as pd
import xgboost as xgb
import numpy as np
import datetime

TEST_DF_PATH = '../data/TP2/train.csv'
ERROR_MSG = 'Unable to serialize result ' + self.__result

class PricesRegressor:

	def __init__(self, test_df = pd.read_csv(TEST_DF_PATH), train_df = pd.read_csv(TRAIN_DF_PATH)):
		self.test_df = test_df
		self.train_df = train_df
		self.regressor = None
		self.__result = None

	def train_and_predict(self, n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=6, objective='reg:squarederror'):

		regressor = xgb.XGBRegressor(self, n_estimators=n_estimators, learning_rate=learning_rate,
																gamma=gamma, subsample=subsample, colsample_bytree=colsample_bytree,
																max_depth=max_depth, objective=objective)

		training_data = self.train_df.drop(columns=['precio']).values
		training_target = self.train_df['precio'].values
		regressor.fit(training_data, training_target)

		self.__result = regressor.predict(self.test_df.values)
		return self.__result

	def serialize_result(dir = '../res/reg/'):
		try:
			self.__result.to_csv(self.__export_path(dir), index=False, header=True )
		except AttributeError:
			print(ERROR_MSG)

	def __export_path(export_path):
		filename = 'regression_' + datetime.datetime.now().strftime("%m-%d-%Y") + '.csv'
		return (export_path + filename)
