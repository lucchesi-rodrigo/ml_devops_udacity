"""
author: Rodrigo Lucchesi
date: May 2022
"""
# library doc string
# import libraries
import os
import pickle
from datetime import datetime
import traceback
import logging as log
import pandas as pd
import churn_library
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from loguru import logger
log.basicConfig(
    filename='./logs/churn_library.log',
    level = log.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

class PlotNotAllowedError(Exception):
    """Custom exception for plot not allowed on EDA"""
    pass
class PlotNotAllowedError(Exception):
    """Custom exception for plot not allowed on EDA"""
    pass
class CreateVisualEdaError(Exception):
    """Custom exception for create visual eda"""
    pass


def import_data(df_path: str) -> pd.DataFrame:
    """
    Returns dataframe for the csv found at path
    
    Parameters
    ----------
    df_path: str
        A path to the csv file
        
    Returns:
    --------
    df: pd.DataFrame
        A pandas dataframe
        
    Examples:
    ---------
        >>> df = import_data(df_path='path/file.csv')
    """	
    try:
        df = pd.read_csv(df_path)
    except BaseException as exc:
        logger.error(f'(ERROR import_data({df_path}) -> Exception: {exc}!')
        raise FileNotFoundError(f'Could not read data from inserted path: {df_path}!')
    logger.info(f'(SUCCESS import_data({df_path}) -> msg: DataFrame read successfully -> df: {df.head().to_dict()}')
    return df

def create_visual_eda(plot_type:str,df:pd.DataFrame,col:str) -> bool:
    """
    Create images to visual inspection on eda pipeline

    Parameters
    ----------
    plot_type: str
        Plot type
    df: pd.DataFrame
        Dataframe to be used in ml project
    col: str
        Column name to be used on plotting
    
    Returns:
    --------
    bool: Tag to return if process was processed

    Examples:
    --------
    create_visual_eda(plot_type='histogram',df=df)

    """
    logger.info(f'(SUCCESS perform_eda_pipeline.create_visual_eda) -> msg: Starting process -> params -> plot_type: {plot_type}, df: {df.head().to_dict()}, col: {col}')
    try:
        plt.figure(figsize=(20,10))
        import pdb;pdb.set_trace()
        if plot_type not in ['histogram','normalized_barplot','distplot','heatmap']:
            logger.warning(
                '(WARNING perform_eda_pipeline.visual_eda) -> msg: Finishing process. Plot not allowed! -> '
                f'params -> plot_type: {plot_type}, df: {df.head().to_dict()}, col: {col}'
            )
            raise PlotNotAllowedError('Plot not allowed on visual_eda method!')
        if plot_type == 'histogram':
            df[f'{col}'].hist()
        elif plot_type == 'normalized_barplot':
            df.col.value_counts('normalize').plot(kind='bar')
        elif plot_type == 'distplot':
            sns.distplot(df[f'{col}'])
        elif plot_type == 'heatmap':
            sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    except BaseException as exc:
        logger.error(
        '(ERROR  perform_eda_pipeline.visual_eda) -> msg: Finishing process -> params -> plot_type: {plot_type}, df: {df.head().to_dict()}, col: {col}'
        f'Exception {exc}'
        )
        raise CreateVisualEdaError('perform_eda_pipeline.create_visual_eda problem!')
    exec_time = datetime.now()
    plt.savefig(f"images/{plot_type}-{exec_time}.jpg")
    logger.info(
        f'(SUCCESS perform_eda_pipeline.create_visual_eda) -> msg: Finishing process. {plot_type} plot created and saved at /images -> params -> plot_type: {plot_type}, df: {df.head().to_dict()}, col: {col}'
        )
    return 1

def create_stats_info(df:pd.DataFrame,stats_calc:bool=False) -> None:
    """
    Create statistic analysis on eda pipeline

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to be used in ml project
    stats_calc: bool
        Flag to execute and save the statistical analysis
    
    Returns:
    --------
    None

    Examples:
    --------
    create_stats_info(df=df,stats_calc=False)

    """
    logger.info(
        f'(SUCCESS perform_eda_pipeline.create_stats_info) -> msg: Starting process ->  params -> df:{df.head().to_dict()}, stats_calc:{stats_calc}'
        )
    if stats_calc:
        stats_data = {
            'shape': df.shape,
            'null_vals': df.isnull().sum(),
            'stats_desccription': df.describe().to_dict()
        }
        logger.info(
            f'(SUCCESS perform_eda_pipeline.create_stats_info) -> msg: Created stats data! ->  params -> df:{df.head().to_dict()}, stats_calc:{stats_calc}'
        )
    now = datetime.now()
    with open(f"data_{now}.pickle", "wb") as output_file:
        pickle.dump(stats_data, output_file)
    logger.info(
        f'(SUCCESS perform_eda_pipeline.create_stats_info) -> msg: Finishing process ->  params -> df:{df.head().to_dict()}, stats_calc:{stats_calc}'
        )
    return 1
    
def perform_eda_pipeline(**kwargs) -> None:
    """
    Perform eda pipeline on df and save figures to images folder
    
    Keyword Arguments
    ----------
    plot_type: str
        Plot type
    df: pd.DataFrame
        Dataframe to be used in ml project
    col: str
        Column name to be used on plotting
    stats_calc: bool
        Flag to execute and save the statistical analysis
        
    Returns:
    --------
    None
        
    Examples:
    ---------
    perform_eda_pipeline(
        plot_type='histogram',
        df=df,
        col = 'Churn'
        stats_calc = False
        )
    """
    plot_type = kwargs.get('plot_type')
    df = kwargs.get('df')
    col = kwargs.get('col')  
    stats_calc= kwargs.get('stats_calc',False)
    log.info(
        f'(SUCCESS perform_eda_pipeline) -> msg: Starting process ->  kwargs: {kwargs}'
    )
    
    create_visual_eda(
        plot_type = plot_type,
        df = df,
        col=col
    )
    
    create_stats_info(df=df,stats_calc=stats_calc)
    
    log.info(
        f'(SUCCESS perform_eda_pipeline) -> msg: Finishing process ->  kwargs: {kwargs}'
    )
    return

def encoder_helper(df: pd.DataFrame, target_col:str,response:str='',category_lst:List=[]) -> pd.DataFrame:
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
    
        
            df: pandas dataframe with new columns for
    '''
    log.info(
        f'(SUCCESS encoder_helper) -> Starting process'
    )
    try:
        if not categoric_cols:
            categoric_cols = [
                col for col in df.columns if df[column].dtype == 'object']
            log.info(
                f'(SUCCESS encoder_helper) -> Retrieving categoric cols from df: {categoric_cols}'
            )
        for col in categoric_cols:
            category_groups = df.groupby(col).mean()[target_col]
            for val in df[col]:
                category_lst.append(category_groups.loc[val])

            df[f'{col_name}_{target_col}'] = category_lst
            
        log.info(
            f'(SUCCESS encoder_helper) -> Finishing process. New df column {col_name}_{target_col} created!'
        )
    except BaseException as exc:
        log.error(
            f'(ERROR encoder_helper) -> Finishing process -> Exception: {exc}'
        )
        raise EncoderHelperException('Error during encoder_helper execution! ')
    return df

def perform_feature_engineering(**kwargs):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    log.info(
        f'(SUCCESS perform_feature_engineering) -> Starting process -> kwargs: {kwargs}'
    )
    try:
        target_col = kwargs.get('target_col')
        response = kwargs.get('response',None)
        keep_cols = kwargs.get('keep_cols')
        test_size = kwargs.get('test_size',0.3)
        
        y = df[f'{target_col}']
        X = pd.DataFrame()
        X[keep_cols] = df[keep_cols]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=11)
    except BaseException as exc:
        log.error(
            f'(ERROR perform_feature_engineering) -> Finishing process -> kwargs: {kwargs} -> Exception: {exc}'
        )
        raise FeatureEngineeringError('Feature engineering can not be executed')
    
    log.info(
            f'(SUCCESS perform_feature_engineering) -> Finishing process. Feature engineering executed! -> kwargs: {kwargs}'
        )
    return X_train, X_test, y_train, y_test

def classification_report_image(**kwargs):
    """
    Produces classification report for training and testing results and stores report as image
    in images folder
    Keyword arguments
    -----------------
        y_train (pd.DataFrame): training response values
        y_test (pd.DataFrame):  test response values
        y_train_preds (pd.DataFrame): training predictions from logistic regression
        y_test_preds (pd.DataFrame): test predictions from random forest
        model_name (str): Model algorithm name

    Returns:
        None
    """
    try:
        y_train = kwargs.get('y_train')
        y_test = kwargs.get('y_test')
        y_train_preds = kwargs.get('y_train_preds')
        y_test_preds = kwargs.get('y_test_preds')
        model_name = kwargs.get('model_name')
        log.info(
                f'(SUCCESS classification_report_image) -> Starting process -> kwargs: {kwargs}'
            )
        plt.figure(figsize=(5, 5))
        plt.text(
            0.01, 1.25, model_name, {
                'fontsize': 10}, fontproperties='monospace')
        plt.text(
            0.01, 0.05, str(
                classification_report(
                    y_test, y_test_predicted)), {
                'fontsize': 10}, fontproperties='monospace')
        plt.text(
            0.01, 0.6, model_name, {
                'fontsize': 10}, fontproperties='monospace')
        plt.text(
            0.01, 0.7, str(
                classification_report(
                    y_train, y_train_predicted)), {
                'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')
        plt.savefig(f'reports/{model_name}_report.pdf')
    except BaseException as exc:
        log.error(
            f'(ERROR classification_report_image) -> Finishing process -> Exception: {traceback.format_exc()}'
        )
        raise ClassificationReportImageError('Could not process classification report')
        
    log.info(
        f'(SUCCESS classification_report_image) -> Finishing process. Classification report created at repoorts folder! -> kwargs: {kwargs}'
    )
    return


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    def feature_importance_plot_1(model_data: Dict = None):
        """
        Generates a matplotlib bar plot to describe feature importance on
        X matrix targeting dimensionality reduction to avoid overfitting
        and decrease model complexity -> Matplotlib backend
        Parameters
        ----------
        model_data: Dict
            Machine learning model data
        Returns:
        --------
        None
        
        Examples:
        ---------
            >>> model = mlvc.CreateMlModel('test')
            >>> model.data_loading('db/data.csv')
            >>> model.test_train_data_split(test_size= 0.3,random_state= 11)
            >>> model_data = fit_predict_to_best_estimator(model_name='rfc',model_algorithm='RandomForestClassifier(),param_grid= {data ...},folds= 5)
            >>> model.feature_importance_plot_1(model_data=model_data)
        """
        model_data = kwargs.get('model_data')
        try:
            log.info(
                f'(SUCCESS feature_importance_plot_1) -> Starting process -> kwargs: {kwargs}'
            )
            model = model_data['model']
            importance = model.best_estimator_.feature_importances_
            indices = np.argsort(importance)[::-1]
            names = [self.X.columns[i] for i in indices]

            plt.figure(figsize=(20, 5))

            plt.title("Feature Importance")
            plt.ylabel('Importance')
            plt.bar(range(self.X.shape[1]), importance[indices])
            plt.xticks(range(self.X.shape[1]), names, rotation=90)
            plt.savefig(f'plots/feature_importance/feature_importance_plot1_{model_name}.pdf')
        except BaseException as exc:
            log.error(
                f'(ERROR feature_importance_plot_1) -> Finishing process -> kwargs: {kwargs} -> '
                f'Exception: {traceback.format_exc()}'
            )
            raise FeatureImportancePlot1Error(f'Feature importance 1 could not be processed. kwargs: {kwargs}')
        log.info(
            f'(SUCCESS feature_importance_plot_1) -> Starting process -> kwargs: {kwargs}'
        )
        return

    def feature_importance_plot_2(model_data: Dict = None):
        """
        Generates a matplotlib bar plot to describe feature importance on
        X matrix targeting dimensionality reduction to avoid overfitting
        and decrease model complexity -> Shap backend
        Parameters
        ----------
        self: CreateMlModel
            Create model object data
        model_data: str
            Machine learning model data
        Returns:
        --------
        shap: Shap
            Figure with feature importance information
        Examples:
        ---------
            >>> model = mlvc.CreateMlModel('test')
            >>> model.data_loading('db/data.csv')
            >>> model.test_train_data_split(test_size= 0.3,random_state= 11)
            >>> model_data = fit_predict_to_best_estimator(model_name='rfc',model_algorithm='RandomForestClassifier(),param_grid= {data ...},folds= 5)
            >>> model.feature_importance_plot_2(model_data=model_data)        """
        try:
            log.info(
                f'(SUCCESS feature_importance_plot_1) -> Starting process -> kwargs: {kwargs}'
            )
            model = model_data['model']
            explainer = shap.TreeExplainer(model.best_estimator_)
            shap_values = explainer.shap_values(self.X_test)
            shap.summary_plot(shap_values, self.X_test, plot_type="bar")
            plt.show()
        except BaseException as exc:
            log.error(
                f'(ERROR feature_importance_plot_2) -> Finishing process -> kwargs: {kwargs} -> '
                f'Exception: {traceback.format_exc()}'
            )
            raise FeatureImportancePlot2Error(f'Feature importance 2 could not be processed. kwargs: {kwargs}')
        log.info(
            f'(SUCCESS feature_importance_plot_2) -> Starting process -> kwargs: {kwargs}'
        )
    
    
    _= feature_importance_plot_1()
    _= feature_importance_plot_2()
    log.info(
        f'(SUCCESS feature_importance) -> Finishing process. Feature importance plots created! -> kwargs: {kwargs}'
        )
    return 1
    

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    """Add from mlvc!"""
    return True
    
if __name__ == '__main__':
    os.system('pytest -s --cov=. tests/')
