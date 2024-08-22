def blackbox(randomforestclassifier_max_depth, randomforestclassifier_min_samples_split, randomforestclassifier_min_samples_leaf, randomforestclassifier_max_features, randomforestclassifier_bootstrap, lgbmclassifier_learning_rate, lgbmclassifier_max_depth, lgbmclassifier_num_leaves, lgbmclassifier_subsample, lgbmclassifier_min_child_weight, lgbmclassifier_reg_lambda, lgbmclassifier_reg_alpha, xgbclassifier_learning_rate, xgbclassifier_max_depth, xgbclassifier_subsample, xgbclassifier_gamma, xgbclassifier_colsample_bytree, xgbclassifier_min_child_weight, xgbclassifier_reg_lambda, xgbclassifier_reg_alpha, catboostclassifier_learning_rate, catboostclassifier_depth, catboostclassifier_l2_leaf_reg, catboostclassifier_bagging_temperature, ) -> float:
	
	
	
	import os
	import random as rn
	os.environ['PYTHONHASHSEED'] = '51'
	rn.seed(89)
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	import warnings
	warnings.filterwarnings("ignore")
	
	import numpy as np, pandas as pd
	import matplotlib.pyplot as plt
	import seaborn as sns
	from sklearn.pipeline import make_pipeline, Pipeline
	from sklearn.preprocessing import StandardScaler, MinMaxScaler
	from category_encoders import OneHotEncoder, CatBoostEncoder, MEstimateEncoder
	from sklearn.model_selection import StratifiedGroupKFold
	
	
	from xgboost import XGBClassifier
	from catboost import CatBoostClassifier
	from lightgbm import LGBMClassifier
	from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
	from sklearn.linear_model import RidgeClassifier, LogisticRegression
	
	from sklearn import set_config
	import os
	from sklearn.preprocessing import FunctionTransformer
	from sklearn.model_selection import StratifiedKFold
	
	from sklearn.compose import ColumnTransformer
	from prettytable import PrettyTable
	
	from sklearn.compose import make_column_transformer
	from sklearn.base import clone
	from sklearn.base import BaseEstimator, TransformerMixin
	from sklearn.metrics import accuracy_score
	
	from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
	
	
	
	
	pd.set_option("display.max_rows",100)
	
	FILE_PATH="./workspace/hyperopt/obesity-risk/data/"
	TARGET ="NObeyesdad"
	submission_path="ori_submission.csv"
	n_splits = 9
	RANDOM_SEED = 73
	
	
	
	
	train = pd.read_csv(os.path.join(FILE_PATH, "train.csv"))
	test = pd.read_csv(os.path.join(FILE_PATH, "test.csv"))
	sample_sub = pd.read_csv(os.path.join(FILE_PATH, "sample_submission.csv"))
	train_org = pd.read_csv(os.path.join(FILE_PATH, "ObesityDataSet.csv"))
	
	
	
	def prettify_df(df):
	    table = PrettyTable()
	    table.field_names = df.columns
	
	    for row in df.values:
	        table.add_row(row)
	    print(table)
	
	
	train.head(10)
	
	
	print("Train Data")
	print(f"Total number of rows: {len(train)}")
	print(f"Total number of columns: {train.shape[1]}\n")
	
	
	print("Test Data")
	print(f"Total number of rows: {len(test)}")
	print(f"Total number of columns:{test.shape[1]}")
	
	
	
	train_copy = train.rename(columns={"family_history_with_overweight":"FHWO"})
	tmp = pd.DataFrame(index=train_copy.columns)
	tmp['count'] = train_copy.count()
	tmp['dtype'] = train_copy.dtypes
	tmp['nunique'] = train_copy.nunique()
	tmp['%nunique'] = (tmp['nunique']/len(train_copy))*100
	tmp['%null'] = (train_copy.isnull().sum()/len(train_copy))*100
	tmp['min'] = train_copy.min()
	tmp['max'] = train_copy.max()
	tmp
	
	tmp.reset_index(inplace=True)
	tmp = tmp.rename(columns = {"index":"Column Name"})
	tmp = tmp.round(3)
	prettify_df(tmp)
	del tmp, train_copy
	
	
	
	pd.set_option('display.float_format', '{:.2f}'.format)
	tmp = pd.DataFrame(train.groupby([TARGET,'Gender'])["id"].agg('count'))
	tmp.columns = ['Count']
	train[TARGET].value_counts()
	tmp = pd.merge(tmp,train[TARGET].value_counts(),left_index=True, right_index=True)
	tmp.columns = ['gender_count','target_class_count']
	tmp['%gender_count'] = tmp['gender_count']/tmp['target_class_count']
	tmp["%target_class_count"] = tmp['target_class_count']/len(train) 
	tmp = tmp[['gender_count','%gender_count','target_class_count','%target_class_count']]
	print("Target Distribution with Gender")
	tmp
	
	
	raw_num_cols = list(train.select_dtypes("float").columns) 
	raw_cat_cols = list(train.columns.drop(raw_num_cols+[TARGET]))
	
	full_form = dict({'FAVC' :"Frequent consumption of high caloric food",                  'FCVC' :"Frequency of consumption of vegetables",                  'NCP' :"Number of main meal",                  'CAEC':"Consumption of food between meals",                  'CH2O':"Consumption of water daily",                  'SCC': "Calories consumption monitoring",                  'FAF':"Physical activity frequency",                  'TUE':"Time using technology devices",                  'CALC':"Consumption of alcohol",                  'MTRANS' :"Transportation used"})
	
	
	from sklearn.decomposition import PCA
	from sklearn.cluster import KMeans
	
	
	pca = PCA(n_components=2)
	pca_top_2 = pca.fit_transform(train[raw_num_cols])
	
	tmp = pd.DataFrame(data = pca_top_2, columns = ['pca_1','pca_2'])
	tmp['TARGET'] = train[TARGET]
	
	
	kmeans = KMeans(7,random_state=RANDOM_SEED)
	kmeans.fit(tmp[['pca_1','pca_2']])
	
	
	def age_rounder(x):
	    x_copy = x.copy()
	    x_copy['Age'] = (x_copy['Age']*100).astype(np.uint16)
	    return x_copy
	
	def height_rounder(x):
	    x_copy = x.copy()
	    x_copy['Height'] = (x_copy['Height']*100).astype(np.uint16)
	    return x_copy
	
	def extract_features(x):
	    x_copy = x.copy()
	    x_copy['BMI'] = (x_copy['Weight']/x_copy['Height']**2)
	
	    return x_copy
	
	def col_rounder(x):
	    x_copy = x.copy()
	    cols_to_round = ['FCVC',"NCP","CH2O","FAF","TUE"]
	    for col in cols_to_round:
	        x_copy[col] = round(x_copy[col])
	        x_copy[col] = x_copy[col].astype('int')
	    return x_copy
	
	AgeRounder = FunctionTransformer(age_rounder)
	HeightRounder = FunctionTransformer(height_rounder)
	ExtractFeatures = FunctionTransformer(extract_features)
	ColumnRounder = FunctionTransformer(col_rounder)
	
	
	
	
	
	from sklearn.base import BaseEstimator, TransformerMixin
	class FeatureDropper(BaseEstimator, TransformerMixin):
	    def __init__(self, cols):
	        self.cols = cols
	    def fit(self,x,y):
	        return self
	    def transform(self, x):
	        return x.drop(self.cols, axis = 1)
	
	
	
	
	target_mapping = {
	                  'Insufficient_Weight':0,
	                  'Normal_Weight':1,
	                  'Overweight_Level_I':2,
	                  'Overweight_Level_II':3, 
	                  'Obesity_Type_I':4,
	                  'Obesity_Type_II':5,
	                  'Obesity_Type_III':6
	                  }
	
	
	skf = StratifiedKFold(n_splits=n_splits)
	
	def cross_val_model(estimators,cv = skf, verbose = True):
	    '''
	        estimators : pipeline consists preprocessing, encoder & model
	        cv : Method for cross validation (default: StratifiedKfold)
	        verbose : print train/valid score (yes/no)
	    '''
	    
	    X = train.copy()
	    y = X.pop(TARGET)
	
	    y = y.map(target_mapping)
	    test_predictions = np.zeros((len(test),7))
	    valid_predictions = np.zeros((len(X),7))
	
	    val_scores, train_scores = [],[]
	    for fold, (train_ind, valid_ind) in enumerate(skf.split(X,y)):
	        model = clone(estimators)
	        
	        X_train = X.iloc[train_ind]
	        y_train = y.iloc[train_ind]
	        
	        X_valid = X.iloc[valid_ind]
	        y_valid = y.iloc[valid_ind]
	
	        model.fit(X_train, y_train)
	        if verbose:
	            print("-" * 100)
	            print(f"Fold: {fold}")
	            print(f"Train Accuracy Score:{accuracy_score(y_true=y_train,y_pred=model.predict(X_train))}")
	            print(f"Valid Accuracy Score:{accuracy_score(y_true=y_valid,y_pred=model.predict(X_valid))}")
	            print("-" * 100)
	
	        
	        test_predictions += model.predict_proba(test)/cv.get_n_splits()
	        valid_predictions[valid_ind] = model.predict_proba(X_valid)
	        val_scores.append(accuracy_score(y_true=y_valid,y_pred=model.predict(X_valid)))
	    if verbose: 
	        print(f"Average Mean Accuracy Score: {np.array(val_scores).mean()}")
	    return val_scores, valid_predictions, test_predictions
	
	
	
	train.drop(['id'],axis = 1, inplace = True)
	test_ids = test['id']
	test.drop(['id'],axis = 1, inplace=True)
	
	train = pd.concat([train,train_org],axis = 0)
	train = train.drop_duplicates()
	train.reset_index(drop=True, inplace=True)
	
	
	score_list, oof_list, predict_list = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	'''
	RandomForestClassifier parameters: 
	
	
	    max_depth : int, default=None
	        The maximum depth of the tree. If None, then nodes are expanded until
	        all leaves are pure or until all leaves contain less than
	        min_samples_split samples.
	
	    min_samples_split : int or float, default=2
	        The minimum number of samples required to split an internal node:
	
	        - If int, then consider `min_samples_split` as the minimum number.
	        - If float, then `min_samples_split` is a fraction and
	          `ceil(min_samples_split * n_samples)` are the minimum
	          number of samples for each split.
	
	       .. versionchanged:: 0.18
	           Added float values for fractions.
	
	    min_samples_leaf : int or float, default=1
	        The minimum number of samples required to be at a leaf node.
	        A split point at any depth will only be considered if it leaves at
	        least ``min_samples_leaf`` training samples in each of the left and
	        right branches.  This may have the effect of smoothing the model,
	        especially in regression.
	
	        - If int, then consider `min_samples_leaf` as the minimum number.
	        - If float, then `min_samples_leaf` is a fraction and
	          `ceil(min_samples_leaf * n_samples)` are the minimum
	          number of samples for each node.
	
	       .. versionchanged:: 0.18
	           Added float values for fractions.
	
	    min_weight_fraction_leaf : float, default=0.0
	        The minimum weighted fraction of the sum total of weights (of all        the input samples) required to be at a leaf node. Samples have
	        equal weight when sample_weight is not provided.
	
	    max_features : {"sqrt","log2", None}, int or float, default="sqrt"
	        The number of features to consider when looking for the best split:
	
	        - If int, then consider `max_features` features at each split.
	        - If float, then `max_features` is a fraction and
	          `max(1, int(max_features * n_features_in_))` features are considered at each
	          split.
	        - If"sqrt", then `max_features=sqrt(n_features)`.
	        - If"log2", then `max_features=log2(n_features)`.
	        - If None, then `max_features=n_features`.
	
	       .. versionchanged:: 1.1
	            The default of `max_features` changed from `"auto"` to `"sqrt"`.
	
	        Note: the search for a split does not stop until at least one
	        valid partition of the node samples is found, even if it requires to
	        effectively inspect more than ``max_features`` features.
	'''
	
	
	RFC = make_pipeline(                        ExtractFeatures,                        MEstimateEncoder(cols=['Gender','family_history_with_overweight','FAVC','CAEC',                                           'SMOKE','SCC','CALC','MTRANS']),                       RandomForestClassifier(max_depth=randomforestclassifier_max_depth, min_samples_split=randomforestclassifier_min_samples_split, min_samples_leaf=randomforestclassifier_min_samples_leaf, max_features=randomforestclassifier_max_features, bootstrap=randomforestclassifier_bootstrap)                    )
	
	
	val_scores,val_predictions,test_predictions = cross_val_model(RFC)
	
	
	for k,v in target_mapping.items():
	    oof_list[f"rfc_{k}"] = val_predictions[:,v]
	
	for k,v in target_mapping.items():
	    predict_list[f"rfc_{k}"] = test_predictions[:,v]
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	    
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	numerical_columns = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
	categorical_columns = train.select_dtypes(include=['object']).columns.tolist()
	categorical_columns.remove('NObeyesdad')
	
	
	
	
	
	
	
	
	
	
	
	params = {'learning_rate': 0.04325905707439143, 'max_depth': 4, 
	          'subsample': 0.6115083405793659, 'min_child_weight': 0.43633356137010687, 
	          'reg_lambda': 9.231766981717822, 'reg_alpha': 1.875987414096491, 'num_leaves': 373,
	          'n_estimators' : 1000,'random_state' : RANDOM_SEED, 'device_type' :"gpu",
	         }
	
	best_params = {
	   "objective":"multiclass",          
	   "metric":"multi_logloss",          
	   "verbosity": -1,                    
	   "boosting_type":"gbdt",            
	   "random_state": 42,       
	   "num_class": 7,                     
	    'learning_rate': 0.030962211546832760,  
	    'n_estimators': 500,                
	    'lambda_l1': 0.009667446568254372,  
	    'lambda_l2': 0.04018641437301800,   
	    'max_depth': 10,                    
	    'colsample_bytree': 0.40977129346872643,  
	    'subsample': 0.9535797422450176,    
	    'min_child_samples': 26             
	}
	
	lgbm = make_pipeline(                            ColumnTransformer(                        transformers=[('num', StandardScaler(), numerical_columns),                                  ('cat', OneHotEncoder(handle_unknown="ignore"), categorical_columns)]),                        LGBMClassifier(learning_rate=lgbmclassifier_learning_rate, max_depth=lgbmclassifier_max_depth, num_leaves=lgbmclassifier_num_leaves, subsample=lgbmclassifier_subsample, min_child_weight=lgbmclassifier_min_child_weight, reg_lambda=lgbmclassifier_reg_lambda, reg_alpha=lgbmclassifier_reg_alpha)                    )
	
	
	
	val_scores,val_predictions,test_predictions = cross_val_model(lgbm)
	
	for k,v in target_mapping.items():
	    oof_list[f"lgbm_{k}"] = val_predictions[:,v]
	    
	for k,v in target_mapping.items():
	    predict_list[f"lgbm_{k}"] = test_predictions[:,v]
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	    
	    
	
	
	
	
	
	
	    
	
	
	
	
	
	
	
	
	
	
	
	
	
	params = {
	    'n_estimators': 1312,
	    'learning_rate': 0.018279520260162645,
	    'gamma': 0.0024196354156454324,
	    'reg_alpha': 0.9025931173755949,
	    'reg_lambda': 0.06835667255875388,
	    'max_depth': 5,
	    'min_child_weight': 5,
	    'subsample': 0.883274050086088,
	    'colsample_bytree': 0.6579828557036317
	}
	
	
	params = {'grow_policy': 'depthwise', 'n_estimators': 690, 
	               'learning_rate': 0.31829021594473056, 'gamma': 0.6061120644431842, 
	               'subsample': 0.9032243794829076, 'colsample_bytree': 0.44474031945048287,
	               'max_depth': 10, 'min_child_weight': 22, 'reg_lambda': 4.42638097284094,
	               'reg_alpha': 5.927900973354344e-07,'seed':RANDOM_SEED}
	
	best_params = {'grow_policy': 'depthwise', 'n_estimators': 982, 
	               'learning_rate': 0.050053726931263504, 'gamma': 0.5354391952653927, 
	               'subsample': 0.7060590452456204, 'colsample_bytree': 0.37939433412123275, 
	               'max_depth': 23, 'min_child_weight': 21, 'reg_lambda': 9.150224029846654e-08,
	               'reg_alpha': 5.671063656994295e-08}
	best_params['booster'] = 'gbtree'
	best_params['objective'] = 'multi:softmax'
	best_params["device"] ="cuda"
	best_params["verbosity"] = 0
	best_params['tree_method'] ="gpu_hist"
	    
	XGB = make_pipeline(                    MEstimateEncoder(cols=['Gender','family_history_with_overweight','FAVC','CAEC',                                           'SMOKE','SCC','CALC','MTRANS']),                    XGBClassifier(learning_rate=xgbclassifier_learning_rate, max_depth=xgbclassifier_max_depth, subsample=xgbclassifier_subsample, gamma=xgbclassifier_gamma, colsample_bytree=xgbclassifier_colsample_bytree, min_child_weight=xgbclassifier_min_child_weight, reg_lambda=xgbclassifier_reg_lambda, reg_alpha=xgbclassifier_reg_alpha)                   )
	
	val_scores,val_predictions,test_predictions = cross_val_model(XGB)
	
	for k,v in target_mapping.items():
	    oof_list[f"xgb_{k}"] = val_predictions[:,v]
	
	for k,v in target_mapping.items():
	    predict_list[f"xgb_{k}"] = test_predictions[:,v]
	    
	
	
	
	
	
	
	
	
	
	
	    
	
	        
	
	
	
	
	
	
	
	
	
	    
	
	
	
	
	
	
	
	
	
	
	
	    
	
	
	params = {'learning_rate': 0.13762007048684638, 'depth': 5, 
	          'l2_leaf_reg': 5.285199432056192, 'bagging_temperature': 0.6029582154263095,
	         'random_seed': RANDOM_SEED,
	        'verbose': False,
	        'task_type':"GPU",
	         'iterations':1000}
	
	cat_features_indices = [train.columns.get_loc(col) for col in categorical_columns]
	
	CB = make_pipeline(    MEstimateEncoder(cols=categorical_columns),    CatBoostClassifier(learning_rate=catboostclassifier_learning_rate, depth=catboostclassifier_depth, l2_leaf_reg=catboostclassifier_l2_leaf_reg, bagging_temperature=catboostclassifier_bagging_temperature))
	
	
	
	
	
	
	
	
	
	
	val_scores,val_predictions,test_predictions = cross_val_model(CB)
	for k,v in target_mapping.items():
	    oof_list[f"cat_{k}"] = val_predictions[:,v]
	
	for k,v in target_mapping.items():
	    predict_list[f"cat_{k}"] = test_predictions[:,v]
	
	
	
	
	
	
	
	weights = {"rfc_":0,
	          "lgbm_":3,
	          "xgb_":1,
	          "cat_":0}
	tmp = oof_list.copy()
	for k,v in target_mapping.items():
	    tmp[f"{k}"] = (weights['rfc_']*tmp[f"rfc_{k}"] +              weights['lgbm_']*tmp[f"lgbm_{k}"]+              weights['xgb_']*tmp[f"xgb_{k}"]+              weights['cat_']*tmp[f"cat_{k}"])    
	tmp['pred'] = tmp[target_mapping.keys()].idxmax(axis = 1)
	tmp['label'] = train[TARGET]
	print(f"Ensemble Accuracy Scoe: {accuracy_score(train[TARGET],tmp['pred'])}")
	    
	cm = confusion_matrix(y_true = tmp['label'].map(target_mapping),                      y_pred = tmp['pred'].map(target_mapping),                     normalize='true')
	
	cm = cm.round(2)
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	for k,v in target_mapping.items():
	    predict_list[f"{k}"] = (weights['rfc_']*predict_list[f"rfc_{k}"]+                            weights['lgbm_']*predict_list[f"lgbm_{k}"]+                            weights['xgb_']*predict_list[f"xgb_{k}"]+                            weights['cat_']*predict_list[f"cat_{k}"])
	
	final_pred = predict_list[target_mapping.keys()].idxmax(axis = 1)
	
	sample_sub[TARGET] = final_pred
	sample_sub.to_csv(os.path.join(FILE_PATH,submission_path),index=False)
	
	
	score= 1-accuracy_score(train[TARGET],tmp['pred'])
	
	return score