import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split



# class BootstrapSplitter:

#     def __init__(self, reps, train_size, replace = True, random_state=None):
#         self.reps = reps
#         self.train_size = train_size
#         self.random_state = random_state
#         self.replace = replace
#         self.RNG = np.random.default_rng(self.random_state)

#     def get_n_splits(self):
#         return self.reps

#     def split(self, x, y=None, groups=None):
#         for _ in range(self.reps):
#             # train_idx = self.RNG.choice(np.arange(len(x)), size=round(self.train_size*len(x)), replace=True)
#             train_idx = self.RNG.choice(np.arange(len(x)), size=self.train_size, replace=self.replace)
#             test_idx = np.setdiff1d(np.arange(len(x)), train_idx)
#             # np.random.shuffle(test_idx)
#             yield train_idx, test_idx


import numpy as np

class BootstrapSplitter:
    def __init__(self, reps, train_size, replace=True, random_state=None):
        self.reps = reps
        self.train_size = train_size
        self.replace = replace
        self.random_state = random_state
        self.RNG = np.random.default_rng(self.random_state)

    def get_n_splits(self):
        return self.reps

    def split(self, x, y=None, groups=None):
        n = len(x)
        for _ in range(self.reps):
            train_idx = self.RNG.choice(np.arange(n), size=self.train_size, replace=self.replace)
            test_idx = np.setdiff1d(np.arange(n), train_idx)

            yield train_idx, test_idx

class TrainTestSplitter:
    def __init__(self, reps, train_size, replace=False, random_state=None):
        self.reps = reps
        self.train_size = train_size
        self.replace = replace
        self.random_state = random_state
        self.RNG = np.random.default_rng(self.random_state)

    def get_n_splits(self):
        return self.reps

    def split(self, x, y=None, groups=None):
        n = len(x)
        for _ in range(self.reps):
            train_idx, test_idx = train_test_split(np.arange(n), train_size=self.train_size, random_state=self.RNG.integers(0, 1e9), shuffle=True)
            yield train_idx, test_idx


class CallData:

    def __init__(self, filepath = 'datasets/', standardize = True):

        self.filepath = filepath
        self.stndz = standardize
        self.feature_names_dict = {
            'california_housing_price': ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'],
            'used_car_price': ['count', 'km', 'year', 'powerPS'],
            'red_wine': ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'],
            'diabetes':['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'],
            'diabetes_class': ['Age', 'BMI', 'BloodPressure', 'Glucose', 'Insulin', 'Pregnancies', 'SkinThickness'],
            'make_friedman1': ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'],
            'make_friedman2': ['x1', 'x2', 'x3', 'x4'],
            'make_friedman3': ['x1', 'x2', 'x3', 'x4'],
            'synthetic1': ['x1','x2'],
            'breast_cancer': ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
                            'mean smoothness', 'mean compactness', 'mean concavity',
                            'mean concave points', 'mean symmetry', 'mean fractal dimension',
                            'radius error', 'texture error', 'perimeter error', 'area error',
                            'smoothness error', 'compactness error', 'concavity error',
                            'concave points error', 'symmetry error', 'fractal dimension error',
                            'worst radius', 'worst texture', 'worst perimeter', 'worst area',
                            'worst smoothness', 'worst compactness', 'worst concavity',
                            'worst concave points', 'worst symmetry', 'worst fractal dimension'],
            'iris': ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
            'banknote': ['variance', 'skewness', 'curtosis', 'entropy'],
            'magic04': ['fLen1t-1',	'fWidt-1', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Lon1', 'fM3Trans', 'fAlp-1a', 'fDist'],
            'voice': ['meanfreq', 'sd', 'median', 'Q25', 'Q75', 'IQR', 'skew', 'kurt', 'sp.ent', 'sfm', 'mode', 'centroid', 
                      'meanfun', 'minfun', 'maxfun',
                      'meandom', 'mindom', 'maxdom', 'dfrange', 'modindx'],
            'synthetic': ['x1', 'x2'],
            'liver': ['mcv', 'alkphos',	'sgpt', 'sgot', 'gammagt', 'drinks'],
            'adult': ['age', 'workclass', 'fnlwgt', 'education-num', 'marital-status',
                    'occupation', 'relationship', 'race', 'sex', 'capital-gain',
                    'capital-loss', 'hours-per-week', 'native-country'],
            'digits0': ['pixel_0_0', 'pixel_0_1', 'pixel_0_2', 'pixel_0_3', 'pixel_0_4', 'pixel_0_5', 'pixel_0_6', 'pixel_0_7', 'pixel_1_0', 'pixel_1_1',
                        'pixel_1_2', 'pixel_1_3', 'pixel_1_4', 'pixel_1_5', 'pixel_1_6', 'pixel_1_7', 'pixel_2_0', 'pixel_2_1', 'pixel_2_2', 'pixel_2_3', 
                        'pixel_2_4', 'pixel_2_5', 'pixel_2_6', 'pixel_2_7', 'pixel_3_0', 'pixel_3_1', 'pixel_3_2', 'pixel_3_3', 'pixel_3_4', 'pixel_3_5', 
                        'pixel_3_6', 'pixel_3_7', 'pixel_4_0', 'pixel_4_1', 'pixel_4_2', 'pixel_4_3', 'pixel_4_4', 'pixel_4_5', 'pixel_4_6', 'pixel_4_7', 
                        'pixel_5_0', 'pixel_5_1', 'pixel_5_2', 'pixel_5_3' , 'pixel_5_4', 'pixel_5_5', 'pixel_5_6', 'pixel_5_7', 'pixel_6_0', 'pixel_6_1', 
                        'pixel_6_2', 'pixel_6_3', 'pixel_6_4', 'pixel_6_5', 'pixel_6_6', 'pixel_6_7', 'pixel_7_0', 'pixel_7_1', 'pixel_7_2', 'pixel_7_3',
                        'pixel_7_4', 'pixel_7_5', 'pixel_7_6', 'pixel_7_7'],
            'bmi': ['Gender', 'Height', 'Weight'],
            'winequality': ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'],
            'titanic': ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
            'car_price': ['manufacturer', 'model', 'prod._year', 'category', 'leather_interior', 'fuel_type', 'engine_volume', 'mileage', 'cylinders', 'gear_box_type', 'drive_wheels', 'doors', 'wheel', 'color', 'airbags', 'turbo'],
            'octet_selected': ['IP(A)', 'EA(A)', 'H(A)', 'L(A)', 'rs(A)', 'rp(A)', 'rd(A)', 'IP(B)', 'EA(B)', 'H(B)', 'L(B)', 'rs(B)', 'rp(B)', 'rd(B)', 'EN(A)', 'EN(B)'],
            'iris_pca': ['PC1', 'PC2'],
            # 'nhanes_diabetes': ['age', 'gender', 'height_cm', 'height_m', 'weight_kg', 'ln_height_m', 'ln_weight_kg'],
            'nhanes_diabetes': ['age', 'gender', 'glycohemoglobin_pct', 'ln_height_m', 'ln_weight_kg'],
            'concrete': ['cement', 'blast_furnace_slag', 'fly_ash', 'water', 'superplasticizer', 'coarse_aggregate', 'fine_aggregate', 'age'],
            'tic_tac_toe': ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9'],
            'protein': ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9'],
            'parkinsons': ["age", "sex", "test_time", "jitter", "jitter_abs", "jitter_rap", "jitter_ppq5", 
                           "jitter_ddp", "shimmer", "shimmer_db", "shimmer_apq3", "shimmer_apq5", 
                           "shimmer_apq11", "shimmer_dda", "nhr", "hnr", "rpde", "dfa", "ppe"],
        }
        self.target_name_dict = {
            'california_housing_price': 'MedHouseVal',
            'used_car_price': 'avgPrice',
            'red_wine': 'quality',
            'diabetes': 'target',
            'diabetes_class': 'Outcome',
            'make_friedman1': 'y',
            'make_friedman2': 'y',
            'make_friedman3': 'y',
            'synthetic1': 'y',
            'breast_cancer': 'y',
            'iris': 'y',
            'banknote': 'class',
            'magic04': 'class',
            'voice': 'label',
            'synthetic': 'y',
            'liver': 'selector',
            'adult': 'output',
            'digits0': 'target',
            'bmi': 'Index',
            'winequality': 'quality',
            'titanic': 'Survived',
            'car_price': 'price',
            'octet_selected': 'Delta E',
            'iris_pca': 'target',
            'nhanes_diabetes': 'diabetes_label',
            'concrete': 'concrete_compressive_strength',
            'tic_tac_toe': 'label',
            'protein': 'RMSD',
            'parkinsons': 'total_updrs',
        }
        self.learning_type_dict = {
            'california_housing_price': 'linreg',
            'used_car_price': 'linreg',
            'red_wine': 'linreg',
            'diabetes': 'linreg',
            'diabetes_class': 'logreg',
            'make_friedman1': 'linreg',
            'make_friedman2': 'linreg',
            'make_friedman3': 'linreg',
            'synthetic1': 'linreg',
            'breast_cancer': 'logreg',
            'iris': 'logreg',
            'banknote': 'logreg',
            'magic04': 'logreg',
            'voice': 'logreg',
            'synthetic': 'logreg',
            'liver': 'logreg',
            'adult': 'logreg',
            'digits0': 'logreg',
            'bmi': 'linreg',
            'winequality': 'linreg',
            'titanic': 'logreg',
            'car_price': 'linreg',
            'octet_selected': 'linreg',
            'iris_pca': 'logreg',
            'nhanes_diabetes': 'logreg',
            'concrete': 'linreg',
            'tic_tac_toe': 'logreg',
            'protein': 'linreg',
            'parkinsons': 'linreg',
        }
        self.categorical_cols_dict = {
            'adult': ['workclass', 'marital-status', 'occupation', 'relationship', 'race','sex'],
            'bmi': ['Gender'],
            'titanic': ['Pclass', 'Sex', 'Embarked', 'Parch', 'SibSp'],
            'car_price': ['manufacturer', 'model', 'category', 'leather_interior', 'fuel_type','gear_box_type', 'drive_wheels', 
                          'doors', 'wheel', 'color', 'turbo'],
            'nhanes_diabetes': ['gender'],
            'tic_tac_toe': ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9'],
            'parkinsons': ['sex'],
        }
        self.numerical_cols_dict = {'adult' : ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'],
                                    'bmi': ['Weight', 'Height'],
                                    'titanic': ['Age', 'Fare'],
                                    'car_price': ['engine_volume', 'prod._year', 'mileage', 'cylinders', 'airbags'],
                                    'nhanes_diabetes': ['age', 'glycohemoglobin_pct', 'ln_height_m', 'ln_weight_kg'],
                                    'parkinsons': ['age', 'test_time', 'jitter', 'jitter_abs', 'jitter_rap', 'jitter_ppq5', 
                                                   'jitter_ddp', 'shimmer', 'shimmer_db', 'shimmer_apq3', 'shimmer_apq5', 
                                                   'shimmer_apq11', 'shimmer_dda', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe']
                                    }
    @staticmethod
    def standardize(x):
        x = (x - x.mean(axis=0))/(x.std(axis=0))
        return x

    def call(self, dataset_name = 'california_housing_price', data_format = 'arr'):
        """
        data_format: 'arr' (numpy array) or 'df' (pandas dataframe)
        """
        if dataset_name is None:
            raise ValueError("dataset_name must be provided")
        encoder = OneHotEncoder(sparse_output=False)

        filename = dataset_name+'.csv'
        data = pd.read_csv(f'{self.filepath}{filename}')
        self.original_data = data.copy()
        self.learning_type = self.learning_type_dict[dataset_name]
        self.target_name = self.target_name_dict[dataset_name]
        self.features_name = self.feature_names_dict[dataset_name]
        data = data[[self.target_name] + self.features_name].dropna().reset_index(drop=True)
        X_df = data[self.features_name]
        X_df = X_df.loc[:, (X_df != 0).any(axis=0)]
        y_df = data[self.target_name]
        if dataset_name in self.categorical_cols_dict.keys():
            cat_X_one_hot_encoded = encoder.fit_transform(X_df[self.categorical_cols_dict[dataset_name]])
            cat_X_one_hot_df = pd.DataFrame(cat_X_one_hot_encoded, columns=encoder.get_feature_names_out(self.categorical_cols_dict[dataset_name]))
            self.x_means = X_df[self.numerical_cols_dict[dataset_name]].mean(axis=0) if dataset_name in self.numerical_cols_dict.keys() else None
            self.x_stds = X_df[self.numerical_cols_dict[dataset_name]].std(axis=0) if dataset_name in self.numerical_cols_dict.keys() else None
            if self.stndz is True and dataset_name in self.numerical_cols_dict.keys():
                num_X_standardized = self.standardize(X_df[self.numerical_cols_dict[dataset_name]])
                X_df = pd.concat([cat_X_one_hot_df,num_X_standardized], axis=1)
            else:
                X_df = pd.concat([cat_X_one_hot_df,X_df[self.numerical_cols_dict[dataset_name]]], axis=1) if dataset_name in self.numerical_cols_dict.keys() else cat_X_one_hot_df
        else:
            self.x_means = X_df.mean(axis=0)
            self.x_stds = X_df.std(axis=0)
            if self.stndz is True:
                X_df = self.standardize(X_df)

        
        if self.learning_type == 'linreg':
            self.y_means = y_df.mean(axis=0)
            self.y_stds = y_df.std(axis=0)
            if self.stndz is True:
                y_df = self.standardize(y_df)
        elif self.learning_type == 'logreg': #and y_df.isin([0, 1]).all().all():
            y_df = (y_df>0).astype(float)
        
        if data_format == 'arr':
            self.X = X_df.to_numpy()
            self.y = y_df.to_numpy()
        else:
            self.X = X_df
            self.y = y_df
        
        return self
    