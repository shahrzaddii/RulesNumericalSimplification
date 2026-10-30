from decimal import Decimal

import numpy as np
from sklearn.linear_model import LogisticRegression, ridge_regression, Ridge
from sklearn.base import BaseEstimator
from scipy.special import expit
from sklearn.metrics import log_loss, mean_squared_error, zero_one_loss
import pandas as pd
from dataset import BootstrapSplitter
from sklearn.model_selection import train_test_split, KFold, cross_validate
from .optikon import greedy_maximization, WeightedSupport, equal_width_propositionalization, full_propositionalization
from numba import int64, float64, boolean, njit
from numba.experimental import jitclass
import copy
import time
import math
from itertools import permutations

def integer_digit_count(v):
    if math.trunc(abs(v)) == 0:
        return 0
    return int(math.floor(math.log10(abs(v)))) + 1

def truncate_decimals(x, n, round = False):
    factor = 10 ** n
    if round:
        check = np.round(x,n)
        return np.round(x, n)
    else:
        return np.trunc(x * factor) / factor

def round_to_n_significant(x, n, round = False):
    x = np.asarray(x, dtype=float)

    # handle zeros separately to avoid log10 issues
    mask = x != 0

    # compute scale (power of 10)
    scale = np.zeros_like(x)
    scale[mask] = np.floor(np.log10(np.abs(x[mask]))) - (n - 1)

    # round
    result = np.zeros_like(x)
    result[mask] = truncate_decimals(x[mask] / (10**scale[mask]), 0, round) * (10**scale[mask])
    # result[mask] = np.round(x[mask] / (10**scale[mask])) * (10**scale[mask])

    return result.astype(int)

def num_decimal_digits(x):
    x = float(x)
    if x == 0:
        return 0
    d = Decimal(str(x)).normalize()
    return max(-d.as_tuple().exponent, 0)

def max_min_decimal_digits_per_column(w):
    w = np.asarray(w)
    max_digits = []
    min_digits = []

    for j in range(w.shape[1]):
        col = w[:, j]
        nonzero_col = col[col != 0]

        if len(nonzero_col) == 0:
            max_digits.append(0)
            min_digits.append(0)
        else:
            digits = [num_decimal_digits(v) for v in nonzero_col]
            max_digits.append(max(digits))
            min_digits.append(min(digits))

    return np.array(max_digits), np.array(min_digits)

def scale_columns_to_integers(w):
    w = np.asarray(w, dtype=float)
    max_digits, min_digits = max_min_decimal_digits_per_column(w)
    scales = 10 ** max_digits
    w_int = np.round(w * scales).astype(int)
    return w_int, max_digits, min_digits

import numpy as np

def trailing_zeros_int(x):
    x = int(round(abs(x)))
    if x == 0:
        return 0
    s = str(x)
    return len(s) - len(s.rstrip('0'))

def min_shared_power10_per_column(w):
    w = np.asarray(w)
    p = []

    for j in range(w.shape[1]):
        col = w[:, j]
        nonzero_col = col[col != 0]

        if nonzero_col.size == 0:
            p.append(0)
        else:
            p.append(min(trailing_zeros_int(v) for v in nonzero_col))

    return np.array(p, dtype=int)

# risk
@jitclass
class LeastSquaresRisk:

    def __init__(self):
        pass

    def gradient(self, spec, state):
        # return state.phi[:, :state.current_features].dot(state.coef[:state.current_features]) - spec.y
        return state.grads
    
    def hessian_diagonal(self, spec, state):
        return np.ones_like(spec.y)
    
    def value(self, spec, state):
        return np.sum((state.phi[:, :state.current_features].dot(state.coef[:state.current_features]) - spec.y))**2
    
@njit
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

@jitclass
class LogisticRisk:

    def __init__(self):
        pass

    def gradient(self, spec, state):
        # return sigmoid(state.phi[:, :state.current_features].dot(state.coef[:state.current_features])) - spec.y
        return state.grads
    
    def hessian_diagonal(self, spec, state):
        p = sigmoid(state.phi[:, :state.current_features].dot(state.coef[:state.current_features]))
        return p*(1-p)


# spec
regression_spec_spec = [
    ('y', float64[:]),
    ('x', float64[:, :]),
    ('max_features', int64),
    ('intercept', boolean),
    ('lam', float64)
]

@jitclass(regression_spec_spec)
class RegressionSpec:
    def __init__(self, y, x, max_features, intercept, lam):
        self.y = y
        self.x = x
        self.max_features = max_features
        self.intercept = intercept
        self.lam = lam

classification_spec_spec = [
    ('y', int64[:]),
    ('x', float64[:, :]),
    ('max_features', int64),
    ('intercept', boolean),
    ('lam', float64),
    ('max_iter', int64),
    ('tol', float64)
]

@jitclass(classification_spec_spec)
class ClassificationSpec:
    def __init__(self, y, x, max_features, intercept, lam):
        self.y = y
        self.x = x
        self.max_features = max_features
        self.intercept = intercept
        self.lam = lam
        self.max_iter=100
        self.tol=1e-6

# state
state_spec = [
    ('phi', float64[:, :]),
    ('coef', float64[:]),
    ('current_features', int64),
    ('grads', float64[:]),
]

@jitclass(state_spec)
class BoostingState:
    def __init__(self, phi, coef, current_features, grads):
        self.phi = phi
        self.coef = coef
        self.current_features = current_features
        self.grads = grads

    @staticmethod
    def from_spec(spec, g):
        phi = np.zeros(shape=(len(spec.y), spec.max_features+spec.intercept))
        coef = np.zeros(spec.max_features+spec.intercept)
        current_features = 0
        grads = g
        return BoostingState(phi, coef, current_features, grads)


# prop_factory
prop_options = {
    'equal_width': equal_width_propositionalization,
    'full': full_propositionalization
} 

# @jitclass
class GreedyGradientSumBaseLearnerCustomized:

    max_depth: int64

    def __init__(self, spec, max_depth=5, prop_factory=None):
        self.max_depth = max_depth

    def compute(self, spec, state, risk_function):
        g = risk_function.gradient(spec, state)

        opt_q_pos, opt_val_pos, _ = greedy_maximization(spec.x, WeightedSupport(g), self.max_depth)
        # opt_q_neg, opt_val_neg, _ = greedy_maximization(spec.x, WeightedSupport(-g), self.max_depth)
        # if opt_val_pos >= opt_val_neg:
        #     return opt_q_pos
        # else:
        return opt_q_pos


def expit_bnd(x):
    x_clipped = np.clip(x, -36, 36)
    return expit(x_clipped)


class L1LogisticRegression(BaseEstimator):
    """ Logistic Regression with L1 penalty for desired sparsity by using binary search to find the optimal C value. """
    def __init__(self, max_num_nonzero_coef=100, C_range=(0, 1e4), max_iter=1000,
                 tol=1e-4, fit_intercept=True, seed=None, max_binary_search_iter=10000, l2_reg=1e-30, no_refit=False):
        self.max_iter = max_iter
        self.tol = tol
        self.max_num_nonzero_coef = max_num_nonzero_coef
        self.C_range = C_range
        self.fit_intercept = fit_intercept
        self.max_binary_search_iter = max_binary_search_iter
        self.l2_reg = l2_reg
        self.no_refit = no_refit
        # self.current_seed = np.random.SeedSequence().entropy
        # if seed is not None:
        #     self.current_seed = seed
        # self.rng = np.random.default_rng(self.current_seed)

    def upper_bound_binary_search(self, x, y, g):
        high_sparse = False
        bound = 1 
        while(high_sparse is False):
            model = LogisticRegression(
                penalty='l1',
                solver='liblinear',
                max_iter=self.max_iter,
                tol=self.tol,
                fit_intercept=self.fit_intercept,
                C=bound
            ).fit(x, y, sample_weight=g)
            sparsity = np.count_nonzero(model.coef_[0])
            if sparsity >= self.max_num_nonzero_coef:
                high_sparse = True
            else:
                bound = bound * 2
        return bound, sparsity, model
    
    def refit(self, x, y, g=None):
        model = LogisticRegression(
                    penalty='l2',
                    solver='newton-cg',
                    max_iter=self.max_iter,
                    tol=self.tol,
                    fit_intercept=self.fit_intercept,
                    C=1/self.l2_reg
            ).fit(x, y, sample_weight=g)
        
        return model
    
    def fit(self, x, y, g=None):
        refit_model = None
        if g is None:
            g = np.ones_like(y)
        low = self.C_range[0]
        self.coef_ = np.zeros((x.shape[1]))
        high, sparsity, model = self.upper_bound_binary_search(x, y, g)
        coef_backup = np.zeros((x.shape[1]))
        if sparsity == self.max_num_nonzero_coef:
            nz_coef = np.nonzero(model.coef_[0])[0]
            x_sparse = x[:,nz_coef]
            refit_model = self.refit(x_sparse, y, g)
            self.coef_[nz_coef] = refit_model.coef_[0]
            self.intercept_ = refit_model.intercept_[0]
            self.C = high
            self.l1_coef_ = model.coef_[0]
            self.l1_intercept_ = model.intercept_[0]
        else:
            for _ in range(self.max_binary_search_iter):
                mid = (low + high) / 2
                model = LogisticRegression(
                    penalty='l1',
                    solver='liblinear',
                    max_iter=self.max_iter,
                    tol=self.tol,
                    fit_intercept=self.fit_intercept,
                    C=mid
                ).fit(x, y, sample_weight=g)
                nonzero_count = np.count_nonzero(model.coef_[0])

                if nonzero_count == self.max_num_nonzero_coef:
                    nz_coef = np.nonzero(model.coef_[0])[0]
                    x_sparse = x[:,nz_coef]
                    refit_model = self.refit(x_sparse, y, g)
                    self.coef_[nz_coef] = refit_model.coef_[0]
                    self.intercept_ = refit_model.intercept_[0]
                    self.C = mid
                    self.l1_coef_ = model.coef_[0]
                    self.l1_intercept_ = model.intercept_[0]
                    break
                elif nonzero_count > self.max_num_nonzero_coef:
                    high = mid
                else:
                    low = mid

                if nonzero_count != 0:
                    nz_coef_backup = np.nonzero(model.coef_.reshape(1,-1)[0])[0]
                    x_sparse_backup = x[:,nz_coef_backup]
                    refit_model_backup = self.refit(x_sparse_backup, y, g)
                    coef_backup[nz_coef_backup] = refit_model_backup.coef_.reshape(1,-1)[0]
                    intercept_backup = np.array(refit_model_backup.intercept_).reshape(1,-1)[0]
                    C_backup = mid
                    l1_coef_backup = model.coef_[0]
                    l1_intercept_backup = model.intercept_[0]

        if refit_model is None:
            self.refit_model = refit_model_backup
            self.coef_ = coef_backup
            self.intercept_ = intercept_backup
            self.C = C_backup
            self.l1_coef_ = l1_coef_backup
            self.l1_intercept_ = l1_intercept_backup

        if self.no_refit is True:
            self.coef_ = self.l1_coef_
            self.intercept_ = self.l1_intercept_

        if np.count_nonzero(self.coef_) == 1:
            self.intercept_ = self.intercept_/max(abs(self.coef_))
            self.coef_ = np.where(self.coef_ != 0, 1, 0)*np.sign(self.coef_)
        return self

class Proposition(BaseEstimator):
    def __init__(self, sparsity=500, no_refit=False, task='linreg', greedy=False):
        self.sparsity = sparsity
        self.no_refit = no_refit
        self.no_improvement = False
        self.task = task
        self.greedy = greedy

    def one_var_proposition(self,x,y, g):
        n, d = x.shape
        self.coef_ = np.zeros(d)
        self.intercept_ = np.zeros(1)
        max_features = 1
        intercept = False
        lam = 1e-10
        max_depth = 1
        if self.task == 'logreg':
            y_arr = np.asarray(y, dtype=np.int64)      # 1D int64
            X_arr = np.asarray(x, dtype=np.float64)    # 2D float64

            spec = ClassificationSpec(y_arr, X_arr, max_features, bool(intercept), float(lam))
            state = BoostingState.from_spec(spec, g)
            prop = prop_options['equal_width']
            risk_func = LogisticRisk()
        else:
            y_arr = np.asarray(y, dtype=np.float64)      # 1D float64
            X_arr = np.asarray(x, dtype=np.float64)    # 2D float64


            spec = RegressionSpec(y_arr, X_arr, max_features, bool(intercept), float(lam))
            state = BoostingState.from_spec(spec, g)
            prop = prop_options['equal_width']
            risk_func = LeastSquaresRisk()
        
        greedy_base = GreedyGradientSumBaseLearnerCustomized(spec, np.int64(max_depth), prop)
        opt_p = greedy_base.compute(spec, state, risk_func)
        v = opt_p.v
        self.intercept_ = -opt_p.t
        self.coef_[v[0]] = opt_p.s
        return self
    
    def fit(self, x, y, g_class, g=None):
        if g is None:
            g = np.ones_like(g_class)
        if self.sparsity == 1 and self.greedy is True:
            self.prop = self.one_var_proposition(x,y,g)
        else:
            self.prop = L1LogisticRegression(max_num_nonzero_coef=self.sparsity, no_refit=self.no_refit).fit(x=x,y=g_class,g=abs(g))
        return self

class LearnRule():
    def __init__(self, max_complexity=10, epsilon_r=0.1, weighted_logloss = True, no_refit=False, 
                 task = 'linreg', greedy=False, x_std=None, x_mean=None, axis_paral=False):
        self.epsilon_r = epsilon_r
        self.max_complexity = max_complexity
        self.rng = np.random.default_rng()
        self.weighted_logloss = weighted_logloss
        self.no_refit = no_refit
        self.task = task
        self.greedy = greedy
        self.x_std = x_std
        self.x_mean = x_mean
        self.axis_paral = axis_paral

    def gradient_sum_obj(self, g, ext):
        if len(ext)==0:
            return -np.inf
        # return sum(g[ext])/(sum(ext)+1e-15)
        return sum(g[ext])
    
    def unnormalize_weights(self):
        if self.x_std is not None and self.x_mean is not None:
            self.w_tilde_ = self.w_.copy()
            self.t_tilde_ = self.t_.copy()
            self.w_tilde_ = np.round(self.w_tilde_.T / self.x_std, decimals=16)
            self.t_tilde_ = self.t_tilde_ - np.dot(self.w_tilde_, self.x_mean)
            self.w_tilde_ = self.w_tilde_.T
        return self
    
    def fit(self, x, y, g, rulens_pred_fn=None):
        best_valid_risk = np.inf
        best_obj = -np.inf
        n, d = x.shape
        best_w = np.zeros((d, 0))
        best_t = np.zeros(0)
        best_z = 1
        for z in [1,-1]:
            w, t, obj, valid_risk = self.learn_conjunction(x, y, g, z, rulens_pred_fn)
            if valid_risk < best_valid_risk:
                best_w = w
                best_t = t
                best_obj = obj
                best_valid_risk = valid_risk
                best_z = z
                self.best_obj_trend_to_out = self.obj_trend_all_complexity
                self.best_valid_trend_to_out = self.valid_trend_all_complexity

        self.w_ = best_z*best_w
        self.t_ = best_z*best_t
        self.obj_ = best_obj
        self.best_valid_risk_ = best_valid_risk
        # self.unnormalize_weights()

        return self

    def learn_conjunction(self, x, y, g, z, rulens_pred_fn=None):
        g = g*z
        complexity_level = 0
        configurations = {0: []}
        best_risk = np.inf
        best_config = None
        prop_sparsity = 0
        _, d = x.shape
        self.obj_trend_all_complexity = []
        self.valid_trend_all_complexity = []
        while complexity_level < self.max_complexity:
            next_config = []
            w = np.zeros((d, self.max_complexity))
            t = np.zeros(self.max_complexity)
            prop = configurations[complexity_level]

            if len(prop) > 0:
                num_learned_props = len(prop[1])
            else:
                num_learned_props = 0

            if num_learned_props < self.max_complexity:
                num_to_learn_props = num_learned_props + 1

            # initialising the variables
            best_obj = -np.inf
            best_valid = np.inf
            obj_trend = []
            updated_prop_obj = self.gradient_sum_obj(g, self._predict(x, w, t).astype(bool))
            updated_prop_val_risk = np.inf
            for i in range(num_to_learn_props):
                if len(prop) > 0:
                    w[:, :num_learned_props] = prop[0]
                    t[:num_learned_props] = prop[1]
                prop_sparsity = np.count_nonzero(w[:,i])
                if prop_sparsity < d:
                    w_fixed = np.concatenate((w[:, :i], w[:, i+1:num_to_learn_props]), axis=1)
                    t_fixed = np.concatenate((t[:i], t[i+1:num_to_learn_props]), axis=0)
                    selected = self._predict(x, w_fixed, t_fixed).astype(bool)
                    selected_size = np.sum(selected)
                    g_pos_size = sum(g[selected]>=0)
                    g_neg_size = sum(g[selected]<0)
                    if not (len(g[selected])==sum(g[selected]>=0) or len(g[selected])==sum(g[selected]<0)):
                        if self.axis_paral is True:
                            updated_prop = Proposition(sparsity=1, no_refit=self.no_refit, task=self.task, 
                                                   greedy=self.greedy).fit(x=x[selected], y=y[selected], g_class=g[selected]>=0, g=g[selected])
                        else:
                            updated_prop = Proposition(sparsity=prop_sparsity+1, no_refit=self.no_refit, task=self.task, 
                                                   greedy=self.greedy).fit(x=x[selected], y=y[selected], g_class=g[selected]>=0, g=g[selected])
                        # if updated_prop.no_improvement is False:
                        w[:,i] = updated_prop.prop.coef_
                        t[i] = updated_prop.prop.intercept_
                        new_selected = self._predict(x, w[:,:num_to_learn_props], t[:num_to_learn_props]).astype(bool)
                        new_selected_size = np.sum(new_selected)
                        updated_prop_obj = self.gradient_sum_obj(g, new_selected)

                        self.w_ = z*w[:,:num_to_learn_props]
                        self.t_ = z*t[:num_to_learn_props]
                        updated_prop_val_risk = rulens_pred_fn(self)
                        obj_trend.append(updated_prop_obj)

                    if updated_prop_obj > best_obj:
                        w_sel = np.copy(w[:,:num_to_learn_props])
                        t_sel = np.copy(t[:num_to_learn_props])
                        best_obj = updated_prop_obj
                        if np.count_nonzero(w_sel[:,-1]) == 0:
                            w_sel = w_sel[:,:-1]
                            t_sel = t_sel[:-1]
                        next_config = [w_sel, t_sel, updated_prop_obj, updated_prop_val_risk]


            configurations[complexity_level + 1] = next_config
            rule_complexity = np.count_nonzero(next_config[0])
            self.obj_trend_all_complexity.append(obj_trend)
            self.valid_trend_all_complexity.append(next_config[3])

            if next_config[3] < best_risk*(1-self.epsilon_r):
                best_risk = next_config[3]
                best_config = next_config
            complexity_level += 1

        if best_config is None:
            w_sel = np.zeros((d, 0))
            t_sel = np.zeros(0)
            obj = -np.inf
            valid_risk = np.inf
        else:
            w_sel, t_sel, obj, valid_risk = best_config
        
        return w_sel, t_sel, obj, valid_risk

    
    @staticmethod
    def _predict(x,w,t):
        l = x@w
        s = l>=-t
        return np.prod(s, axis=1).astype(bool)

    def predict(self, x):
        return self._predict(x, self.w_, self.t_)


class LinRepRuleEnsemble(BaseEstimator):

    def __init__(self, num_rules=3, epsilon_r=0.1, max_complexity=10, reg = 1e-10, max_digits=0,
                task = 'linreg', weighted_logloss = True, no_refit=False, greedy=False, x_std=None, x_mean=None, axis_paral=False):
        self.epsilon_r = epsilon_r
        self.num_rules = num_rules
        self.task = task
        self.weighted_logloss = weighted_logloss
        self.max_complexity = max_complexity
        self.conditions_ = []
        self.reg = reg
        self.no_refit = no_refit
        self.ensembles_ = []
        self.greedy = greedy
        self.x_std = x_std.to_numpy() if isinstance(x_std, pd.Series) else x_std
        self.x_mean = x_mean.to_numpy() if isinstance(x_mean, pd.Series) else x_mean
        self.max_digits = max_digits
        self.axis_paral = axis_paral

    def gradient(self, x, y):
        return y - self.predict(x)
        
    def first_background_rule(self, y):
        if self.task == 'linreg':
            return np.array([y.mean()])
        else:
            y_bar = np.array([y.mean()])
            return np.log(y_bar/(1-y_bar))

    def fit(self, x,y):
        self.feature_name = x.columns
        x_original = x.to_numpy() if isinstance(x, pd.DataFrame) else x
        y_original = y.to_numpy() if isinstance(y, pd.Series) else y

        ### Fixed validation set ###
        # splitter = BootstrapSplitter(reps=1, train_size=len(x_original), replace=True, random_state=42)
        # for train_idx, valid_idx in splitter.split(x_original, y_original):
        #     x_valid, y_valid = x_original[valid_idx], y_original[valid_idx]
        #     x, y = x_original[train_idx], y_original[train_idx]
        random_seeds = [123, 456, 789, 101112, 131415, 161718, 192021, 222324, 252627, 282930, 313233, 
                        343536, 373839, 404142, 434445, 464748, 495051, 525354, 555657, 585960, 616263, 646566, 676869, 
                        707172, 737475, 767778, 798081, 828384, 858687, 888990]
        self.weights_ = self.first_background_rule(y)
        self.time_elapsed = 0
        for _ in range(self.num_rules):
            start_time_one_iteration = time.time()
            ### Variable validation set ###
            splitter = BootstrapSplitter(reps=1, train_size=len(x_original), replace=True, random_state=random_seeds[_])
            for train_idx, valid_idx in splitter.split(x_original, y_original):
                x_valid, y_valid = x_original[valid_idx], y_original[valid_idx]
                x, y = x_original[train_idx], y_original[train_idx]

            self.valid_idx_to_out = valid_idx
            self.train_idx_to_out = train_idx
                
            g = self.gradient(x, y)
            self.gradient_valid_to_out = self.gradient(x_valid, y_valid)
            self.gradient_train_to_out = g
            def predict_fn(temp_condition):
                self.conditions_ = self.conditions_ + [temp_condition]
                res = self.compute_C(x)
                self.estimate_weights(res,y,reg=self.reg)
                valid_pred = self.predict(x_valid)
                if self.task == 'linreg':
                    valid_loss = mean_squared_error(y_valid, valid_pred)
                else:
                    valid_loss = log_loss(y_valid, valid_pred, labels=[0,1])
                self.conditions_.pop()
                return valid_loss

            if _ == 0:
                self.g_to_out = g

            rule = LearnRule(epsilon_r= self.epsilon_r, weighted_logloss=self.weighted_logloss, greedy = self.greedy,
                                          max_complexity=self.max_complexity, no_refit=self.no_refit,
                                            task = self.task, x_std=self.x_std, x_mean=self.x_mean, axis_paral=self.axis_paral).fit(x,y,g,predict_fn)
            if rule.w_.shape[1] != 0:
                self.conditions_.append(rule)
            c = self.compute_C(x)
            rank = np.linalg.matrix_rank(c.T.dot(c))
            # if rank < min(c.T.dot(c).shape):
            #     self.conditions_.pop()
            #     c = self.compute_C(x)
            #     self.estimate_weights(c,y)
                # print(f'break happened after singularity check and number of conditions is {len(self.conditions_)} and shape of weights is {self.weights_.shape} #####################')
                # break
            self.estimate_weights(c,y, reg=self.reg)
            zero_rule_ind = np.where(self.weights_==0)
            if len(zero_rule_ind[0]) > 0:
                self.conditions_.pop(zero_rule_ind[0][0]-1)
                self.weights_ = np.delete(self.weights_, zero_rule_ind[0][0])
            end_time_one_iteration = time.time()
            self.time_elapsed = self.time_elapsed + (end_time_one_iteration - start_time_one_iteration)
            self.obj_trends_to_out = rule.best_obj_trend_to_out
            self.valid_trends_to_out = rule.best_valid_trend_to_out

            copy_model = copy.deepcopy(self)
            # copy_model = self.truncate_ensemble_weights(copy_model, max_digits=self.max_digits)
            self.ensembles_.append(copy_model)
        return self
    
    def compute_loss(self, x, y):
        pred = self.predict(x)
        if self.task == 'linreg':
            loss = mean_squared_error(y, pred)
        else:
            loss = log_loss(y, pred, labels=[0,1])
        return loss
    
    def estimate_weights(self, c, y, reg):
        if self.task == 'linreg':
            linr = Ridge(alpha=reg, fit_intercept=False).fit(c,y)
            self.weights_ = linr.coef_
        elif self.task == 'logreg':
            logr = LogisticRegression(fit_intercept=False, solver='newton-cg', penalty='l2', C=1/reg).fit(c,y)
            self.weights_ = logr.coef_[0]
        
        return self

    
    def compute_C(self, x):
        res = np.zeros(shape=(len(x), len(self.conditions_)+1))
        res[:,0] = 1
        for i, c in enumerate(self.conditions_):
            res[:, i+1] = c.predict(x)
        return res

    def predict(self, x):
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        pred = self.compute_C(x)
        if pred.shape[1] != len(self.weights_):
            print(f'pre shape is {pred.shape} weight vector shape is {self.weights_.shape} and x shape is {x.shape}')
        if self.task == 'linreg':
            out = pred.dot(self.weights_)
        else:
            out = expit_bnd(pred.dot(self.weights_))
        return out
    
    def num_of_rules(self):
        num_rules = 0
        for r in self.conditions_:
            if np.count_nonzero(r.w_) > 0:
                num_rules += 1
        return num_rules

    def num_of_propositions(self):
        num_props = 0
        for r in self.conditions_:
            num_props += (np.where(np.abs(r.w_) > 0, 1, 0).sum(axis=0) != 0).sum()
        return num_props
    
    def model_complexity(self):
        return self.num_of_rules() + self.num_of_propositions() + sum([np.count_nonzero(c.w_) for c in self.conditions_])
    
    def rules_complexity(self):
        return sum([np.count_nonzero(c.w_) for c in self.conditions_])
    
    # def digit_complexity(self):
    #     num_digs_per_rule = []
    #     for r in self.conditions_:
    #         nonzero_w = r.w_[(r.w_ != 0)]
    #         num_digs = [abs(Decimal(Decimal(str(w)).normalize()).as_tuple().exponent) for w in nonzero_w]
    #         num_digs_per_rule.append(np.sum(num_digs))
    #     return np.sum(num_digs_per_rule)


    @staticmethod
    def count_digits(x):
        x = float(x)
        if x == 0:
            return 0

        s = format(x, 'f').lstrip('-')

        if '.' in s:
            int_part, dec_part = s.split('.')
            dec_part = dec_part.rstrip('0')  # remove trailing zeros only

            if int_part == '0':
                # only decimal part matters (keep leading zeros)
                return len(dec_part)
            else:
                return len(int_part) + len(dec_part)
        else:
            return len(s)


    def digit_complexity(self):
        total_digits = 0

        for r in self.conditions_:
            nonzero_w = np.asarray(r.w_)[np.asarray(r.w_) != 0]
            total_digits += sum(self.count_digits(w) for w in nonzero_w)

            nonzero_t = np.asarray(r.t_)[np.asarray(r.t_) != 0]
            total_digits += sum(self.count_digits(t) for t in nonzero_t)

        return total_digits

    # def rules(self, y_mu=None, y_sigma=None, x_mu=None, x_sigma=None, rounding_digits=16):
    #     rule_w = self.weights_
    #     if y_mu is not None and y_sigma is not None:
    #         rule_w = np.round(np.hstack([np.array([self.weights_[0]*y_sigma + y_mu]), self.weights_[1:]*y_sigma]), rounding_digits)
    #     rule_expression = [f"{rule_w[0]:.4f}"]
    #     for r_i, r in enumerate(self.conditions_):
    #         w = r.w_
    #         t = r.t_
    #         if x_mu is not None and x_sigma is not None:
    #             w = w.T / x_sigma
    #             t = t - np.dot(w, x_mu)
    #             w = w.T

    #         # t = t/np.max(np.abs(w), axis=0)
    #         # w = w/np.max(np.abs(w), axis=0)
    #         propos_expression = []
    #         for p_i, p in enumerate(range(w.shape[1])):
    #             terms = []
    #             for i in range(len(self.feature_name)):
    #                 weight = w[i,p_i]
    #                 if weight != 0:
    #                     sign = '+' if weight > 0 and len(terms) > 0 else ''
    #                     if weight == 1:
    #                         terms.append(f"{sign}{self.feature_name[i]}")
    #                     elif weight == -1:
    #                         terms.append(f"{'-'}{self.feature_name[i]}")
    #                     else:
    #                         terms.append(f"{sign}{weight:.{rounding_digits}f} {self.feature_name[i]}")
    #             if w.shape[1] > 1 and p_i != w.shape[1]-1:
    #                 and_ = ' & '
    #             else:
    #                 and_ = ''
    #             propos_expression.append(f" ".join(terms) + f" ≥ {-t[p_i]:.{rounding_digits}f}" + and_)

    #         rule_expression.append(f"{rule_w[r_i+1]:.4f} if " + " ".join(propos_expression))

    #     return rule_expression
    
    def rules(self, y_mu=None, y_sigma=None, x_mu=None, x_sigma=None, rounding_digits=4):
        rule_w = self.weights_
        if y_mu is not None and y_sigma is not None:
            rule_w = np.hstack([np.array([self.weights_[0]*y_sigma + y_mu]), self.weights_[1:]*y_sigma])
        rule_expression = [f"{rule_w[0]:.{rounding_digits}f}"]
        for r_i, r in enumerate(self.conditions_):
            w = r.w_
            t = r.t_
            if x_mu is not None and x_sigma is not None:
                w = w.T / x_sigma.to_numpy() if isinstance(x_sigma, pd.Series) else x_sigma
                t = t - np.dot(w, x_mu.to_numpy() if isinstance(x_mu, pd.Series) else x_mu)
                w = w.T

            # t = t/np.max(np.abs(w), axis=0)
            # w = w/np.max(np.abs(w), axis=0)
            propos_expression = []
            for p_i, p in enumerate(range(w.shape[1])):
                terms = []
                for i in range(len(self.feature_name)):
                    weight = w[i,p_i]
                    if weight != 0:
                        sign = '+' if weight > 0 and len(terms) > 0 else ''
                        if weight == 1:
                            terms.append(f"{sign}{self.feature_name[i]}")
                        elif weight == -1:
                            terms.append(f"{'-'}{self.feature_name[i]}")
                        else:
                            terms.append(f"{sign}{weight:.{rounding_digits}f} {self.feature_name[i]}")
                if w.shape[1] > 1 and p_i != w.shape[1]-1:
                    and_ = ' & '
                else:
                    and_ = ''
                propos_expression.append(f" ".join(terms) + f" ≥ {-t[p_i]:.{rounding_digits}f}" + and_)

            rule_expression.append(f"{rule_w[r_i+1]:.{rounding_digits}f} if " + " ".join(propos_expression))

        return rule_expression
    


class SimplifiedRuleEnsemble:
    def __init__(self, model):
        self.model = model


    @staticmethod
    def decimal_digits_count(x):
        nonzero_w = np.asarray(x)[np.asarray(x) != 0]
        total_digits = sum(SimplifiedRuleEnsemble.count_digits(w) for w in nonzero_w)
        return total_digits
    
    def truncate_decimal_digits_whole_ensemble(self, max_digits=0, round=False):
    # easily rounding the weights and threshold to max_digits number of decimals digits
        for r in self.model.conditions_:
            r.w_ = np.round(r.w_, decimals=16)
            r.w_ = truncate_decimals(r.w_, max_digits, round)
            r.t_ = np.round(r.t_, decimals=16)
            r.t_ = truncate_decimals(r.t_, max_digits, round)
        return self.model

    def truncate_integer_digits_whole_ensemble(self, max_digits=0, gcd=True, round=False):
        for r in self.model.conditions_:
            # fixing the number of decimals to 16 then turning all digits to significant by multiplying with 10^16 
            # and then truncating the number of decimals to max_digits
            wt_stacked = np.vstack((r.w_, r.t_))
            wt_int, max_digits_to_power, min_digits_to_power = scale_columns_to_integers(wt_stacked)
        big_w = wt_int[:r.w_.shape[0], :]
        big_t = wt_int[r.w_.shape[0]:, :][0]
        r.w_ = round_to_n_significant(big_w, max_digits, round)
        r.t_ = round_to_n_significant(big_t, max_digits, round)
        wp = min_shared_power10_per_column(np.vstack((r.w_, r.t_)))
        # wp = min_shared_power10_per_column(r.w_)
        r.w_ = r.w_/(10**wp)
        r.t_ = r.t_/(10**wp)
        if gcd is True:
            gcd_wt = np.gcd.reduce(np.vstack((r.w_, r.t_)).astype(int), axis=0)
            r.w_ = r.w_/gcd_wt
            r.t_ = r.t_/gcd_wt
        # r.t_ = np.round(r.t_/(10**wp),0)
        return self.model


    def truncate_integer_digits_rule_by_rule(self, risk_increase_thrsh=0.01, x=None, y=None, gcd=True, round=False):
        # the goal is to find the minimum number of significant digits that keeps the risk increase
        # less than epsilon_r for each rule
        risk = 0
        risk_original = self.model.compute_loss(x, y)
        for r in self.model.conditions_:
            wt_stacked = np.vstack((r.w_, r.t_))
            wt_int, max_digits_to_power, min_digits_to_power = scale_columns_to_integers(wt_stacked)
            big_w = wt_int[:r.w_.shape[0], :]
            big_t = wt_int[r.w_.shape[0]:, :][0]
            w_list = []
            t_list = []
            for dig in range(max_digits_to_power.max(), -1, -1):
                r.w_ = round_to_n_significant(big_w, dig, round)
                wp = min_shared_power10_per_column(np.vstack((r.w_, r.t_)))
                r.w_ = r.w_/(10**wp)
                r.t_ = round_to_n_significant(big_t, dig, round)
                r.t_ = r.t_/(10**wp)
                # r.t_ = np.round(r.t_/(10**wp),0)
                risk = self.model.compute_loss(x, y)
                if risk - risk_original > risk_increase_thrsh*risk_original:
                    r.w_ = w_list[-1]
                    r.t_ = t_list[-1]
                    break
                w_list.append(r.w_.copy())
                t_list.append(r.t_.copy())
            if gcd is True:
                gcd_wt = np.gcd.reduce(np.vstack((r.w_, r.t_)).astype(int), axis=0)
                r.w_ = r.w_/gcd_wt
                r.t_ = r.t_/gcd_wt
        return self.model


    def truncate_decimal_digits_rule_by_rule(self, risk_increase_thrsh=0.01, x=None, y=None, gcd=True, round=False):
        # the goal is to find the minimum number of decimal digits that keeps the risk increase
        # less than epsilon_r for each proposition in each rule
        risk = 0
        rind = 0
        rng = np.random.default_rng()
        num_rules = len(self.model.conditions_)
        random_rule_ind = rng.permutation(num_rules)
        risk_original = self.model.compute_loss(x, y)
        for r_ind in random_rule_ind:
            r = self.model.conditions_[r_ind]
            wt_stacked = np.vstack((r.w_, r.t_))
            _, max_digits_to_power, _ = scale_columns_to_integers(wt_stacked)
            w_list = []
            t_list = []
            mm = max_digits_to_power.max()
            for dig in range(mm, -1, -1):
                r.w_ = truncate_decimals(r.w_, dig, round)
                r.t_ = truncate_decimals(r.t_, dig, round)
                risk = self.model.compute_loss(x, y)
                if risk - risk_original > risk_increase_thrsh*risk_original:
                    if len(w_list) > 0:
                        r.w_ = w_list[-1]
                        r.t_ = t_list[-1]
                    break
                w_list.append(r.w_.copy())
                t_list.append(r.t_.copy())
            rind += 1
        return self.model


    def truncate_decimal_digits_prop_by_prop(self, risk_increase_thrsh=0.01, x=None, y=None, gcd=True, round=False):
        # the goal is to find the minimum number of decimal digits that keeps the risk increase
        # less than epsilon_r for each proposition in each rule
        risk = 0
        rind = 0
        rng = np.random.default_rng()
        num_rules = len(self.model.conditions_)
        random_rule_ind = rng.permutation(num_rules)
        # risk_original = self.model.compute_loss(x, y)
        for r_ind in random_rule_ind:
            r = self.model.conditions_[r_ind]
            wt_stacked = np.vstack((r.w_, r.t_))
            _, max_digits_to_power, _ = scale_columns_to_integers(wt_stacked)
            num_prop = r.w_.shape[1]
            random_prop_ind = rng.permutation(num_prop)
            for prop in random_prop_ind:
                original_w = r.w_[:, prop].copy()
                original_t = r.t_[prop].copy()
                original_classification = (original_w.dot(x.T)+original_t>=0).astype(bool)
                w_list = []
                t_list = []
                mm = max_digits_to_power[prop]
                for dig in range(mm, -1, -1):
                    r.w_[:, prop] = truncate_decimals(r.w_[:, prop], dig, round)
                    r.t_[prop] = truncate_decimals(r.t_[prop], dig, round)
                    # risk = model.compute_loss(x, y)
                    simplified_classification = (r.w_[:, prop].dot(x.T)+r.t_[prop]>=0).astype(bool)
                    mismaches = np.sum(simplified_classification != original_classification)
                    # if risk - risk_original > risk_increase_thrsh*risk_original:
                    if mismaches > risk_increase_thrsh*len(x):
                        if len(w_list) > 0:
                            r.w_[:, prop] = w_list[-1]
                            r.t_[prop] = t_list[-1]
                        break
                    w_list.append(r.w_[:, prop].copy())
                    t_list.append(r.t_[prop].copy())
            rind += 1
        return self.model

    def truncate_decimal_digits_param_by_param(self, risk_increase_thrsh=0.01, x=None, y=None, gcd=True, seed=None, round=False):
        # the goal is to find the minimum number of decimal digits that keeps the risk increase
        # less than epsilon_r for each parameter per rule
        rng = np.random.default_rng(seed=seed)
        num_rules = len(self.model.conditions_)
        random_rule_ind = rng.permutation(num_rules)
        for r_ind in random_rule_ind:
            r = self.model.conditions_[r_ind]
            num_prop = r.w_.shape[1]
            random_prop_ind = rng.permutation(num_prop)
            for prop in random_prop_ind:
                original_w = r.w_[:, prop].copy()
                original_t = r.t_[prop].copy()
                original_classification = (original_w.dot(x.T)+original_t>=0).astype(bool)
                wt = np.hstack((r.w_[:, prop], r.t_[prop]))
                wt_sel = wt.copy()
                random_param_ind = rng.permutation(len(wt))
                for param_ind in random_param_ind:
                    new_wt = wt.copy()
                    if wt[param_ind] == 0:
                        continue
                    num_dec_dig = num_decimal_digits(wt[param_ind])
                    if num_dec_dig == 0:
                        continue
                    for dig in range(num_dec_dig, -1, -1):
                        trunc_param = truncate_decimals(wt[param_ind], dig, round)
                        new_wt[param_ind] = trunc_param
                        new_classification = (new_wt[:-1].dot(x.T)+new_wt[-1]>=0).astype(bool)
                        mismatches = np.sum(new_classification != original_classification)
                        if mismatches <= int(risk_increase_thrsh*len(original_classification)):
                            wt = new_wt.copy()
                r.w_[:, prop] = wt[:-1]
                r.t_[prop] = wt[-1]
        return self.model


    def truncate_decimal_digits_param_by_param_extensively(self, risk_increase_thrsh=0.01, x=None, y=None, gcd=True,
                                                            seed=None, round=False, number_of_permutations=None, return_all_enumerations=False):
        # the goal is to find the minimum number of decimal digits that keeps the risk increase
        # less than epsilon_r for each parameter per rule
        all_enumerations_per_rule = [] if return_all_enumerations else None
        rng = np.random.default_rng(seed=seed)
        num_rules = len(self.model.conditions_)
        random_rule_ind = rng.permutation(num_rules)
        for r_ind in random_rule_ind:
            all_enumerations_per_proposition = [] if return_all_enumerations else None
            r = self.model.conditions_[r_ind]
            num_prop = r.w_.shape[1]
            random_prop_ind = rng.permutation(num_prop)
            for prop in random_prop_ind:
                all_enumerations = []
                original_w = r.w_[:, prop].copy()
                original_t = r.t_[prop].copy()
                original_classification = (original_w.dot(x.T)+original_t>=0).astype(bool)
                wt = np.hstack((r.w_[:, prop], r.t_[prop]))
                nonzero_coeff_indices = np.where(wt != 0)[0]
                zero_coeff_indices = np.where(wt == 0)[0]
                # wt_sel = wt.copy()
                # random_param_ind = rng.permutation(len(nonzero_coeff_indices))
                all_permutations = list(permutations(nonzero_coeff_indices))
                if number_of_permutations is not None:
                    min_num = np.min((number_of_permutations, len(all_permutations)))
                    all_permutations = all_permutations[:min_num+1]
                best_digits_count = np.inf
                for perm in all_permutations:
                    all_zero_nonzero_coefs = np.concatenate((perm, zero_coeff_indices))
                    for param_ind in all_zero_nonzero_coefs:
                        new_wt = wt.copy()
                        if wt[param_ind] == 0:
                            continue
                        num_dec_dig = num_decimal_digits(wt[param_ind])
                        if num_dec_dig == 0:
                            continue
                        for dig in range(num_dec_dig, -1, -1):
                            trunc_param = truncate_decimals(wt[param_ind], dig, round)
                            new_wt[param_ind] = trunc_param
                            new_classification = (new_wt[:-1].dot(x.T)+new_wt[-1]>=0).astype(bool)
                            mismatches = np.sum(new_classification != original_classification)
                            if mismatches <= int(risk_increase_thrsh*len(original_classification)):
                                wt = new_wt.copy()
                    digits_count = np.sum([LinRepRuleEnsemble.count_digits(w) for w in wt.ravel()])
                    if return_all_enumerations:
                        all_enumerations.append(wt.copy())
                    if digits_count < best_digits_count:
                        best_digits_count = digits_count
                        best_wt = wt.copy()
                        wt = best_wt.copy()
                if return_all_enumerations:
                    all_enumerations_per_proposition.append(all_enumerations)
            
                r.w_[:, prop] = wt[:-1]
                r.t_[prop] = wt[-1]
            if return_all_enumerations:
                all_enumerations_per_rule.append(all_enumerations_per_proposition)
        return self.model, all_enumerations_per_rule

    def truncate_integer_digits_param_by_param(self, risk_increase_thrsh=0.01, x=None, y=None, gcd=True, seed=None, round=False):
        # the goal is to find the minimum number of decimal digits that keeps the risk increase
        # less than epsilon_r for each parameter per rule
        rng = np.random.default_rng(seed=seed)
        num_rules = len(self.model.conditions_)
        random_rule_ind = rng.permutation(num_rules)
        for r_ind in random_rule_ind:
            r = self.model.conditions_[r_ind]
            num_prop = r.w_.shape[1]
            random_prop_ind = rng.permutation(num_prop)
            wt_stacked = np.vstack((r.w_, r.t_))
            wt_int, max_digits_to_power, min_digits_to_power = scale_columns_to_integers(wt_stacked)
            big_w = wt_int[:r.w_.shape[0], :]
            r.w_ = big_w.copy()
            big_t = wt_int[r.w_.shape[0]:, :][0]
            r.t_ = big_t.copy()
            for prop in random_prop_ind:
                original_w = r.w_[:, prop].copy()
                original_t = r.t_[prop].copy()
                original_classification = (original_w.dot(x.T)+original_t>=0).astype(bool)
                wt = np.hstack((r.w_[:, prop], r.t_[prop]))
                random_param_ind = rng.permutation(len(wt))
                mm= max_digits_to_power[prop]
                for param_ind in random_param_ind:
                    new_wt = wt.copy()
                    if wt[param_ind] == 0:
                        continue
                    for dig in range(mm, -1, -1):
                        trunc_param = round_to_n_significant(wt[param_ind], dig, round)
                        new_wt[param_ind] = trunc_param
                        gcd_new_wt = np.gcd.reduce(new_wt.astype(int))
                        new_wt = new_wt/gcd_new_wt if gcd_new_wt != 0 else new_wt
                        new_classification = (new_wt[:-1].dot(x.T)+new_wt[-1]>=0).astype(bool)
                        mismaches = np.sum(new_classification != original_classification)
                        if mismaches <= risk_increase_thrsh*len(x):
                            wt = new_wt.copy()
                r.w_[:, prop] = wt[:-1]
                r.t_[prop] = wt[-1]
        return self.model
        
    def truncate_integer_digits_prop_by_prop(self, risk_increase_thrsh=0.01, x=None, y=None, gcd=True, round=False):
        # the goal is to find the minimum number of significant digits that keeps the risk increase
        # less than epsilon_r for each proposition in each rule
        risk = 0
        rind = 0
        risk_original = self.model.compute_loss(x, y)
        for r in self.model.conditions_:
            wt_stacked = np.vstack((r.w_, r.t_))
            wt_int, max_digits_to_power, min_digits_to_power = scale_columns_to_integers(wt_stacked)
            big_w = wt_int[:r.w_.shape[0], :]
            r.w_ = big_w.copy()
            big_t = wt_int[r.w_.shape[0]:, :][0]
            r.t_ = big_t.copy()
            for prop in range(r.w_.shape[1]):
                w_list = []
                t_list = []
                mm= max_digits_to_power[prop]
                for dig in range(mm, -1, -1):
                    r.w_[:, prop] = round_to_n_significant(big_w[:, prop], dig, round)
                    wp = min_shared_power10_per_column(np.vstack((r.w_[:, prop].reshape(-1, 1), r.t_[prop].reshape(-1, 1))))
                    r.w_[:, prop] = r.w_[:, prop]/(10**wp)
                    r.t_[prop] = round_to_n_significant(big_t[prop], dig, round)
                    r.t_[prop] = r.t_[prop]/(10**wp)
                    # r.t_ = np.round(r.t_/(10**wp),0)
                    risk = self.model.compute_loss(x, y)
                    if risk - risk_original > risk_increase_thrsh*risk_original:
                        if len(w_list) > 0:
                            r.w_[:, prop] = w_list[-1]
                            r.t_[prop] = t_list[-1]
                        else:
                            r.w_[:, prop] = big_w[:, prop]
                            r.t_[prop] = big_t[prop]
                        break
                    w_list.append(r.w_[:, prop].copy())
                    t_list.append(r.t_[prop].copy())
                if gcd is True:
                    gcd_wt = np.gcd.reduce(np.hstack((r.w_[:, prop], r.t_[prop])).astype(int))
                    r.w_[:, prop] = r.w_[:, prop]/gcd_wt if gcd_wt != 0 else r.w_[:, prop]
                    r.t_[prop] = r.t_[prop]/gcd_wt if gcd_wt != 0 else r.t_[prop]
            rind += 1
        return self.model