import os
import shutil
import artm
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, ClusterMixin

def _generate_names(number, names, preffix=''):
    
    if (number is None) and (names is None):
        number = 0
        
    return [f'{preffix}_{index}' for index in range(number)] if names is None else list(names)

def _generate_names_levels(number_levels, number_names_levels, names_levels, preffix):
    
    if names_levels is None:
        names_levels = number_levels*[None]
    
    if number_names_levels is None:
        number_names_levels = [0 if names is None else len(names) for names in names_levels]
    
    return [_generate_names(number, names, preffix) for number, names in zip(number_names_levels, names_levels)]        

class TopicModel(BaseEstimator, ClusterMixin):
    
    def __init__(self, 
                 dictionary, 
                 class_ids, 
                 tmp_files_path='', 
                 theta_columns_naming='title',
                 cache_theta = True,
                 num_levels=None, 
                 level_names=None, 
                 num_topics=None, 
                 topic_names=None, 
                 num_backgrounds=None, 
                 background_names=None,
                 smooth_background_tau=None,
                 decorrelate_phi_tau=None,
                 parent_topics_proportion=None,
                 spars_psi_tau=None,
                 smooth_theta_fit=1.0,
                 num_collection_passes=1,
                 num_tokens=10):
        
        self.model = artm.hARTM(dictionary=dictionary, 
                                class_ids=class_ids, 
                                theta_columns_naming=theta_columns_naming, 
                                tmp_files_path=tmp_files_path, 
                                cache_theta=cache_theta)
        
        self.level_names = _generate_names(num_levels, level_names, 'level')
        
        topic_names = _generate_names_levels(len(self.level_names), num_topics, topic_names, 'topic')
        background_names = _generate_names_levels(len(self.level_names), num_backgrounds, background_names, 'background')
            
        for topic_names_level, background_names_level in zip(topic_names, background_names):
            
            topic_names_level = topic_names_level + background_names_level
            level = self.model.add_level(num_topics=len(topic_names_level), topic_names=topic_names_level)
            
        if smooth_background_tau is not None:
            
            for level, background_names_level in zip(self.model, background_names):
                level.regularizers.add(artm.SmoothSparsePhiRegularizer('SPhi_back', 
                                                                       tau=smooth_background_tau, 
                                                                       gamma=0,
                                                                       topic_names=background_names_level))
            
        if decorrelate_phi_tau is not None:
            
            for level in self.model:
                level.regularizers.add(artm.DecorrelatorPhiRegularizer('DPhi', tau=decorrelate_phi_tau, gamma=0))
            
        if (parent_topics_proportion is not None) and (spars_psi_tau is not None):
            
            for level, parent_topics_proportion_level in zip(self.model[1:], parent_topics_proportion):
                
                for topic_name, parent_topic_proportion in parent_topics_proportion_level.items(): 
                    level.regularizers.add(artm.HierarchySparsingThetaRegularizer(name=f'HSTheta_{topic_name}', 
                                                                                  topic_names=topic_name, 
                                                                                  tau=spars_psi_tau, 
                                                                                  parent_topic_proportion=parent_topic_proportion))
                    
        self.smooth_theta_fit = smooth_theta_fit
        self.num_collection_passes = num_collection_passes
                    
        for level in self.model:
            
            for class_id, weight in class_ids.items():
                
                if weight > 0:
                    level.scores.add(artm.TopTokensScore(name=f'TT_{class_id}', class_id=class_id, num_tokens=num_tokens))
    
    def get_top_tokens(self):
        
        top_tokens = []
        for level in self.model:
           
            level_top_tokens = {}
            for modality, modality_top_tokens_tracker in level.score_tracker.items():
                
                for topic_name, modality_top_tokens in modality_top_tokens_tracker.last_tokens.items():
                    level_top_tokens.setdefault(topic_name, {})
                    level_top_tokens[topic_name][modality] = modality_top_tokens
                    
            top_tokens.append(level_top_tokens)
                    
        return top_tokens
    
    def fit_supervised(self, model, X, y):
                                           
        len_y = len(y)
        topic_names = model.topic_names
        
        doc_topic_coef = np.zeros((len_y, model.num_topics))
        doc_topic_coef[range(len_y), [topic_names.index(topic_name) for topic_name in y]] = 1.0
        
        model.regularizers.add(artm.SmoothSparseThetaRegularizer(name='SST', 
                                                                 tau=self.smooth_theta_fit, 
                                                                 doc_titles=y.index, 
                                                                 doc_topic_coef=doc_topic_coef.tolist()))
        
    def fit(self, X, y=None):
        
        if y is not None:
            
            for level_name, y_level in y.iteritems():
                level_index = self.level_names.index(level_name)
                self.fit_supervised(self.model[level_index], X, y_level)
            
        self.model.fit_offline(X, num_collection_passes=self.num_collection_passes)
        self.labels_ = self.predict(X)
        
        return self
    
    def predict_proba(self, X):
        return [level.transform(X).T for level in self.model]
        
    def predict(self, X):
        return pd.concat([level.idxmax(axis=1).rename(level_name) for level_name, level in zip(self.level_names, self.predict_proba(X))], axis=1)
    
class KFoldTM:
    
    def __init__(self, n_splits=3, shuffle=True, random_state=None):
        self.kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
    def validation(self, 
                   X, 
                   y, 
                   scores,
                   class_ids,
                   num_epoch=10,
                   tmp_files_path='tmp', 
                   theta_columns_naming='title', 
                   num_levels=None, 
                   level_names=None, 
                   num_topics=None, 
                   topic_names=None, 
                   num_backgrounds=None, 
                   background_names=None,
                   smooth_background_tau=None,
                   decorrelate_phi_tau=None,
                   parent_topics_proportion=None,
                   spars_psi_tau=None,
                   smooth_theta_fit=1.0,
                   num_collection_passes=1):
        
        kfold_scores = {name:[] for name in scores.keys()}
        for train_indecies, test_indecies in self.kfold.split(X.values, y.values):
            
            os.mkdir(tmp_files_path)
            
            X_train, X_test = X.iloc[train_indecies], X.iloc[test_indecies]
            y_train, y_test = y.iloc[train_indecies], y.iloc[test_indecies]
            
            with open(os.path.join(tmp_files_path, 'X_train.txt'), 'w') as fl:
                fl.write('\n'.join(X_train.index + ' ' + X_train.values)+'\n')
                
            with open(os.path.join(tmp_files_path, 'X_test.txt'), 'w') as fl:
                fl.write('\n'.join(X_test.index + ' ' + X_test.values)+'\n')
                
            X_train = artm.BatchVectorizer(data_path=os.path.join(tmp_files_path, 'X_train.txt'), 
                                           data_format='vowpal_wabbit',
                                           target_folder=os.path.join(tmp_files_path, 'X_train'))
            
            X_test = artm.BatchVectorizer(data_path=os.path.join(tmp_files_path, 'X_test.txt'), 
                                          data_format='vowpal_wabbit', 
                                          target_folder=os.path.join(tmp_files_path, 'X_test'))
            
            topic_model = TopicModel(X_train.dictionary, 
                                     class_ids=class_ids, 
                                     tmp_files_path=tmp_files_path, 
                                     theta_columns_naming=theta_columns_naming, 
                                     num_levels=num_levels, 
                                     level_names=level_names, 
                                     num_topics=num_topics, 
                                     topic_names=topic_names, 
                                     num_backgrounds=num_backgrounds, 
                                     background_names=background_names,
                                     smooth_background_tau=smooth_background_tau,
                                     decorrelate_phi_tau=decorrelate_phi_tau,
                                     parent_topics_proportion=parent_topics_proportion,
                                     spars_psi_tau=spars_psi_tau,
                                     smooth_theta_fit=smooth_theta_fit,
                                     num_collection_passes=num_collection_passes)

            topic_model.fit(X_train, y_train)
            y_pred = topic_model.predict(X_test).loc[y_test.index]
            
            calculation_score = lambda score, y_pred, y: [score(y_pred[column], y[column]) for column in y.columns]
            epoch_scores = {name:[calculation_score(score, y_pred, y_test)] for name, score in scores.items()}
            
            for _ in range(num_epoch):
                
                topic_model.fit(X_train)
                y_pred = topic_model.predict(X_test).loc[y_test.index]
                
                for name, score in scores.items():
                    epoch_scores[name].append(calculation_score(score, y_pred, y_test))
                    
            for name, epoch_score in epoch_scores.items():
                kfold_scores[name].append(epoch_score)
        
            shutil.rmtree(tmp_files_path)
            
        return {name:np.array([list(zip(*score)) for score in kfold_score]).mean(axis=0) for name, kfold_score in kfold_scores.items()}      
            