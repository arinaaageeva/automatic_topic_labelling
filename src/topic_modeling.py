import artm

#from sklearn import metrics
from sklearn.base import BaseEstimator, ClusterMixin

class TopicModeling(BaseEstimator, ClusterMixin):
    
    def __init__(self, dictionary, 
                 num_topics=3, num_background_topics=0,
                 num_document_passes=10, num_collection_passes=10,
                 theta_columns_naming='title',
                 class_ids=None, smooth_background_class_ids=None, decorrelat_phi_class_ids=None,
                 num_top_tokens=10):
        
        self.num_collection_passes = num_collection_passes
        
        topic_names = [f'topic_{index}' for index in range(num_topics)]
        background_names = [f'background_{index}' for index in range(num_background_topics)]
        
        self.model = artm.ARTM(dictionary=dictionary, 
                               topic_names=topic_names+background_names,
                               class_ids=class_ids, 
                               theta_columns_naming=theta_columns_naming, 
                               cache_theta=True)
        
        #self.model.scores.add(artm.PerplexityScore(name='perplexity', dictionary=dictionary))
        
        if class_ids is not None:
            for class_id, weight in class_ids.items():
                if weight > 0:
                    self.model.scores.add(artm.TopTokensScore(name=f'{class_id}_top_tokens', class_id=class_id, num_tokens=num_top_tokens))
        
        #smooth background
        if (num_background_topics > 0) and (smooth_background_class_ids is not None):
            for class_id, tau in smooth_background_class_ids.items():
                if tau:
                    self.model.regularizers.add(artm.SmoothSparsePhiRegularizer(name=f'smooth_background_{class_id}', tau=tau, gamma=0, class_ids=class_id,
                                                                                topic_names=background_names))
        
        #decorrelator phi
        if decorrelat_phi_class_ids is not None:
            for class_id, tau in decorrelat_phi_class_ids.items():
                if tau:
                    self.model.regularizers.add(artm.DecorrelatorPhiRegularizer(name=f'decorrelator_phi_{class_id}', tau=tau, gamma=0, class_ids=class_id))
        
    def fit(self, X, y=None):
        self.model.fit_offline(X, num_collection_passes=self.num_collection_passes)
        self.labels_ = self.model.get_theta().T.idxmax(axis=1)
        return self
        
    def predict(self, X):
        return self.model.transform(X).T.idxmax(axis=1)