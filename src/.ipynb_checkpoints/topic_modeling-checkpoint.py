import artm

from sklearn import metrics
from sklearn.base import BaseEstimator, ClusterMixin

class TopicModeling(BaseEstimator, ClusterMixin):
    
    def __init__(self, dictionary, num_topics=3, num_collection_passes=10, class_ids=None,
                 smooth_background_tau=0, sparse_phi_tau=0, decorrelator_phi_tau=0):
        
        self.num_collection_passes = num_collection_passes
        
        topic_names = [f'topic_{index}' for index in range(num_topics)]
        self.model = artm.ARTM(topic_names=topic_names+['background'], 
                               dictionary=dictionary, class_ids=class_ids,
                               cache_theta = True, theta_columns_naming='title')
        
        self.model.scores.add(artm.PerplexityScore(name='perplexity', dictionary=dictionary))
        self.model.scores.add(artm.TopTokensScore(name='per_top_tokens', class_id='per', num_tokens=100))
        self.model.scores.add(artm.TopTokensScore(name='loc_top_tokens', class_id='loc', num_tokens=100))
        self.model.scores.add(artm.TopTokensScore(name='org_top_tokens', class_id='org', num_tokens=100))
        self.model.scores.add(artm.TopTokensScore(name='title_top_tokens', class_id='title', num_tokens=100))
        self.model.scores.add(artm.TopTokensScore(name='text_top_tokens', class_id='text', num_tokens=100))
        
        #smooth background
        self.model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='smooth_background', 
                                                                    tau=smooth_background_tau, gamma=0,
                                                                    topic_names=['background']))
        
        #sparse phi
        self.model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='sparse_phi',
                                                                    tau=-sparse_phi_tau, gamma=0,
                                                                    topic_names=topic_names))
        
        #decorrelator phi
        self.model.regularizers.add(artm.DecorrelatorPhiRegularizer(name='decorrelator_phi',
                                                                    tau=decorrelator_phi_tau, gamma=0))
        
    def fit(self, X, y=None):
        self.model.fit_offline(X, num_collection_passes=self.num_collection_passes)
        self.labels_ = self.model.get_theta().T.idxmax(axis=1)
        return self
        
    def predict(self, X):
        return sefl.model.transform(X).T.idxmax(axis=1)
    
def get_metrics(labels_true, labels_pred):
    
    labels_true = labels_true.set_index('id')
    labels_pred = labels_pred[~(labels_pred == 'background')]
    
    labels_true, labels_pred = zip(*[(labels_true.xs(int(index)).hr_level_0, labels_pred[index]) 
                                     for index in labels_pred.index])
    
    return metrics.adjusted_rand_score(labels_true, labels_pred), \
           metrics.adjusted_mutual_info_score(labels_true, labels_pred, average_method='arithmetic'), \
           metrics.homogeneity_score(labels_true, labels_pred), \
           metrics.fowlkes_mallows_score(labels_true, labels_pred)

def print_metrics(labels_true, labels_pred):
    
    metric_names = ['Adjusted Rand index',
                    'Mutual Information based scores',
                    'Homogeneity, completeness and V-measure',
                    'Fowlkes-Mallows scores']
    metric_scores = get_metrics(labels_true, labels_pred)
    
    for name, score in zip(metric_names, metric_scores):
        print(f'{name} {score}\n')
    
def get_background_articles(articles, labels_pred):
    
    background_ids = labels_pred[labels_pred == 'background'].index
    return articles[articles.id.isin(background_ids.astype(int))]

def print_background_articles(articles, labels_pred):
    
    background_articles = get_background_articles(articles, labels_pred)
    
    result = ''
    for level_0, group in background_articles.groupby('hr_level_0'):
        result += f'LEVEL 0: {level_0}\n\n'
        for title in group.title.values:
            result += f'{title}\n'
        result += '\n'
        
    return result