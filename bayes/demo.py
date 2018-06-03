import numpy as np
from sklearn.cross_validation import train_test_split
np.set_printoptions(predict)

class MultinomialNB(object):
    def __init__(self,alpha=1.0):
        self.alpha = alpha

    def fit(self,X,y):
        count_sample = X.shape[0]
        separated = [[x for x, t in zip(X,y) if t == c ] for c in np.unique(y)]
        self.class_log_prior_ = [np.log(len(i)/count_sample) for i in separated]
        count = np.array([np.array(i).sum(axis=0) for i in separated]) + self.alpha
        self.feature_log_prob_ = np.log(count/count.sum(axis=1)[np.newaxis].T)
        return self

    def predict_log_proba(self,X):
        return [(self.feature_log_prob_*x).sum(axis=1) + self.class_log_prior_ for x in X]

    def predict(self,X):
        return np.argmax(self.predict_log_proba(X),axis=1)


class BernoulliNB(object):
    def __init__(self,alpha=1.0):
        self.alpha = alpha

    def fit(self,X,y):
        count_sample = X.shape[0]
        separated = [[ x for x,t in zip(X,y) if t == c] for c in np.unique(y)]
        self.class_log_prior_ = [np.log(len(i)/count_sample) for i in separated]
        count = np.array([np.array(i).sum(axis=0) for i in separated]) + self.alpha
        smoothing = 2 * self.alpha
        n_doc = np.array([len(i) + smoothing for i in separated])
        self.feature_log_prob_ = count/n_doc[np.newaxis].T
        return self

    def predict_log_proba(self,X):
        return [(np.log(self.feature_prob_) *x + np.log(1 -self.feature_prob_)* np.abs(x -1)).sum(axis=1) + self.class_log_prior_ for x in X]

    def predict(self,X):
        return np.argmax(self.predict_log_proba(X),axis=1)
    

class GaussianNB(object):
    def __init__(self):
        pass
    def fit(self,X,y):
        separated = [[ x for x , t in zip(X,y) if t == c] for c in np.unique(y)]
        self.model = np.array([np.c_[np.mean(i,axis=0),np.std(i,axis=0)]] for i in separated)
        return self
    def _prob(self,x,mean,std):
        exponent = np.exp(-((x -mean) **2/(2 * std**2)))
        return np.log(exponent/(np.sqrt(2*np.pi) * std))
    
    def predict_log_proba(self,X):
        return [[sum(self._prob(i,*s) for s,i in zip(summaries,x)) for summaries in self.model] for x in X]

    def predict(self,X):
        return np.argmax(self.predict_log_proba(X),axis=1)
    
    def score(self,X,y):
        return sum(self.predict(X) == y)/len(y)


X = np.array([[2,1,0,0,0,0],[2,0,1,0,0,0],[1,0,0,1,0,0],[1,0,0,0,1,1]])
y = np.array([0,0,0,1])

nb = MultinomialNB().fit(X,y)
X_test = np.array([[3,0,0,0,1,1],[0,1,1,0,1,1]])
print(nb.predict(X_test))


X_test = np.array([[1,0,0,0,1,1],[1,1,1,0,0,1]])
nb = BernoulliNB(alpha=1).fit(np.where(X>0,1,0),y)
print(nb.predict_log_proba(X_test))

iris = datasets.load_iris()
X,y = iris.data,iris.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.25)
nb = GaussianNB().fit(X_train,y_train)
print(nb.score(X_train,y_train))
print(nb.score(X_test,y_test))
