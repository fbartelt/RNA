import numpy as np
from sklearn.cluster import KMeans

class km_rbf:
    def __init__(self, x_train, y_train, hidden_dim, indep = False):
        self.n_clusters = hidden_dim
        km = KMeans(hidden_dim).fit(x_train)
        self.centers = km.cluster_centers_
        self.labels = km.labels_
        self.w, self.cov_list = self._train(x_train, y_train, hidden_dim, indep)
        

    def _h_response(self, x, centers, cov_mat):
        m = x.shape[0]
        dist = x - centers
        
        if cov_mat.ndim < 2:
            cov_mat = np.eye(m) * cov_mat
        
        norm_factor = 1/np.sqrt((2*np.pi)**m * np.linalg.det(cov_mat))
        h = norm_factor * np.exp(-0.5 * dist @ np.linalg.inv(cov_mat) @ dist.T)

        return h

    def _train(self, x_train, y_train, hidden_dim, indep=False):
        N = x_train.shape[0]
        m = x_train.shape[1]
        cov_list = []
        H = np.zeros((N, hidden_dim))

        for p in range(hidden_dim):
            p_samples = (self.labels == p).nonzero()
            
            if indep:
                cov_mat = np.diag(np.var(x_train[p_samples], axis=0)) + 0.001*np.diag(np.ones(m))
            else:
                cov_mat = np.cov(x_train[p_samples], rowvar=False) + 0.001*np.diag(np.ones(m))
            
            cov_list.append(np.copy(cov_mat))
            #center = centers[p,:]
            H[:,p] = np.array([self._h_response(x_sample, self.centers[p], cov_mat) 
                              for x_sample in x_train])
        
        H_aug = np.append(np.ones((N, 1)), H, axis=1)
        #w, *_ = np.linalg.lstsq(h_aug, y_train)
        #w = np.nan_to_num(w)
        w = np.linalg.pinv(H_aug) @ y_train

        return (w, cov_list)

    def eval(self, x_test):
        N = x_test.shape[0]
        h = np.zeros((N, self.n_clusters))
        
        for p in range(self.n_clusters):
            cov_mat = np.copy(self.cov_list[p])
            h[:,p] = np.array([self._h_response(x_sample, self.centers[p], cov_mat) 
                              for x_sample in x_test])
        
        #h = np.nan_to_num(h)
        h = np.append(np.ones((N, 1)), h, axis=1)
        y_hat = h @ self.w
        
        return y_hat
