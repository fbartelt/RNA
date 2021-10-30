import numpy as np

class rnd_rbf:
    def __init__(self, x_train, y_train, hidden_dim, metric, same_radii = True, mean = True, r=5, l=-1):
        self.n_clusters = hidden_dim
        self.metric = metric
        self.centers, self.labels = self._rnd_clustering(x_train, mean = mean, r = r, l = l)
        self.radii = self._get_radii(x_train, same_radii)
        self.w = self._train(x_train, y_train, hidden_dim)
        
    def distance(self, X, C, metric = 'euclidean'):
        if metric == 'euclidean':
            dist = np.sum((X - C[:, None])**2, axis=2)
        elif metric == 'supremum':
            dist = (np.max(np.abs(X - C[:, None]), axis=2))
        elif metric == 'manhattan':
            dist = (np.sum(np.abs(X - C[:, None]), axis=2))
        elif metric == 'cosine':
            dist = (C @ X.T)**2/(np.sum(X**2, axis=1) * np.sum(C**2, axis=1)[:, None])
        return dist

    def _prob_kmp(self, dist, cost):
        prob = np.min(dist, axis=0) / cost
        return prob

    def _cost_kmp(self, dist):
        return np.sum(np.min(dist,axis=0))

    def _sample_kmp(self, X, prob, l):
        X = X.copy()
        rng = np.random.default_rng()
        samples = rng.choice(X,l, p=prob)

        return samples

    def _h_response(self, x, centers, cov_mat):
        m = x.shape[0]
        dist = x - centers
        
        if cov_mat.ndim < 2:
            cov_mat = np.eye(m) * cov_mat
        
        cov_mat = cov_mat + 0.001*np.diag(np.ones(m))
        norm_factor = 1/np.sqrt((2*np.pi)**m * np.linalg.det(cov_mat))
        h = norm_factor * np.exp(-0.5 * dist @ np.linalg.inv(cov_mat) @ dist.T)

        return h

    def _rnd_clustering(self, samples, mean = True, r=5, l=-1):
        X = samples.copy()
        N = samples.shape[0]
        rng = np.random.default_rng()
        C_set = X[rng.integers(N, size=1), :]

        if l == -1:
            l = 2*self.n_clusters

        for _ in range(r):
            dist = self.distance(X, C_set, self.metric)
            cost = self._cost_kmp(dist)
            prob = self._prob_kmp(dist, cost)
            C_temp = self._sample_kmp(X, prob, l)
            C_set = np.r_[C_set, C_temp]
        
        dist = self.distance(X, C_set, self.metric)
        closest = np.zeros(dist.shape)
        closest[np.argmin(dist, axis=0), range(dist.shape[1])] = 1
        count = np.sum(closest, axis=1)
        best_c = np.argsort(count)[::-1]
        if mean and self.n_clusters > 2:
            a = C_set[best_c].copy()
            fixed = int(np.floor(self.n_clusters/2))+1
            #temp = a[0:fixed]
            temp = np.array_split(a[fixed::], self.n_clusters-fixed)
            m = np.array([np.mean(chunk, axis=0) for chunk in temp])
            C_set = np.r_[a[:fixed], m]
        else:
            C_set = C_set[best_c[:self.n_clusters],:]
        labels = np.argmin(self.distance(X, C_set, self.metric), axis=0)
        #weights = count/np.sum(count)
        
        return (C_set, labels)

    def _get_radii(self, X, same = True):
        C = self.centers.copy()
        X = X.copy()

        if same:
            radius = np.min(np.mean((self.distance(C, C, self.metric)), axis=0))
            radii = radius * np.ones((self.n_clusters, 1))
            ## [[radius]] * self.n_clusters
        else:
            radii = []
            for label in range(self.n_clusters):
                radius = np.max(self.distance(X[np.nonzero(self.labels == label)], C[label, None],
                                self.metric))
                radii.append(radius)
        radii = np.array(radii)

        if self.metric == 'euclidean':
            radii = np.sqrt(radii)

        return radii

    def _train(self, x_train, y_train, hidden_dim):
        #centers, radii, labels = random_clustering(x_train, hidden_dim)
        N = x_train.shape[0]
        H = np.zeros((N, hidden_dim))

        for p in range(hidden_dim):
            #p_samples = (self.labels == p).nonzero()
            H[:,p] = np.array([self._h_response(x_sample, self.centers[p], self.radii[p]) 
                              for x_sample in x_train])
        
        #h = np.nan_to_num(h)
        H_aug = np.append(np.ones((N, 1)), H, axis=1)
        #w = np.linalg.pinv(h_aug) @ y_train
        #w, *_ = np.linalg.lstsq(h_aug, y_train)
        #w = np.nan_to_num(w)
        w = np.linalg.pinv(H_aug) @ y_train

        return w

    def eval(self, x_test):
        N = x_test.shape[0]
        H = np.zeros((N, self.n_clusters))
        
        for p in range(self.n_clusters):
            cov_mat = np.copy(self.radii[p])
            H[:,p] = np.array([self._h_response(x_sample, self.centers[p], cov_mat) 
                              for x_sample in x_test])
        
        #h = np.nan_to_num(h)
        H = np.append(np.ones((N, 1)), H, axis=1)
        y_hat = H @ self.w
        
        return y_hat