import numpy as np

def h_rbf(x, centers, cov_mat):
    m = x.shape[0]
    dist = x - centers
    
    if cov_mat.ndim < 2:
        cov_mat = np.eye(m) * cov_mat
    
    norm_factor = 1/np.sqrt((2*np.pi)**m * np.linalg.det(cov_mat))
    h = norm_factor * np.exp(-0.5 * dist @ np.linalg.inv(cov_mat) @ dist.T)

    return h

def random_clustering(X, number):
    N = X.shape[0]
    rand_idx = np.arange(0,N)
    np.random.default_rng().shuffle(rand_idx)
    centers = []
    radii = []
    y = []

    for i in range(number):
        center = X[rand_idx[(i*number)%N:((i+1)*number)%N], :]
        centers.append(np.mean(center, axis=0))
        radii.append(np.mean(np.var(center, axis=1)))
    
    for sample in X:
        distance = np.sqrt(np.sum((sample - centers)**2, axis=1))
        label = np.argmin(distance)
        y.append(label)
    
    y = np.array(y)

    return np.array(centers), np.array(radii), y.reshape((-1,1))

def rand_rbf(x_train, y_train, hidden_dim):
    centers, radii, labels = random_clustering(x_train, hidden_dim)
    N = x_train.shape[0]
    h = np.zeros((N, hidden_dim))

    for p in range(hidden_dim):
        p_samples = (labels == p).nonzero()
        h[:,p] = np.array([h_rbf(x_sample, centers[p], radii[p]) for x_sample in x_train])
    
    h = np.nan_to_num(h)
    h_aug = np.append(np.ones((N, 1)), h, axis=1)
    #w = np.linalg.pinv(h_aug) @ y_train
    w, *_ = np.linalg.lstsq(h_aug, y_train)
    w = np.nan_to_num(w)

    return (h, w, centers, radii)

def eval_RBF(x_test, rbf_params):
    w = rbf_params[1]
    hidden_dim = rbf_params[0].shape[1]
    centers = rbf_params[2]
    cov_list = rbf_params[3]
    N = x_test.shape[0]
    h = np.zeros((N, hidden_dim))
    
    for p in range(hidden_dim):
        cov_mat = np.copy(cov_list[p])
        h[:,p] = np.array([h_rbf(x_sample, centers[p], cov_mat) for x_sample in x_test])
    
    #h = np.nan_to_num(h)
    h = np.append(np.ones((N, 1)), h, axis=1)
    y_hat = h @ w
    
    return y_hat

def distance(X, C, metric = 'euclidean'):
    if metric == 'euclidean':
        dist = np.sum((X - C[:, None])**2, axis=2)
    elif metric == 'supremum':
        dist = (np.max(np.abs(X - C[:, None]), axis=2))
    elif metric == 'manhattan':
        dist = (np.sum(np.abs(X - C[:, None]), axis=2))
    elif metric == 'cosine':
        dist = (C @ X.T)**2/(np.sum(X**2, axis=1) * np.sum(C**2, axis=1)[:, None])
    return dist

def prob_kmp(dist, cost):
    prob = np.min(dist, axis=0) / cost
    return prob

def cost_kmp(dist):
    return np.sum(np.min(dist,axis=0))

def sample_kmp(X, prob, l):
    X = X.copy()
    rng = np.random.default_rng()
    samples = rng.choice(X,l, p=prob)

    return samples

def rnd_clustering(samples, n_cluster, metric = 'euclidean', r=5, l=-1):
    X = samples.copy()
    N = samples.shape[0]
    rng = np.random.default_rng()
    C_set = X[rng.integers(N, size=1), :]

    if l == -1:
        l = 2*samples.shape[1]

    for _ in range(r):
        dist = distance(X, C_set, metric)
        cost = cost_kmp(dist)
        prob = prob_kmp(dist, cost)
        C_temp = sample_kmp(X, prob, l)
        C_set = np.r_[C_set, C_temp]
    
    dist = distance(X, C_set, metric)
    closest = np.zeros(dist.shape)
    closest[np.argmin(dist, axis=0), range(dist.shape[1])] = 1
    count = np.sum(closest, axis=1)
    best_c = np.argsort(count)[::-1]
    #weights = count/np.sum(count)
    
    return C_set[best_c[:n_cluster],:]

def rnd_rbf(x_train, y_train, hidden_dim, r=5, l=10, metric= 'euclidean'):
    pass

