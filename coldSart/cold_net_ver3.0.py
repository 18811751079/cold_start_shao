from data import Data_coldStart
import numpy as np
import logging

DATAPATH = r'E:\PPPP___Python\test_data\hetrec2011-movielens-2k-v2'
EPOCH = 30
LEARNRATE = 0.02
DECAY = 0.95
DIM = 10
FEATURE = 2
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")

log_writer = logging.FileHandler('cold_net_3.0.log', 'a')
log_writer.setLevel(logging.DEBUG)
log_writer.setFormatter(formatter)
log_print = logging.StreamHandler()
log_print.setLevel(logging.INFO)
log_print.setFormatter(formatter)

logger.addHandler(log_writer)
logger.addHandler(log_print)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def vector_sigmoid(vector):
    sig_vec = np.zeros_like(vector)
    length = vector.shape[0]
    for i in range(length):
        sig_vec[i] = round(sigmoid(vector[i]), 3)
    return sig_vec


class cold_model():
    def __init__(self, act_num, country_num, director_num,
                 genre_num, users, dim=10, feature=4):

        self.dim = dim
        self.act_num = act_num
        self.cty_num = country_num
        self.drt_num = director_num
        self.genre_num = genre_num
        self.feature = feature
        # printformat = 'dim=%d, act_num=%d, cty_num=%d, drt_num=%d, genre_num=%d'
        logger.debug('dim=%d, act_num=%d, cty_num=%d, drt_num=%d, genre_num=%d' % (dim, act_num, country_num, director_num, genre_num))
        self.actor_w = np.random.uniform(0, 0.1, (dim, act_num))
        self.country_w = np.random.uniform(0, 0.1, (dim, country_num))
        self.director_w = np.random.uniform(0, 0.1, (dim, director_num))
        self.genre_w = np.random.uniform(0, 0.1, (dim, genre_num))

        self.actor_v = np.random.uniform(0, 0.1, dim)
        self.country_v = np.random.uniform(0, 0.1, dim)
        self.director_v = np.random.uniform(0, 0.1, dim)
        self.genre_v = np.random.uniform(0, 0.1, dim)

        self.P = {}
        self.mk_user_mat(users)
        self.init_gradient_param()

    def mk_user_mat(self, users):
        for user in users:
            self.P[user] = np.random.uniform(0, 1, self.feature)

    def prediction(self, actor, country, director, genre, user):
        # vector shape = dim
        self.actor_f = vector_sigmoid(np.dot(self.actor_w, actor))
        self.country_f = vector_sigmoid(np.dot(self.country_w, country))
        self.director_f = vector_sigmoid(np.dot(self.director_w, director))
        self.genre_f = vector_sigmoid(np.dot(self.genre_w, genre))

        # scalar shape = 1
        actor_n = np.dot(self.actor_f, self.actor_v)
        country_n = np.dot(self.country_f, self.country_v)
        director_n = np.dot(self.director_f, self.director_v)
        genre_n = np.dot(self.genre_f, self.genre_v)

        self.vector_n = np.array([actor_n, country_n, director_n, genre_n])

        return round(np.dot(self.P[user], self.vector_n), 2)

    def init_gradient_param(self):
        '''

        :param users: ????????????
        :description: ????????????????????????????????????
        '''
        self.ud_P = np.zeros(self.feature)

        self.ud_actor_w = np.zeros_like(self.actor_w)
        self.ud_country_w = np.zeros_like(self.actor_w)
        self.ud_director_w = np.zeros_like(self.director_w)
        self.ud_genre_w = np.zeros_like(self.genre_w)

        self.ud_actor_v = np.zeros_like(self.actor_v)
        self.ud_country_v = np.zeros_like(self.country_v)
        self.ud_director_v = np.zeros_like(self.director_v)
        self.ud_genre_v = np.zeros_like(self.genre_v)

    def backward(self, user, eui, actor, country, director, genre, maxgrad=3):
        for i in range(self.feature):
            self.ud_P[i] = -eui * self.vector_n[i]

        for t in range(self.dim):
            self.ud_actor_v[t] = -eui * self.P[user][0] * self.actor_f[t]
            self.ud_country_v[t] = -eui * self.P[user][1] * self.country_f[t]
            self.ud_director_v[t] = -eui * self.P[user][2] * self.director_f[t]
            self.ud_genre_v[t] = -eui * self.P[user][3] * self.genre_f[t]

            mid_act = self.ud_actor_v[t] * self.actor_v[t] * (1 - self.actor_f[t])
            mid_country = self.ud_country_v[t] * self.country_v[t] * (1 - self.country_f[t])
            mid_director = self.ud_director_v[t] * self.director_v[t] * (1 - self.director_f[t])
            mid_genre = self.ud_genre_v[t] * self.genre_v[t] * (1 - self.genre_f[t])

            for k in range(self.act_num):
                self.ud_actor_w[t][k] = mid_act * actor[k]
            for k in range(self.cty_num):
                self.ud_country_w[t][k] = mid_country * country[k]
            for k in range(self.drt_num):
                self.ud_director_w[t][k] = mid_director * director[k]
            for k in range(self.genre_num):
                self.ud_genre_w[t][k] = mid_genre * genre[k]

        # print('ud_P', self.ud_P)
        # print('\nud_actor_v', self.actor_v)
        # print('\nud_country_v', self.ud_country_v)
        # print('\nud_dirctor_v', self.ud_director_v)
        # print('\nud_genre_v', self.ud_genre_v)
        # print('\nud_actor_w', self.ud_actor_w)
        # print('\nud_country_w', self.ud_country_w)
        # print('\nud_director_w', self.ud_director_w)
        # print('\nud_genre_w\n\n', self.ud_genre_w)

    def update(self, user, lr=0.01):
        for i in range(self.feature):
            self.P[user][i] -= lr * self.ud_P[i]

        for t in range(self.dim):
            self.actor_v[t] -= lr * self.ud_actor_v[t]
            self.country_v[t] -= lr * self.ud_country_v[t]
            self.country_v[t] -= lr * self.ud_director_v[t]
            self.genre_v[t] -= lr * self.ud_genre_v[t]

        for t in range(self.dim):
            for k in range(self.act_num):
                self.actor_w[t][k] -= lr * self.ud_actor_w[t][k]

            for k in range(self.cty_num):
                self.country_w[t][k] -= lr * self.ud_country_w[t][k]

            for k in range(self.drt_num):
                self.director_w[t][k] -= lr * self.ud_director_w[t][k]

            for k in range(self.genre_num):
                self.genre_w[t][k] -= lr * self.ud_genre_w[t][k]

    def train(self, train, actors, countries, directors, genres, lr, epoch=10 ** 6):
        MAE = 0
        RMSE = 0
        for j, (user, item, rating) in enumerate(train):
            eui = rating - self.prediction(actors[item], countries[item],
                                           directors[item], genres[item], user)
            # print(eui)
            self.backward(user, eui, actors[item], countries[item],
                          directors[item], genres[item])
            self.update(user, lr)

            MAE += abs(eui)
            RMSE += eui ** 2
        return MAE, RMSE

    def test(self, test, actors, countries, directors, genres):
        MAE = 0
        RMSE = 0
        for j, (user, item, rating) in enumerate(test):
            eui = rating - self.prediction(actors[item], countries[item],
                                           directors[item], genres[item], user)
            MAE += abs(eui)
            RMSE += eui ** 2
        return MAE, RMSE

    def save_param(self):
        path = 'model'
        import os

        if not os.path.exists(path):
            os.makedirs(path)

        self.writefile(path + 'actors_w.param', self.actor_w)
        self.writefile(path + 'country_w.param', self.country_w)
        self.writefile(path + 'director_w', self.director_w)
        self.writefile(path + 'genre_w', self.genre_w)

        self.writefile(path + 'actor_v', self.actor_v)
        self.writefile(path + 'country_v', self.country_v)
        self.writefile(path + 'director_v', self.director_v)
        self.writefile(path + 'genre_v', self.genre_v)

        self.writefile(path + 'P', self.P)

        print('saving the model is successful!')

    def writefile(self, filename, param):
        with open(filename, 'w') as f:
            f.writelines(str(param))


if __name__ == '__main__':

    data = Data_coldStart()
    data.get_actor(DATAPATH + r'\movie_actors.dat', weight=20)
    data.get_country(DATAPATH + r'\movie_countries.dat')
    data.get_director(DATAPATH + r'\movie_directors.dat')
    data.get_genres(DATAPATH + r'\movie_genres.dat', weight=30)
    data.get_data(DATAPATH + r'\user_ratedmovies.dat')
    # data.read_data()
    actor_num, country_num, director_num, genre_num = data.size()
    CM = cold_model(actor_num, country_num, director_num, genre_num, data.train.users, DIM)
    train_len = len(data.train.ratings)
    test_len = len(data.test.ratings)

    for i in range(EPOCH):
        MAE, RMSE = CM.train(data.train.ratings, data.item_actors, data.item_country,
                             data.item_director, data.item_genres, LEARNRATE)
        logger.info('%d epoch the train MAE %f, RMSE %f' %
                    (i, MAE / train_len, RMSE / train_len))
        MAE, RMSE = CM.test(data.test.ratings, data.item_actors, data.item_country,
                            data.item_director, data.item_genres)
        logger.info('%d epoch the test MAE %f, RMSE %f\n' %
                    (i, MAE / test_len, RMSE / test_len))
        if i % 3 == 0:
            LEARNRATE *= DECAY
        LEARNRATE *= DECAY