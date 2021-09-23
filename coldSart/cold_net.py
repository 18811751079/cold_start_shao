from data import Data_coldStart
import numpy as np
import logging

DATAPATH = r'E:\PPPP___Python\test_data\hetrec2011-movielens-2k-v2'
EPOCH = 10
LEARNRATE = 0.02
DECAY = 0.9
DIM = 10

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")

log_writer = logging.FileHandler('cold_net.log', 'a')
log_writer.setLevel(logging.DEBUG)
log_writer.setFormatter(formatter)
log_print = logging.StreamHandler()
log_print.setLevel(logging.INFO)
log_print.setFormatter(formatter)

logger.addHandler(log_writer)
logger.addHandler(log_print)


def sigmoid(x):
    return 1/(1+np.exp(-x))


def vector_sigmoid(vector):
    sig_vec = np.zeros_like(vector)
    length = vector.shape[0]
    for i in range(length):
        sig_vec[i] = round(sigmoid(vector[i]), 3)
    return sig_vec

class cold_model():
    def __init__(self, act_num, country_num, director_num,
                 genre_num, users, dim=10):

        self.dim = dim
        self.act_num = act_num
        self.cty_num = country_num
        self.drt_num = director_num
        self.genre_num = genre_num
        logger.debug('dim=%d, act_num=%d, cty_num=%d, drt_num=%d, genre_num=%d'%(dim,
                                                                                 act_num, country_num,
                                                                                 director_num, genre_num))
        self.actor_w = np.random.rand(dim, act_num)
        self.country_w = np.random.rand(dim, country_num)
        self.director_w = np.random.rand(dim, director_num)
        self.genre_w = np.random.rand(dim, genre_num)


        self.actor_v = np.random.rand(dim)
        self.country_v = np.random.rand(dim)
        self.director_v = np.random.rand(dim)
        self.genre_v = np.random.rand(dim)

        self.P = {}
        self.mk_user_mat(users)
        self.init_gradient_param()

    def mk_user_mat(self, users):
        for user in users:
            self.P[user] = np.random.rand(4)

    def prediction(self, actor, country, director, genre, user):
        # vector shape = dim
        self.actor_f = vector_sigmoid(np.dot(self.actor_w, actor))
        self.country_f = vector_sigmoid(np.dot(self.country_w, country))
        self.director_f = vector_sigmoid(np.dot(self.director_w, director))
        self.genre_f = vector_sigmoid(np.dot(self.genre_w, genre))

        # scalar shape = 1
        actor_n = np.dot(self.actor_f,self.actor_v)
        country_n = np.dot(self.country_f, self.country_v)
        director_n = np.dot(self.director_f, self.director_v)
        genre_n = np.dot(self.genre_f, self.genre_v)

        self.vector_n = np.array([actor_n, country_n, director_n, genre_n])

        return round(np.dot(self.P[user], self.vector_n), 2)

    def init_gradient_param(self):
        '''

        :param users: 用户种类
        :description: 用来暂存梯度下降更新的值
        '''
        self.ud_P = np.zeros(4)

        self.ud_actor_w = np.zeros_like(self.actor_w)
        self.ud_country_w = np.zeros_like(self.actor_w)
        self.ud_director_w = np.zeros_like(self.director_w)
        self.ud_genre_w = np.zeros_like(self.genre_w)

        self.ud_actor_v = np.zeros_like(self.actor_v)
        self.ud_country_v = np.zeros_like(self.country_v)
        self.ud_director_v = np.zeros_like(self.director_v)
        self.ud_genre_v = np.zeros_like(self.genre_v)

    def backward(self, user, eui, actor, country, director, genre, maxgrad=3):
        for i in range(4):
            self.ud_P[i] = -eui * self.actor_f[i]

        for t in range(self.dim):
            self.ud_actor_v[t] = -eui * self.P[user][0] * self.actor_f[t]
            self.ud_country_v[t] = -eui * self.P[user][1] * self.country_f[t]
            self.ud_director_v[t] = -eui * self.P[user][2] * self.director_f[t]
            self.ud_genre_v[t] = -eui * self.P[user][3] * self.genre_f[t]

        for t in range(self.dim):
            for k in range(self.act_num):
                self.ud_actor_w[t][k] = -eui * self.P[user][0] * self.actor_f[t] * (1 - self.actor_f[t]) * actor[k]


            for k in range(self.cty_num):
                self.ud_country_w[t][k] = -eui * self.P[user][1] * self.country_f[t] * (1 - self.country_f[t]) * country[k]

            for k in range(self.drt_num):
                self.ud_director_w[t][k] = -eui * self.P[user][2] * self.director_f[t] * (1 -self.director_f[t]) * director[k]

            for k in range(self.genre_num):
                self.ud_genre_w[t][k] = -eui * self.P[user][3] * self.genre_f[t] * (1 - self.genre_f[t]) * genre[k]

    def update(self, user, lr=0.01):
        for i in range(4):
            self.P[user] -= lr * self.ud_P[i]

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

    def train(self, epoch, data, actors, countries, directors, genres, lr):
        data_len = len(data)
        for i in range(epoch):
            MAE = 0
            RMSE = 0
            for user, item, rating in data:

                eui = rating - self.prediction(actors[item], countries[item],
                                               directors[item], genres[item], user)
                # print(eui)
                self.backward(user, eui, actors[item], countries[item],
                              directors[item], genres[item])
                self.update(user, lr)

                lr *= DECAY
                MAE += abs(eui)
                # if i%1000 == 0:
                #     print(abs(eui))
                # print('%d/%d'%(i, data_len))
                # RMSE += eui**2
                # print(abs(eui))
            logger.info('%d epoch the train MAE %f'%(i, MAE/data_len))
            # logger.info('the train RMSE %f'%(RMSE/data_len))

    def save_param(self):
        path = 'model'
        import os

        if not os.path.exists(path):
            os.makedirs(path)

        self.writefile(path+'actors_w.param', self.actor_w)
        self.writefile(path+'country_w.param', self.country_w)
        self.writefile(path+'director_w', self.director_w)
        self.writefile(path+'genre_w', self.genre_w)

        self.writefile(path+'actor_v', self.actor_v)
        self.writefile(path+'country_v', self.country_v)
        self.writefile(path + 'director_v', self.director_v)
        self.writefile(path + 'genre_v', self.genre_v)

        self.writefile(path + 'P', self.P)


        print('saving the model is successful!')

    def writefile(self, filename, param):
        with open(filename, 'w') as f:
            f.writelines(str(param))


if __name__ == '__main__':

    data = Data_coldStart()
    data.get_actor(DATAPATH + r'\movie_actors.dat')
    data.get_country(DATAPATH + r'\movie_countries.dat')
    data.get_director(DATAPATH + r'\movie_directors.dat')
    data.get_genres(DATAPATH + r'\movie_genres.dat')
    data.get_data(DATAPATH + r'\user_ratedmovies.dat')

    actor_num, country_num, director_num, genre_num = data.size()
    CM = cold_model(actor_num, country_num, director_num, genre_num, data.train.users, DIM)
    CM.train(EPOCH, data.train.ratings, data.item_actors, data.item_country,
             data.item_director, data.item_genres, LEARNRATE)
