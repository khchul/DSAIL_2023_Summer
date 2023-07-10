import torch
import torch.nn as nn
import torch.nn.functional as F

import math

ratings_loss = lambda x, y: torch.mean((x - y)**2)

class WRMF():
    def __init__(self, path, args, split='train'):
        self.path = path
        self.N_user = args.N_user
        self.N_movie = args.N_movie

        self.alpha = args.alpha
        self.eps = args.eps
        self.l = args.l
        self.f = args.f

        self.X = torch.rand(size=(self.N_user, self.f), requires_grad=False)
        self.Y = torch.rand(size=(self.N_movie, self.f), requires_grad=False)

        self.read_meta()

    def read_meta(self):
        self.all_users = []
        self.all_movies = []
        self.all_ratings = []

        self.C = torch.zeros(size=(self.N_user, self.N_movie), requires_grad=False)
        self.P = torch.zeros(size=(self.N_user, self.N_movie), requires_grad=False)
   
        with open(self.path, 'r') as f:
            for line in f:
                user, movie, rating, *others = map(int, line.split())
                self.all_users += [user]
                self.all_movies += [movie]
                self.all_ratings += [rating]

        self.all_users = torch.tensor(self.all_users)
        self.all_movies = torch.tensor(self.all_movies)
        self.all_ratings = torch.Tensor(self.all_ratings)

        self.C[self.all_users-1, self.all_movies-1] =  1 + self.alpha * torch.log10(
                                                            1 + self.all_ratings / self.eps
        )
        self.P[self.all_users-1][self.all_movies-1] = 1

    def opt_X(self):
        yty = torch.mm(self.Y.T, self.Y)
        c = self.C.unsqueeze(1) * torch.eye(self.N_movie).unsqueeze(0) # c : [N_user * N_movie * N_movie]
        ycy = yty + torch.matmul(torch.matmul(self.Y.T, c - torch.eye(self.N_movie)), self.Y) # [N_user * f * f]

        inv = torch.linalg.inv(ycy + self.l * torch.eye(self.f)) # inv : [N_user * f * f]
        ycp = torch.matmul(self.Y.T, torch.matmul(c, self.P[..., None])) # ycp : [N_user * f * 1]

        self.X = torch.matmul(inv, ycp).squeeze(-1)

    def opt_Y(self):
        xtx = torch.mm(self.X.T, self.X)
        c = self.C.T.unsqueeze(1) * torch.eye(self.N_user).unsqueeze(0)
        xcx = xtx + torch.matmul(torch.matmul(self.X.T, c - torch.eye(self.N_user)), self.X)

        inv = torch.linalg.inv(xcx + self.l * torch.eye(self.f))
        xcp = torch.matmul(self.X.T, torch.matmul(c, self.P.T[..., None]))

        self.Y = torch.matmul(inv, xcp).squeeze(-1)

    def get_matrices(self):
        sample = {
            'X': self.X,
            'Y': self.Y,
        }

        return sample
    
    def calc_pref(self):
        pred = torch.mm(self.X, self.Y.T)

        return ratings_loss(pred[self.all_users-1, self.all_movies-1], self.all_ratings)