import torch
from torch.utils.data import Dataset

class movie_lens(Dataset):
    def __init__(self, path, split='train'):
        self.path = path
        self.read_meta()

    def read_meta(self):
        self.all_users = []
        self.all_movies = []
        self.all_ratings = []
   
        with open(self.path, 'r') as f:
            for line in f:
                user, movie, rating, *others = map(int, line.split())
                self.all_users += [user]
                self.all_movies += [movie]
                self.all_ratings += [(rating-1)/4]

    def __len__(self):
        return len(self.all_ratings)
    
    def __getitem__(self, idx):
        sample = {
            'user': self.all_users[idx],
            'movie': self.all_movies[idx],
            'rating': self.all_ratings[idx]
        }

        return sample