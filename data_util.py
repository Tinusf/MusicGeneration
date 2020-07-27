import numpy as np


class DataUtil:
    def __init__(self):
        self.elem_to_index = {}
        self.index_to_elem = {}
        self.i = 0

    def index(self, elem):
        if elem in self.elem_to_index:
            return self.elem_to_index[elem]
        else:
            self.elem_to_index[elem] = self.i
            self.i += 1
            return self.i

    def transform_data_index(self, x, y):
        new_y = []
        for elem in y:
            index = self.index(elem)
            new_y.append(index)

        new_x = []
        for row in x:
            new_x_row = []
            for elem in row:
                index = self.index(elem)
                new_x_row.append(index)
            new_x.append(new_x_row)

        self.index_to_elem = {v: k for k, v in self.elem_to_index.items()}
        return np.array(new_x), np.array(new_y)
