from .conversion_utils import partition_data


class meg_2d_parallel_map(object):
    def __init__(self, pp_degree, tp_degree):
        self.pp_degree = pp_degree
        self.tp_degree = tp_degree
        self.map = {}

    def simple_init(self):
        self.map = {
            self._make_key(i // self.tp_degree, i % self.tp_degree): [i]
            for i in range(self.pp_degree * self.tp_degree)
        }

    def add_data(self, pp_index, tp_index, data):
        self._validate_indices(pp_index, tp_index)
        assert type(data) is list

        key = self._make_key(pp_index, tp_index)
        if not key in self.map.keys():
            self.map[key] = []
        self.map[key] += data

    def get_data(self, pp_index=None, tp_index=None):
        self._validate_indices(pp_index, tp_index)
        pp_indices = list(range(
            self.pp_degree)) if pp_index is None else [pp_index]
        tp_indices = list(range(
            self.tp_degree)) if tp_index is None else [tp_index]

        result = []
        for i in pp_indices:
            for j in tp_indices:
                result += self.map[self._make_key(i, j)]

        return result

    def print_data(self, tag):
        print(f'{tag}')
        for key, value in self.map.items():
            print(f'{key} = {value}')

    def _validate_indices(self, pp_index, tp_index):
        assert pp_index is None or pp_index < self.pp_degree
        assert tp_index is None or tp_index < self.tp_degree

    def _make_key(self, i, j):
        return f'{i},{j}'


def _reshape_tp_dimension(old_2d_map, new_tp_degree):
    old_pp_degree = old_2d_map.pp_degree
    new_2d_map = meg_2d_parallel_map(old_pp_degree, new_tp_degree)
    for i in range(old_pp_degree):
        ranks_for_pp_index = old_2d_map.get_data(pp_index=i, tp_index=None)
        split_ranks = partition_data(ranks_for_pp_index, new_tp_degree)
        for j in range(new_tp_degree):
            new_2d_map.add_data(i, j, split_ranks[j])

    return new_2d_map


def _reshape_pp_dimension(old_2d_map, new_pp_degree):
    old_tp_degree = old_2d_map.tp_degree
    new_2d_map = meg_2d_parallel_map(new_pp_degree, old_tp_degree)
    for i in range(old_tp_degree):
        ranks_for_tp_index = old_2d_map.get_data(pp_index=None, tp_index=i)
        split_ranks = partition_data(ranks_for_tp_index, new_pp_degree)
        for j in range(new_pp_degree):
            new_2d_map.add_data(j, i, split_ranks[j])

    return new_2d_map


def reshape_meg_2d_parallel(old_pp_degree,
                            old_tp_degree,
                            new_pp_degree,
                            new_tp_degree,
                            verbose=False):
    assert new_pp_degree <= old_pp_degree
    assert new_tp_degree <= old_tp_degree

    old_2d_map = meg_2d_parallel_map(old_pp_degree, old_tp_degree)
    old_2d_map.simple_init()
    if verbose:
        old_2d_map.print_data(f'original_2d_map:')

    if old_tp_degree != new_tp_degree:
        new_tp_map = _reshape_tp_dimension(old_2d_map, new_tp_degree)
    else:
        new_tp_map = old_2d_map
    if verbose:
        new_tp_map.print_data(f'after_tp_reshape:')

    if old_pp_degree != new_pp_degree:
        final_map = _reshape_pp_dimension(new_tp_map, new_pp_degree)
    else:
        final_map = new_tp_map

    if verbose:
        final_map.print_data(f'final_2d_map:')

    return final_map
