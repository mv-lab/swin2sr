class LevelIndexIterator:
    def __init__(self, index_list) -> None:
        self._index_list = index_list
        self._N = len(self._index_list)
        self._cur_position = 0

    def next(self):
        output_pos = self._cur_position
        self._cur_position += 1
        self._cur_position = self._cur_position % self._N
        return self._index_list[output_pos]
