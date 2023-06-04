import numpy as np

class MM_Augmentation:
    def __init__(self, tools):

        self.tools_methods: list = [x for x in dir(tools) if "__" not in x]
        self.tools = tools

    def run(self, data):
        new_data = []
        for row_str in data:
            new_data.append(self.process_row(row_str))

        return new_data

    def process_row(self, row):
        size_of_transformations: int = 1

        for i in range(size_of_transformations):
            augmented: list = self.augmentate(list(row))

            return augmented[0]

    def augmentate(self, string: list) -> list:

        augmented: list = []

        random_size_tansform: int = np.random.randint(1, 3)

        num_tranform_func: list = np.random.choice(
            self.tools_methods, size=random_size_tansform
        )
        series_string: list = string

        for func in num_tranform_func:

            transform_func = getattr(self.tools, func)

            if "".join(series_string):
                series_string = transform_func(series_string)

            else:
                break

        to_append_series: str = "".join(series_string)
        augmented.append(to_append_series)

        return augmented
