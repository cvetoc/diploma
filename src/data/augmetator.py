import numpy as np
import string
from typing import NewType

class Augmetator_func_tool:
    __char = NewType('char', str)

    __string_ru: str = (
        'абвгдеёжзийклмнопрстуфхцчшщъыьэюя АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
    )

    __chars: list = list(
        "".join(
            [string.punctuation, string.digits, __string_ru]
        )
    )

    @staticmethod
    def char_swap(list_char: list[__char]) -> list[__char]:
        """
        Меняем два случайных символа местами
        """
        x, y = np.random.randint(len(list_char), size=2)
        list_char[x], list_char[y] = list_char[y], list_char[x]

        return list_char

    @staticmethod
    def char_del(list_char: list[__char]) -> list[__char]:
        """
        Удаление случайного символа
        """
        x: int = np.random.randint(len(list_char))
        list_char[x] = ""

        return list_char

    @classmethod
    def insert_random_chars(self, list_char: list[__char]) -> list[__char]:
        """
        Вставляем случайные символы в случайное место
        """
        x: int = np.random.randint(len(list_char))
        size = np.random.randint(1, 4)
        char_: str = self.__generate_random_string(size=size)[0]
        list_char.insert(x, char_)
        return list_char

    @classmethod
    def __generate_random_string(self, size: int = 2) -> list[__char]:
        """
        Случайный набор символов
        """
        string: list = list(np.random.choice(self.__chars, size=size))
        return string

    @staticmethod
    def swap_words(list_char: list[__char]) -> list[__char]:
        """
        Меняем два случайных слова местами
        """
        list_char = "".join(list_char).split()

        if len(list_char):
            x, y = np.random.randint(len(list_char), size=2)
            list_char[x], list_char[y] = list_char[y], list_char[x]

        return list(" ".join(list_char))

    @staticmethod
    def low_upp_random(list_char: list[__char]) -> list[__char]:
        """
        Случайно приводит слово к верхниму или нижниму регистру
        """
        list_char = "".join(list_char)

        if np.random.randint(2):
            list_char = list_char.upper()
        else:
            list_char = list_char.lower()

        return list(list_char)

    @staticmethod
    def connect_words(list_char: list[__char]) -> list[__char]:
        """
        удалить случайный пробел
        """
        spaces: list = [i for i, char in enumerate(list_char) if char == " "]

        if spaces:
            remove_index: int = np.random.choice(spaces)
            del list_char[remove_index]

        return list_char

    @staticmethod
    def remove_word(list_char: list[__char]) -> list[__char]:
        """
        Удалаем случайное слово
        """
        list_char = "".join(list_char).split()
        if len(list_char) > 2:
            remove_index: int = np.random.choice(len(list_char))
            del list_char[remove_index]

        return list(" ".join(list_char))