
if __name__ == "__main__":
    list_ = [3, 4, 5, 1, 6, 7, 23, 56, 90]
    list_size = len(list_)

    new_size = int(0.7 * list_size)
    print(list_[:new_size])