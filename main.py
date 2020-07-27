import music_util


def main():
    print("Hey")
    arr = music_util.load_all_midi_files()
    frequent_notes = music_util.get_frequent_notes(arr)
    arr_filtered = music_util.filter_frequent_notes(arr, frequent_notes)
    x, y = music_util.create_training_dataset(arr_filtered)
    print(arr)


if __name__ == '__main__':
    main()
