import music_util


def main():
    print("Hey")
    arr = music_util.load_all_midi_files()
    print(arr)


if __name__ == '__main__':
    main()
