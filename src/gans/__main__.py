from gans.data import splits, load_data

def data():
    splits()

def main():
    load_data(True)

if __name__ == "__main__":
    main()