if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train agents')
    parser.add_argument('-u', type=str, default='ws://localhost:8765/player', help='server url')
    args = parser.parse_args()