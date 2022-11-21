import nn
import uuid
import json
import asyncio
import logging
import argparse
import websockets
import numpy as np
import optimization.de as de


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

wslogger = logging.getLogger('websockets')
wslogger.setLevel(logging.WARN)


NN_ARCHITECTURE = [
    {'input_dim': 2, 'output_dim': 2, 'activation': 'swish'},
    {'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}
]


async def player_game(perceptron):
    identification = str(uuid.uuid4())[:8]
    async with websockets.connect('ws://localhost:8765/player') as websocket:
        await websocket.send(json.dumps({'cmd':'join', 'id':identification}))
        done = False
        while not done:
            data = json.loads(await websocket.recv())
            if data['evt'] == 'world_state':
                player = data['players'][identification]
                X = np.array([player['py'], player['v']])
                p = perceptron.predict(X)
                if p[0] >= 0.5:
                    await websocket.send(json.dumps({'cmd':'click'}))
            elif data['evt'] == 'done':
                return data['highscore']


def objective(p):
    #print(f'Parameters: {p}')
    perceptron = nn.NN(NN_ARCHITECTURE)
    # TODO: have to fix
    if type(p) == list:
        perceptron.update(np.array(p))
    else:
        perceptron.update(p)
    highscore = asyncio.run(player_game(perceptron))
    #print(f'Highscore {highscore}')
    return 30-highscore


def main(args):
    bounds = np.asarray([[-1.0, 1.0]]*nn.network_size(NN_ARCHITECTURE))
    best, _, debug = de.differential_evolution(objective, bounds,
    variant=args.v, n_iter=args.l, n_pop=args.p, n_jobs=args.p, cached=False, debug=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train agents')
    #parser.add_argument('-u', type=str, default='ws://localhost:8765/player', help='server url')
    parser.add_argument('-v', type=str, help='DE variant', default='best/1/bin')
    parser.add_argument('-l', type=int, help='number of loops (iterations)', default=10)
    parser.add_argument('-p', type=int, help='population size', default=10)
    args = parser.parse_args()

    main(args)