#!/usr/bin/env python


import nn
import uuid
import json
import pickle
import asyncio
import logging
import argparse
import websockets


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

wslogger = logging.getLogger('websockets')
wslogger.setLevel(logging.WARN)


def load_data(path:str):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        return data['model'], data['parameters']


async def player_game(model):
    identification = str(uuid.uuid4())[:8]
    async with websockets.connect('ws://localhost:8765/player') as websocket:
        await websocket.send(json.dumps({'cmd':'join', 'id':identification}))
        done = False
        while not done:
            data = json.loads(await websocket.recv())
            if data['evt'] == 'world_state':
                player = data['players'][identification]
                # find closest pipe
                closest_pipe = None
                for pipe in data['pipes']:
                    if pipe['px']+60 > player['px']:
                        closest_pipe = pipe
                        break
                
                X = [player['py'], player['v'], pipe['py_t'], pipe['py_b']]
                p = model.predict(X)
                if p[0] >= 0.5:
                    await websocket.send(json.dumps({'cmd':'click'}))
            elif data['evt'] == 'done':
                return data['highscore']


def main(args):
    model_description, parameters = load_data(args.l)
    model = nn.NN(model_description)
    model.update(parameters)

    highscore = asyncio.run(player_game(model))
    logger.info(f'Highscore: {highscore}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train agents')
    parser.add_argument('-l', type=str, help='load a player neural network')
    args = parser.parse_args()

    main(args)