#!/usr/bin/env python


__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import uuid
import json
import asyncio
import logging
import argparse
import websockets
import numpy as np
import src.nn as nn


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

wslogger = logging.getLogger('websockets')
wslogger.setLevel(logging.WARN)


def load_data(path:str) -> tuple:
    '''
    Load a json encoded model.

    Args:
        path(str): the location of the model
    
    Returns:
        tuple: model definition and parameters
    '''
    with open(path, 'rb') as f:
        data = json.load(f)
        return data['model'], np.asarray(data['parameters'])


async def player_game(url:str, model:nn.NN) -> float:
    '''
    Player main loop.

    Receives the world dump, and decides if it click of not.
    Always shares the network information.

    Args:
        url (str): the server url
        model (nn.NN): the NN model
    
    Returns:
        float: highscore
    '''
    identification = str(uuid.uuid4())[:8]
    networkLayer = model.layers()
    async with websockets.connect(f'{url}/player') as websocket:
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
                
                c =  pipe['py_t'] + pipe['py_b'] / 2
                X = [player['py'], player['v'], c, pipe['px']]
                p, activations = model.predict_activations(X)
                if p[0] >= 0.5:
                    await websocket.send(json.dumps({'cmd':'click'}))
                await websocket.send(json.dumps({'cmd':'neural_network', 'neural_network':{'networkLayer':networkLayer,'activations':activations}}))
            elif data['evt'] == 'done':
                done = True
                return data['highscore']


def main(args: argparse.Namespace) -> None:
    '''
    Main method.

    Args:
        args (argparse.Namespace): the program arguments
    '''
    model_description, parameters = load_data(args.l)
    model = nn.NN(model_description)
    model.update(parameters)

    highscore = asyncio.run(player_game(args.p, model))
    logger.info(f'Highscore: {highscore}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train agents')
    parser.add_argument('-u', type=str, help='server url', default='ws://localhost:8765')
    parser.add_argument('-l', type=str, help='load a player neural network', required=True)
    args = parser.parse_args()

    main(args)
