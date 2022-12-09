const socket_player = new WebSocket('ws://localhost:8765/player');

let click = false;

function onMouseClick() {
    click = true;
}

// Connection opened
socket_player.addEventListener('open', (event) => {
    socket_player.send(JSON.stringify({'cmd':'join', 'id': 0}));
    const canvas = document.getElementById('canvas');
    canvas.addEventListener('mousedown', onMouseClick);
});

// Listen for messages
socket_player.addEventListener('message', (event) => {
    const data = JSON.parse(event.data);
    
    if(data.evt == 'world_state') {
        // Update the world state
        if (click == true) {
            socket_player.send(JSON.stringify({'cmd':'click'}));
            click = false;
        }
    }
});