const socket = new WebSocket('ws://localhost:8765/viewer');
const socket_player = new WebSocket('ws://localhost:8765/player');

let px = 158;
let py = 140;
let bx = 0;
let scrollSpeed = 5;
let click = false;
let fps = 0;
let lastTime = performance.now();

// Connection opened
socket.addEventListener('open', (event) => {
    socket.send(JSON.stringify({'cmd':'join'}));
});

// Listen for messages
socket.addEventListener('message', (event) => {
    const data = JSON.parse(event.data);
    console.log('Message from server ', data);
    
    if(data.evt == 'world_state') {
        // Update the world state
        if(data['players'].length > 0) {
        player = data['players'][0]
        px = player.px;
        py = player.py;
        }
        // Draw the scene
        requestAnimationFrame(draw);
    }
});


function onMouseClick() {
    console.log('onMouseClick');
    click = true;
}


// Connection opened
socket_player.addEventListener('open', (event) => {
    socket_player.send(JSON.stringify({'cmd':'join'}));
    const canvas = document.getElementById('canvas');
    canvas.addEventListener('mousedown', onMouseClick);
});

// Listen for messages
socket_player.addEventListener('message', (event) => {
    const data = JSON.parse(event.data);
    console.log('Message from server ', data);
    
    if(data.evt == 'world_state') {
        // Update the world state
        if (click == true) {
            socket_player.send(JSON.stringify({'cmd':'click'}));
            click = false;
        }
    }
});

const bird_img = new Image();
const back = new Image();
let i = 0;


function init() {
    bird_img.src = 'data/bird4.png';
    back.src = 'data/back5.png'
}

function draw() {
    const ctx = document.getElementById('canvas').getContext('2d');
    ctx.globalCompositeOperation = 'destination-over';
    ctx.clearRect(0, 0, 400, 400); // clear canvas
    //ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
    //ctx.strokeStyle = 'rgba(0, 153, 255, 0.4)';

    // write FPS
    ctx.font = '18px Arial';
    ctx.fillText('FPS: '+fps, 0, 18);

    // Draw Tubes

    // Draw players
    ctx.drawImage(bird_img, i, 0, bird_img.width/3, bird_img.height, px, py, bird_img.width/3, bird_img.height);
    i = (i+(bird_img.width/3))%bird_img.width;
    

    // Draw Infinitely Scrolling Background
    // draw image 1
    ctx.drawImage(back, back.width-bx, 0);
    // draw image 2
    ctx.drawImage(back, -bx, 0);
    // update image height
    bx += scrollSpeed;
    //resetting the images when the first image entirely exits the screen
    if (bx >= back.width) {bx = 0;}

    // Update FPS counter
    const currentTime = performance.now();
    fps = Math.round(1000 / (currentTime - lastTime));
    lastTime = currentTime;

}

init();