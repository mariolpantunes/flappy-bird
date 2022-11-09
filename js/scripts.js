const socket = new WebSocket('ws://localhost:8765/viewer');

let px = 0;
let py = 0;
let bx = 0;
let scrollSpeed = 5;


// Connection opened
socket.addEventListener('open', (event) => {
    socket.send(JSON.stringify({'cmd':'join'}));
});

// Listen for messages
socket.addEventListener('message', (event) => {
    const data = JSON.parse(event.data);
    console.log('Message from server ', data);
    px = data['px'];
    py = data['py'];
});

const bird_img = new Image();
const back = new Image();
let i = 0;


function startAnimating(fps) {
    fpsInterval = 1000 / fps;
    then = Date.now();
    startTime = then;
    console.log(startTime);
    animate();
}

function animate() {

    // request another frame

    requestAnimationFrame(animate);

    // calc elapsed time since last loop

    now = Date.now();
    elapsed = now - then;

    // if enough time has elapsed, draw the next frame

    if (elapsed > fpsInterval) {

        // Get ready for next frame by setting then=now, but also adjust for your
        // specified fpsInterval not being a multiple of RAF's interval (16.7ms)
        then = now - (elapsed % fpsInterval);

        // Put your drawing code here
        draw();

    }
}


function init() {
    bird_img.src = 'data/bird3.png';
    back.src = 'data/back3.png'
    
    startAnimating(10);
}

function draw() {
    const ctx = document.getElementById('canvas').getContext('2d');
    ctx.globalCompositeOperation = 'destination-over';
    ctx.clearRect(0, 0, 400, 400); // clear canvas
    //ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
    //ctx.strokeStyle = 'rgba(0, 153, 255, 0.4)';

    ctx.drawImage(bird_img, i, 0, 85, 60, px, py, 85, 60);
    i = (i+85)%255;

    // Draw Infinitely Scrolling Background
    // draw image 1
    ctx.drawImage(back, 400-bx, 0);
 
    // draw image 2
    ctx.drawImage(back, -bx, 0);

    // update image height
    bx += scrollSpeed;

    //resetting the images when the first image entirely exits the screen
    if (bx == 400) {bx = 0;}

    

    console.log('px '+px+' py '+py);
    //window.requestAnimationFrame(draw);
}

init();