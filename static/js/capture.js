// This file handles live capture functionality for emotion detection from the user's webcam.

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');
const captureButton = document.getElementById('captureButton');

navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
        video.srcObject = stream;
        video.play();
    })
    .catch((error) => {
        console.error("Error accessing webcam: ", error);
    });

captureButton.addEventListener('click', () => {
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageData = canvas.toDataURL('image/png');
    
    // Send the captured image to the server for emotion detection
    fetch('/detect-emotion', {
        method: 'POST',
        body: JSON.stringify({ image: imageData }),
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        // Handle the response from the server
        console.log('Emotion detected: ', data.emotion);
    })
    .catch((error) => {
        console.error("Error sending image to server: ", error);
    });
});