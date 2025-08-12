import React from 'react'
import Webcam from 'react-webcam'
import {useRef, useEffect} from 'react';
import { Hands } from "@mediapipe/hands";
import { drawConnectors, drawLandmarks } from "@mediapipe/drawing_utils";
import { Camera } from "@mediapipe/camera_utils";





function WebcamFeed() {
    const videoRef = useRef(null); // grab HTML video element

    useEffect(() => {
        navigator.mediaDevices.getUserMedia({video : true})
        .then((stream) => { // gives live video feed
            if (videoRef.current) {
                videoRef.current.srcObject = stream; // connect the stream to the video element on the screen
            }
        })
        .catch((err) => {
            console.error("Error accessing webcam: ", err);
        });
    }, []); // run once on mount

    const canvasRef = useRef(null); 



    return (
        <div>
            {/* show video */}
            <video ref={videoRef} autoPlay playsInline style={{width: "640px", height: "480px", border: "1px black"}} />
        </div>
    );
}

export default WebcamFeed;