import React, { useRef, useEffect } from 'react';
import { Hands } from '@mediapipe/hands';

function WebcamFeed() {
    const videoRef = useRef(null); // grab HTML video element
    const canvasRef = useRef(null); // canvas element
    const handsRef = useRef(null); // MediaPipe hands model instance

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

    useEffect(() => {
        if (handsRef.current) return;

        const hands = new Hands({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
            },
        });

        hands.setOptions({
            maxNumHands: 1,
            minDetectionConfidence: 0.9,
            minTrackingConfidence: 0.9,
            modelComplexity: 1,
        });

        hands.onResults((results) => {
            if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
                const firstHand = results.multiHandLandmarks[0];
                console.log("Hand landmakrs:", firstHand);
            } else {
                console.log("No hand detected");
            }
        });


        // start a loop that continuosly sends the webcame image to the model 
        let animationFrameId;

        async function sendFramToHands() {
            const videoEl = videoRef.current;
            const handsModel = handsRef.current;

            if (videoEl && videoEl.readyState === 4 && handsModel) {
                await handsModel.send({image: videoEl});
            }

            // ask broswer to call this function again on next frame
            animationFrameId = requestAnimationFrame(sendFramToHands);
        }

        animationFrameId = requestAnimationFrame(sendFramToHands);

        handsRef.current = hands;

        return () => {
            cancelAnimationFrame(animationFrameId);
            if (handsRef.current) {
                handsRef.current.close();
            }
        };
    }, []);



    return (
        <div
            style={{position: "relative", width:"100%", maxWidth: "900px", margin: "0 auto", border: "2px solid black", padding: "8px", boxSizing: "border-box"}}
            >
            <video ref={videoRef} autoPlay playsInline style={{width: "100%", height: "auto", display: "block", backgroundColor: "black",}} />
            <canvas ref={canvasRef} style={{position: "absolute", left: 0, top: 0, width:"100%", height: "100%", pointerEvents: "none"}}
            />
        </div>
    );
}

export default WebcamFeed;