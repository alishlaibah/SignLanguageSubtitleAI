import React from "react";
import WebcamFeed from "./components/webcamfeed";

function App() {
    return (
        <div style={{textAlign: "center", backgroundColor: "black", color: "white"}}>
            <h1>Sign Language Subtitle AI MVP</h1>
            <WebcamFeed />
        </div>
    );
}

export default App;