import React, { useState } from 'react';
import axios from 'axios';

function App() {
    const [selectedFile, setSelectedFile] = useState(null);
    const [result, setResult] = useState(null);

    const handleFileChange = (event) => {
        setSelectedFile(event.target.files[0]);
    };

    const handleUpload = async (endpoint) => {
        if (!selectedFile) return;
        const formData = new FormData();
        formData.append("file", selectedFile);

        try {
            const response = await axios.post(`http://127.0.0.1:5000/${endpoint}`, formData);
            setResult(response.data);
        } catch (error) {
            console.error("Error detecting:", error);
        }
    };

    return (
        <div className="min-h-screen flex flex-col items-center justify-center bg-gray-100">
            <h1 className="text-3xl font-bold mb-4">Image Forensics & Deepfake Detector</h1>
            <input type="file" onChange={handleFileChange} className="mb-4" />
            <div className="flex gap-4">
                <button onClick={() => handleUpload("detect_forgery")} className="bg-blue-500 text-white p-2 rounded">Detect Forgery</button>
                <button onClick={() => handleUpload("detect_deepfake")} className="bg-red-500 text-white p-2 rounded">Detect Deepfake</button>
            </div>
            {result && (
                <div className="mt-4 p-4 bg-white shadow-md rounded">
                    <p><strong>File:</strong> {result.file}</p>
                    <p><strong>Prediction:</strong> {result.prediction}</p>
                    <p><strong>Confidence:</strong> {(result.confidence * 100).toFixed(2)}%</p>
                </div>
            )}
        </div>
    );
}

export default App;
