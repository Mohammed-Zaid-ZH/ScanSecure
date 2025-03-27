document.addEventListener('DOMContentLoaded', function() {
    const video = document.getElementById('qr-video');
    const canvas = document.getElementById('qr-canvas');
    const startScannerBtn = document.getElementById('start-scanner');
    const qrUpload = document.getElementById('qr-upload');
    const resultList = document.getElementById('result-list');
    const videoContainer = document.querySelector('.video-container');
    
    let scannerActive = false;
    let stream = null;

    // Event listener to start/stop scanner
    startScannerBtn.addEventListener('click', toggleScanner);
    qrUpload.addEventListener('change', handleFileUpload);

    // Toggle scanner on/off
    function toggleScanner() {
        if (scannerActive) {
            stopScanner(); // Stop scanner if it's active
            startScannerBtn.innerHTML = '<i class="fas fa-camera"></i> Start Camera Scanner';
            videoContainer.style.display = 'none'; // Hide video container
        } else {
            startScanner(); // Start scanner if it's inactive
            startScannerBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Scanner';
            videoContainer.style.display = 'block'; // Show video container
        }
    }

    // Start camera scanner
    function startScanner() {
        if (stream) {
            console.log("Scanner already active, preventing multiple streams.");
            return; // Prevent multiple streams from being opened
        }
        navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })
            .then(function(s) {
                stream = s; // Assign the media stream
                video.srcObject = stream; // Set the stream to video element
                scannerActive = true; // Mark scanner as active
                video.play(); // Start the video
                scanFrame(); // Begin scanning for QR codes
            })
            .catch(function(err) {
                console.error("Error accessing camera: ", err);
                alert("Could not access the camera. Please check permissions.");
            });
    }

    // Stop the scanner and release the camera
    function stopScanner() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop()); // Stop all video tracks
            stream = null; // Clear the stream
        }
        scannerActive = false; // Mark scanner as inactive
        videoContainer.style.display = 'none'; // Hide the video container
    }

    // Function to continuously scan frames from the video
    function scanFrame() {
        if (!scannerActive) return;

        if (video.readyState === video.HAVE_ENOUGH_DATA) { // Ensure video is fully ready
            canvas.hidden = false;
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const code = jsQR(imageData.data, imageData.width, imageData.height);
            
            if (code) {
                displayResult([{ data: code.data, type: 'QR_CODE' }]);
                stopScanner(); // Stop the scanner after a QR code is found
                startScannerBtn.innerHTML = '<i class="fas fa-camera"></i> Start Camera Scanner';
                return; // Stop scanning after detecting the QR code
            }
        }
        
        requestAnimationFrame(scanFrame); // Keep scanning the next frame
    }

    // Handle file upload for QR code images
    function handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        fetch('/scan', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            displayResult(data.results);
        })
        .catch(error => {
            console.error('Error:', error);
            displayResult([], error.message);
        });
    }

    // Display results of QR code scan
    function displayResult(results, error = null) {
        resultList.innerHTML = '';

        if (error) {
            const errorItem = document.createElement('div');
            errorItem.className = 'result-item error';
            errorItem.textContent = `Error: ${error}`;
            resultList.appendChild(errorItem);
            return;
        }

        if (!results || results.length === 0) {
            const noResultItem = document.createElement('div');
            noResultItem.className = 'result-item';
            noResultItem.textContent = 'No QR codes found';
            resultList.appendChild(noResultItem);
            return;
        }

        results.forEach(result => {
            const resultItem = document.createElement('div');
            resultItem.className = 'result-item';

            const typeElement = document.createElement('p');
            typeElement.textContent = `Type: ${result.type}`;

            const urlElement = document.createElement('p');
            urlElement.className = 'result-url';

            if (isValidUrl(result.data)) {
                const link = document.createElement('a');
                link.href = result.data;
                link.textContent = result.data;
                link.target = '_blank';
                urlElement.appendChild(link);
            } else {
                urlElement.textContent = result.data;
            }

            resultItem.appendChild(typeElement);
            resultItem.appendChild(urlElement);
            resultList.appendChild(resultItem);
        });
    }

    // Check if a string is a valid URL
    function isValidUrl(string) {
        try {
            new URL(string);
            return true;
        } catch (_) {
            return false;
        }
    }

    // Stop scanner on page unload
    window.addEventListener('beforeunload', function() {
        stopScanner();
    });
});
