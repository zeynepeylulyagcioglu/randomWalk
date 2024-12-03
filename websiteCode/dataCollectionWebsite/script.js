document.addEventListener('DOMContentLoaded', () => {
  let position = 0; // Track current position
  let trajectory = []; // Record trajectory
  let timestamps = []; // Record timestamps for each step
  let isRecording = false; // Toggle recording

  // HTML Elements
  const stickFigure = document.getElementById('stick-figure');
  const stepsTaken = document.getElementById('steps-taken');
  const positionDisplay = document.getElementById('position');
  const startRecordingButton = document.createElement('button'); // Button to start/stop recording

  // Add recording button to the controls
  startRecordingButton.textContent = "Start Recording Run";
  startRecordingButton.style.marginTop = "20px";
  startRecordingButton.style.padding = "10px 20px";
  document.querySelector(".controls").appendChild(startRecordingButton);

  // Function to toggle recording
  function toggleRecording() {
    isRecording = !isRecording;
    if (isRecording) {
      position = 0; // Reset position
      trajectory = []; // Clear previous trajectory
      timestamps = []; // Clear previous timestamps
      startRecordingButton.textContent = "Stop Recording Run";
    } else {
      console.log("Recording Stopped. Data:", { trajectory, timestamps });
      downloadData({ trajectory, timestamps });
      startRecordingButton.textContent = "Start Recording Run";
    }
  }

  // Function to move stick figure and record data
  function moveCharacter(direction) {
    position += direction;
    trajectory.push(position);
    timestamps.push(Date.now()); // Record the timestamp

    // Update stick figure position on screen
    const offset = position * 10; // 10px per step
    stickFigure.style.left = `calc(50% + ${offset}px)`;

    // Update stats
    stepsTaken.textContent = trajectory.length;
    positionDisplay.textContent = position;
  }

  // Listen for keyboard input (arrow keys)
  document.addEventListener('keydown', (event) => {
    if (isRecording) {
      if (event.key === 'ArrowRight') {
        moveCharacter(1); // Right arrow moves forward
      } else if (event.key === 'ArrowLeft') {
        moveCharacter(-1); // Left arrow moves backward
      }
    }
  });

  // Function to download data as JSON
  function downloadData(data) {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `run_data_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  // Attach event listener to the recording button
  startRecordingButton.addEventListener('click', toggleRecording);
});
